---
title: "06 · 矩阵乘法：tiling 与 kernel 融合"
lecture: 6
---

# 矩阵乘法：tiling 与 kernel 融合

**矩阵乘法（matmul）是深度学习的“面包与黄油”**——它已经被优化到极致，也是各种系统优化的核心战场。本讲给它加一点小 twist：做 **matmul + ReLU**，因为线性层后面接激活函数是真实世界里天天发生的事（这也是为什么要做 kernel 融合）。

## 朴素做法：为什么算术强度只有 $O(1)$

设 $A$ 是 $M \times K$、$B$ 是 $K \times N$，输出 $C = A \times B$ 是 $M \times N$。朴素做法是：**固定一个输出元素 $(m, n)$，然后对每个 $k$ 迭代**——

1. 从 HBM 读 $A[m, k]$ 与 $B[k, n]$；
2. 相乘并累加（multiply-accumulate）；
3. 最后把 $C[m, n]$ 写回 HBM。

这是一个**正确的** matmul kernel。问题出在读写次数：**对每一对 $(m, n)$ 和每一个 $k$，都要从 HBM 读两个数**，所以总共是 $O(MKN)$ 次读、$O(MN)$ 次写。回想第二讲讲过的**算术强度（arithmetic intensity）**——运算次数除以搬运的字节数：运算量是 $O(MKN)$，搬运量也是 $O(MKN)$，于是**算术强度是 $O(1)$**，这非常不理想。

再仔细观察，你会发现**大量冗余的读取**：假设要算 $C_4$（需要 $A_4, A_5, A_6$），而算 $C_5$ 时又要把 $A_4, A_5, A_6$ **再读一遍**。如果这些数据只读一次，能省下多少读写？

## 理想化做法：全量载入 shared memory

于是自然想到用 **shared memory**：**把 $A$ 和 $B$ 全部载入 shared memory**，然后在上面一口气算完 $C$。这样读写从三次方降为二次方——读 $MK + KN$ 次、写 $MN$ 次，算术强度达到 $O(N)$（对 $N \times N$ 的方阵而言），这正是第二讲所说的“可以指望的理想值”。

问题只有一个：**$A$ 和 $B$ 通常太大，放不进 shared memory**。

## Tiling：全局像朴素、局部像理想

折中方案就是经典的 **tiling（分块）**：**尽量多地把数据塞进 shared memory**。宏观上看，它和朴素做法一样（逐块遍历）；微观上看，它又像理想化做法（在 shared memory 上算 tile 乘法）。

![](/lectures/06/gemm_tiled.png)

具体做法如下：

1. 把输出矩阵 $C$ 划分成**输出 tile（output tile）**，每个 tile 由一个**线程块**负责；
2. 固定一个输出 tile，对每一对（$A$ 的**行 tile**，$B$ 的**列 tile**）：
   - 把对应的 $A$ tile 与 $B$ tile 从 HBM **载入 shared memory**；
   - 在 shared memory 上做这两个 tile 的矩阵乘法；
   - 把结果**累加**进部分和（partial sum，也在 shared memory 里）；
3. 扫完整个行/列后，把输出 tile 写回 HBM。

于是不同的输出 tile 由不同的线程块完全独立地计算。tiling 之后，**算术强度上升到 $O(\text{tile\_size})$**——虽然一般达不到理想化的 $O(N)$（那要求把整个矩阵装进 shared memory），但只要 tile 够大，就已经相当不错了。

### Bonus：kernel 融合

既然 kernel 都写了，**顺便**可以做一件事：如果想在结果上套一个**逐元素激活函数**（比如 $\mathrm{GeLU}(A \times B)$ 或 ReLU），在把输出 tile 写回 HBM **之前**直接加一行即可——这就是**kernel 融合（kernel fusion）**：不用多启动一个 kernel、不用把中间结果搬进搬出 HBM。这正是上面给 matmul 加 ReLU 的用意。

## 实现：先回顾 stride

进入代码之前，先复习一个概念——**stride（步长）**。一个 tensor 是多维数组，但在内存里是**线性化（linearized）**存储的。stride 告诉我们如何把多维下标（如“第 $r$ 行第 $c$ 列”）映射成内存偏移：

$$\text{index} = r \times \text{stride\_row} + c \times \text{stride\_col}$$

例如矩阵 $\begin{bmatrix} 0 & 1 & 2 & 3 \\ 4 & 5 & 6 & 7 \end{bmatrix}$，它的 `stride_row = 4`、`stride_col = 1`：每往下走一行，内存里前进 4 个位置；每往右走一列，前进 1 个位置。如果它是转置存储的，两者就对调。

## 完整的 matmul kernel

启动侧：`A` 是 $M \times K$，`B` 是 $K \times N$，输出 `C` 是 $M \times N$。块大小取 `BLOCK_M = 64`、`BLOCK_N = 64`、`BLOCK_K = 32`，网格是二维的 $\left(\left\lceil \frac{M}{64} \right\rceil, \left\lceil \frac{N}{64} \right\rceil\right)$：

```python
def triton_matmul_relu(a: torch.Tensor, b: torch.Tensor):
    assert a.is_cuda and b.is_cuda
    assert a.is_contiguous() and b.is_contiguous()
    assert a.shape[1] == b.shape[0]

    # A 是 M x K，B 是 K x N
    M, K = a.shape
    K, N = b.shape

    # 分配输出 tensor
    c = torch.empty((M, N), device=a.device)

    # 确定网格
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_relu_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )

    return c
```

kernel 本体（这里有一堆下标操作要细心跟踪，但算法骨架和 row sum 是相通的——只是从“沿一行扫 tile”变成了“同时沿 $A$ 的行 tile 与 $B$ 的列 tile 扫”）：

```python
@triton.jit
def matmul_relu_kernel(
    a_ptr, b_ptr, c_ptr,    # 计算 c = a @ b
    M, N, K,                # a 是 M x K，b 是 K x N，c 是 M x N
    stride_am, stride_ak,   # 如何在 a 中导航
    stride_bk, stride_bn,   # 如何在 b 中导航
    stride_cm, stride_cn,   # 如何在 c 中导航
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 我们负责的是第 (m, n) 个输出 tile
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 下标
    indices_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # a 的行号 [BLOCK_M]
    indices_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # b 的列号 [BLOCK_N]
    indices_k = tl.arange(0, BLOCK_K)                    # a 的行号 = b 的列号 [BLOCK_K]

    # a 与 b 的指针矩阵
    a_ptrs = a_ptr + indices_m[:, None] * stride_am + indices_k[None, :] * stride_ak  # [BLOCK_M, BLOCK_K]
    b_ptrs = b_ptr + indices_k[:, None] * stride_bk + indices_n[None, :] * stride_bn  # [BLOCK_K, BLOCK_N]

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # 沿 a 的行 tile、b 的列 tile 移动
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(indices_m[:, None] < M) & (indices_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(indices_k[:, None] + k < K) & (indices_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak  # 前进到 a 的下一个行 tile
        b_ptrs += BLOCK_K * stride_bk  # 前进到 b 的下一个列 tile

    # 施加激活函数（例如 ReLU）——这就是 kernel 融合
    acc = tl.maximum(acc, 0.0)

    # 写回输出 tile
    c_ptrs = c_ptr + indices_m[:, None] * stride_cm + indices_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(indices_m[:, None] < M) & (indices_n[None, :] < N))
```

逐段看：

- **定位输出 tile**：`pid_m = tl.program_id(0)`、`pid_n = tl.program_id(1)`——这个块负责 $C$ 矩阵中第 $(m, n)$ 个输出 tile；
- **计算下标**：`indices_m` 是 $A$ 的行号、`indices_n` 是 $B$ 的列号、`indices_k` 是“$A$ 的行号 = $B$ 的列号”这一共享维度（从 0 到 `BLOCK_K - 1`）；
- **构造指针矩阵**：`a_ptrs` 是 $A$ 里本块要用的所有地址（$[BLOCK\_M, BLOCK\_K]$），`b_ptrs` 同理（$[BLOCK\_K, BLOCK\_N]$）——用 `indices_m[:, None]` 与 `indices_k[None, :]` 做广播，生成“每个（行，列）位置对应一个内存地址”的指针矩阵；
- **初始化累加器**：`acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)`——这是 $[BLOCK\_M, BLOCK\_N]$ 的部分和矩阵，放在 shared memory 里；
- **沿 $k$ 循环**：`for k in range(0, K, BLOCK_K)`——和 row sum 的循环同一个骨架，只是现在同时沿两个方向扫。每轮用 mask 载入 $A$ 的小 tile 与 $B$ 的小 tile，然后 `acc += tl.dot(a, b)`：**一旦数据在 shared memory 里，代码就又“像 PyTorch”了**——`tl.dot` 就是“做矩阵乘”，Triton 会把它映射到合适的硬件单元；
- **指针前进**：`a_ptrs += BLOCK_K * stride_ak`、`b_ptrs += BLOCK_K * stride_bk`——跳到下一个 $k$ tile；
- **融合激活**：写回之前，`acc = tl.maximum(acc, 0.0)` 就是 ReLU——**不用再启动一个 kernel、不用把中间结果搬回 HBM**；
- **写回**：`tl.store(c_ptrs, acc, mask=...)` 把整个输出 tile 写回全局内存。

注意每一步都带 mask：`indices_m < M`、`indices_n < N`、`indices_k + k < K`，因为矩阵的维度不一定恰好是块大小的整数倍。

## 全课小结

把整讲收回来，这一讲其实讲的是**三个层面**：

1. **编程模型**：PyTorch、Triton，乃至 PTX——这是**程序员能控制**的部分。即使是 PTX，你也可以亲手去写、去特化到任意程度；
2. **硬件现实**：代码最终要跑在硬件上，而硬件只有有限的 SM、有限的 bank、有限的内存与寄存器容量。你带着巨大的矩阵和 transformer 进来，就必须让它**适配硬件的约束**；
3. **连接两者的桥梁**：**benchmark 与 profile**——它们让你理解硬件的“混乱”如何转化为真实的性能。

关于 **Triton**：它是一个很适合“以线程块为单位思考”的语言。现在你应该能体会到，**思考线程块比思考单个线程更容易**——因为你不用操心显式的线程同步、shared memory 的簿记。思维框架始终是：想清楚计算 → 拆解成线程块 → **从（shared）内存读入 → 做计算（顺带融合）→ 写回 HBM**。

这一讲看过的例子，难度层层递进：**逐元素**（GeLU）→ **行内归约**（softmax，行放得进块）→ **行内归约但放不进块**（row sum，“婴儿 tiling”）→ **矩阵乘法**（真正的 tiling）。到这一步，你已经拥有了写 **FlashAttention**（作业里会遇到）所需的全部积木。

下一讲，将从“一块 GPU”走向“更多 GPU”——**多 GPU 编程**。

### 课堂问答

**问答：除了 Triton，还有什么替代方案？**

（同学：如果能自己写 kernel，除了 Triton 还有哪些选择？它们能逼近最优性能到什么程度？）

**Percy**：每种语言都有它的**归纳偏置（inductive bias）**，让某些事更容易、某些事更难。Triton 是由训练 transformer 的人写的，所以**任何涉及 transformer 的东西在 Triton 里都相对容易**。极端情况下你总是可以退到 **PTX** 手写，但我不建议把它当第一步。还有不少其他语言/库：**ThunderKittens**、NVIDIA 的 **CUTE**，以及各种 **DSL**——它们未必能简单地分出高下，而是在技术栈的不同位置提供不同特性，各有取舍。

**问答：高维矩阵乘法该“一次读入”还是“逐元素处理”？**

（同学：做高维矩阵乘法时，是应该把整个张量一次性摊到所有线程块上、还是像本讲这样逐块读入处理、写回 HBM？）

**Percy**：这个问题的答案很难脱离具体计算抽象地回答——它取决于计算的性质。如果有机会，我们可以课下再细聊。

<!-- lecture-nav -->

**→ 下一讲**:[07 Parallelism](../07/)
