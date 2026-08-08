---
title: "06 · 归约与“婴儿 tiling”：softmax 与 row sum"
lecture: 6
---

# 归约与“婴儿 tiling”：softmax 与 row sum

到目前为止看到的都是**逐元素**（elementwise）运算。现在转向**聚合多个值**的运算——这类运算需要跨值通信，也是从“每个元素一个线程”过渡到“每个线程块负责一段数据”的关键一步。

## Softmax 回顾

回忆 softmax 的作用：在注意力里、在生成概率输出时，你都需要它。对矩阵的**每一行**做指数化（exponentiate）并归一化（normalize）：

$$\mathrm{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

例如：

$$[0\ \ 0\ \ 0] \Rightarrow \left[\frac{1}{3}\ \ \frac{1}{3}\ \ \frac{1}{3}\right], \qquad [1\ \ 1\ \ -\infty] \Rightarrow \left[\frac{1}{2}\ \ \frac{1}{2}\ \ 0\right]$$

## 朴素实现：数一数读写次数

先看朴素实现（这个实现应该大家都熟，作业一里也有），并**记录每一步的读写**：

```python
def naive_softmax(x: torch.Tensor):
    # M 行、N 列
    M, N = x.shape

    # 每行取最大值（MN 次读，M 次写）——为了数值稳定性
    x_max = x.max(dim=1)[0]

    # 减去最大值（MN + M 次读，MN 次写）
    x = x - x_max[:, None]

    # 指数化（MN 次读，MN 次写）
    numerator = torch.exp(x)

    # 求归一化常数（MN 次读，M 次写）
    denominator = numerator.sum(dim=1)

    # 归一化（MN 次读，MN 次写）
    y = numerator / denominator[:, None]

    return y
```

这是普通的 PyTorch，**每一步都是一个独立的 kernel**：除非调用 `torch.compile`，这些运算各读各写、数据在 HBM 与 SM 之间反复往返。逐行累加读写次数：

| 步骤 | 读 | 写 |
|------|------|------|
| `x.max(dim=1)` | $MN$ | $M$ |
| `x - x_max` | $MN + M$ | $MN$ |
| `torch.exp(x)` | $MN$ | $MN$ |
| `numerator.sum(dim=1)` | $MN$ | $M$ |
| `numerator / denominator` | $MN$ | $MN$ |
| **合计** | $\mathbf{5MN + M}$ | $\mathbf{3MN + 2M}$ |

而**原则上**只需要读一次 $MN$、写一次 $MN$——把整个 softmax 融合成一个 kernel，就能省下约 **4 倍的读写**。这正是写 Triton kernel 的动机。

## Triton softmax：每行一个线程块

softmax 不是逐元素操作，但它几乎是**逐行**（row-wise）的：每一行的归一化只涉及该行自己，**行与行之间不交互**。所以很自然地把**每一行交给一个线程块**。各行之间不需要 shared memory 通信（块之间本来也没有 shared memory），每个块在自己的行内做归约即可。下图示意了这个思路——输入矩阵的每一行都由一个线程块独立处理：

![](/lectures/06/triton-softmax.png)

先看启动侧：

```python
def triton_softmax(x: torch.Tensor):
    # 分配输出 tensor
    y = torch.empty_like(x)

    # 确定网格
    M, N = x.shape                          # 行数 × 列数
    block_size = triton.next_power_of_2(N)  # 每块容纳所有列
    num_blocks = M                          # 每行一个块

    # 启动 kernel
    triton_softmax_kernel[(M,)](
        x_ptr=x, y_ptr=y,
        x_row_stride=x.stride(0), y_row_stride=y.stride(0),
        num_cols=N, BLOCK_SIZE=block_size
    )

    return y
```

`block_size` 取 `N` 的下一个 2 的幂（图个“吉利”，也满足 Triton 对块大小的要求）；网格有 $M$ 个块，正好每行一个。`stride`（步长）告诉 kernel“往下走一行要跳多远”。

kernel 的核心计算部分——**一旦做完脚手架，它看起来几乎和朴素版本一模一样**：

```python
@triton.jit
def triton_softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, num_cols, BLOCK_SIZE: tl.constexpr):
    assert num_cols <= BLOCK_SIZE

    # 每一行独立处理
    row_idx = tl.program_id(0)                    # 我是哪一行？
    col_offsets = tl.arange(0, BLOCK_SIZE)        # 从 0 到 BLOCK_SIZE-1 的列

    # 从全局内存读入
    x_start_ptr = x_ptr + row_idx * x_row_stride  # 本行起点
    x_ptrs = x_start_ptr + col_offsets
    x_row = tl.load(x_ptrs, mask=col_offsets < num_cols, other=float("-inf"))

    # 计算
    x_row = x_row - tl.max(x_row, axis=0)         # 减去行内最大值（数值稳定）
    numerator = tl.exp(x_row)
    denominator = tl.sum(numerator, axis=0)       # 归一化常数
    y_row = numerator / denominator

    # 写回全局内存
    y_start_ptr = y_ptr + row_idx * y_row_stride
    y_ptrs = y_start_ptr + col_offsets
    tl.store(y_ptrs, y_row, mask=col_offsets < num_cols)
```

逐段看：

- **醒来定位行**：`row_idx = tl.program_id(0)` 决定处理第几行；`col_offsets = tl.arange(0, BLOCK_SIZE)` 给出本行的所有列；
- **读入整行**：`x_start_ptr = x_ptr + row_idx * x_row_stride` 是这一行在内存里的起点；`x_ptrs` 是该行所有元素的地址；`tl.load(..., mask=col_offsets < num_cols, other=float("-inf"))` 读入整行——如果 `N` 不是 2 的幂，多出来的那几列会被 mask 掉，填上 $-\infty$：指数化后 $e^{-\infty} = 0$，对 softmax 来说等价于“没有这个元素”；
- **计算**：`x_row - tl.max(x_row, axis=0)` 先减去行内最大值（数值稳定性），然后 `tl.exp`、`tl.sum`、相除——与朴素版本的计算完全一致；
- **写回**：`tl.store(y_ptrs, y_row, mask=...)` 把结果按行写回全局内存，同样用 mask 防止写到边界之外。

所以这一版本的要点是：**行能放进一个块时，Triton 代码几乎就是 PyTorch 代码**——`tl.max`、`tl.exp`、`tl.sum` 这些块级归约都由 Triton 在块内处理。

### 课堂问答

**问答：列数比块大小还多怎么办？**

（同学：如果列数、行数都比块大小大怎么办？）

**Percy**：这正是接下来要讲的问题——一般情况下行会远大于块大小，我们马上处理它。

**问答：想做“按列”的 softmax 怎么办？**

（同学：如果我想按列做 softmax 呢？）

**Percy**：那也应该可以——因为我们这里追踪的是**指针**，指针可以是任意位置。只需要调整这里的 stride：把“列偏移”改成乘以行 stride（也就是沿列方向取元素），就能按列访问。改一处 stride 即可。

## Row sum：当行放不进块时

接下来热身到“婴儿版 tiling”。假设**一行放不进一个块**——比如一行有 **4096 列**，而块大小只有 **1024**。这时必须做点什么。

**策略**如下：

1. 把一行**切成若干 tile**（上例中就是 4 个）；
2. 每个线程**迭代遍历这些 tile**，把读到的元素**累加**进自己的累加器；
3. 最后做**归约（reduction）**——把所有线程累加器的结果加起来（用 shared memory 或 warp shuffle）。

为了便于思考，从 softmax 换成一个更简单的例子：**row sum（行求和）**——把矩阵每一行的元素加起来。朴素的 `x.sum(dim=1)` 没什么可讲的：

```python
def builtin_row_sum(x: torch.Tensor):
    return x.sum(dim=1)
```

概念上，每个块仍然负责一行（这一点没变）。假设这一行有 12 个元素、块大小（tile 大小）为 4：tile 0 是列 0–3、tile 1 是列 4–7、tile 2 是列 8–11。四个线程 $T_1, T_2, T_3, T_4$ 各自维护一个累加器，依次处理每个 tile：

$$T_1:\ 1 + 5 + 9 = 15, \qquad T_2:\ 2 + 6 + 10 = 18, \qquad T_3:\ 3 + 7 + 11 = 21, \qquad T_4:\ 4 + 8 + 12 = 24$$

每个线程先处理第一个 tile（把 1、2、3、4 放进各自的累加器），再处理第二个 tile（各加 5、6、7、8），再处理第三个 tile（各加 9、10、11、12）。最后把四个累加器**加起来**得到 $15 + 18 + 21 + 24 = 78$，就是这一行的和。下图是这个过程的概念示意——一行被切成多个 tile，四个线程各自跨 tile 累加：

![](/lectures/06/triton-row-sum.png)

对应的 Triton kernel 如下：

```python
def triton_row_sum(x: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty(M, device=x.device, dtype=x.dtype)
    row_sum_kernel[(M,)](x, y, N, BLOCK_SIZE=BLOCK_SIZE)
    return y


@triton.jit
def row_sum_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)  # 处理哪一行？

    # 每个线程一个累加器
    # 一行：T1 T2 T3 T4 | T1 T2 T3 T4 | T1 T2 T3 T4（N = 12，BLOCK_SIZE = 4）
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # 遍历所有 tile
    for start in range(0, N, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0)
        acc += x

    # 最终归约：从 BLOCK_SIZE 个累加值（所有线程）归约成标量
    result = tl.sum(acc, axis=0)

    tl.store(out_ptr + row, result)
```

逐段看：

- **醒来定位行**：`row = tl.program_id(0)`，这一行有 $N$ 个元素，而块大小（tile 大小）是 `BLOCK_SIZE`——**注意块大小是线程数，我们处理的数据比块大，只能循环**；
- **初始化累加器**：`acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)`——每个线程一个累加器；
- **循环遍历 tile**：`for start in range(0, N, BLOCK_SIZE)`，`start` 依次取 $0$、`BLOCK_SIZE`、$2 \times$ `BLOCK_SIZE`……每轮用 `cols = start + tl.arange(0, BLOCK_SIZE)` 得到当前 tile 的列号，`mask` 防止越界，`tl.load` 读入 tile，然后 `acc += x` **累进累加器**；
- **最终归约**：循环结束后，每个线程手里有一个（覆盖若干列的）部分和向量；`result = tl.sum(acc, axis=0)` 把整个向量归约成标量；
- **写回**：`tl.store(out_ptr + row, result)` 把这一行的和写出去。

这比 GeLU 复杂的地方在于**块内部多了一个 for 循环**——这是“数据放不进一个块”时的必然产物。

### 块（block）与 tile 的区别

值得专门强调一个容易混淆的点：**tile 不是块**。回忆 GeLU 里我们也把一行切成了若干块——但那里的每一片都是一个**独立的线程块**，由各自独立的块并行处理。而 row sum 里，**这些 tile 属于同一个块**：同一个块要依次处理完一行里的所有 tile（通过循环）。所以这里的 tile 只是**同一块内的数据分片**。

正是从这里开始，代码**不再像 PyTorch 了**——因为你没法把全部数据“一口气”优雅地处理完，不是所有数据都能放进 shared memory。

### 课堂问答

**问答：能控制累加器放在哪里吗？**

（同学：能不能控制累加器驻留在寄存器还是 shared memory？）

**Percy**：在 Triton 程序里你不能显式指定——这由 **Triton 编译器**决定。一般来说，块大小足够大时，累加器就得放 shared memory；小的话放寄存器。

（同学：如果块大小很大，累加器就放在 shared memory 里？）

**Percy**：对，块足够大时累加器会落到 shared memory 里。

有了逐元素（GeLU）、行内归约（softmax）、“婴儿 tiling”（row sum）这三个例子，就只剩最后一块拼图了——**真正的 tiling：矩阵乘法**。

<!-- lecture-nav -->

**→ 下一节**：[矩阵乘法：tiling 与 kernel 融合](matmul-tiling.md)
