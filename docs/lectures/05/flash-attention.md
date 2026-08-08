---
title: 05 · 胜利一圈:拆解 FlashAttention
lecture: 5
---

# 胜利一圈：拆解 FlashAttention

第三部分是某种意义上的**胜利一圈（victory lap）**。我们已经学到了关于 GPU 的一堆东西，现在把它们全部串起来，去理解一个在"注意力"领域重要的系统改进——**FlashAttention**。

FlashAttention（Dao 等人）是多年来注意力能力的一大进步，而且它**完全是系统层面的**：从 PyTorch 的朴素注意力实现，到一个精心融合的 kernel，推理延迟获得了戏剧性的改善。正如两讲之前展示过的，FlashAttention 是一种**内存高效**的计算方式，让你能处理更大更大的注意力。它的大部分收益，来自**减少 HBM（即全局内存）之间的数据搬运量**。

论文里到底做了什么？如果你去读原文，他们会说：**"我们只做了两件事——应用 tiling 与重计算，并用一种使内存访问次数呈次二次方（sub-quadratic）的方式来做。"** 现在我们已经理解这些组件了，看看实际发生了什么。

## 注意力的计算回顾

回顾注意力的计算：它由三个矩阵乘（与 $K$、$Q$、$V$ 相关的乘法）和中间的一个 softmax 组成：

$$\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

其中 $Q, K \in \mathbb{R}^{n \times d_k}$、$V \in \mathbb{R}^{n \times d_v}$。三个矩阵乘 + 一个 softmax,这就是我们需要处理的操作。矩阵乘我们刚学过怎么做——所以第一部分是平凡的。

## tiling 第一部分:KQV 矩阵乘的 tiling

论文的 **Figure 1 本质上就是一个 tiled 矩阵乘**,我们之前已经见过了:把输入 $K$、$Q$ 切成块,一块一块地乘;算完一块,把结果写出去。这里没有任何花哨的东西。

那 FlashAttention 的难点在哪里?(作业里也会遇到)难点是:**有一个作用在整个序列上的全局 softmax**。softmax 是全局操作,它把不同的 tile 全部"绑"在一起——你不能朴素地逐 tile 做完就了事,因为归一化需要看到所有分数。看起来 attention 不知道该怎么 tiling。

## tiling 第二部分:在线 softmax

关键技巧(一旦看到,其余就非常简单了)是**在线 softmax(online softmax)**,来自 Milakov 与 Gimelshein 2018 年的工作。

**标准 softmax** 是这样算的:先对所有元素做指数 $\exp(x_i)$,再**减去最大值**保证数值稳定,最后归一化:

$$\mathrm{softmax}(x)_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \qquad m = \max_j x_j$$

**在线版本**则边走边算归一化后的 softmax：每当你遇到一个比之前见过的**更大的数**，就把当前的最大值**换掉**，并对已累加的部分做**纠正（correction）**。除此之外，它就是"到目前为止所有元素的指数之和"的在线运行累加，最后再统一除以累加器。因为它是**在线**的、按块（block by block）推进的，所以可以**逐 tile 计算 softmax**——我不需要看到其余的 tile 才能算我这一块。我可以算完我这个 tile 的 softmax，保存部分结果（甚至可以写回全局内存），然后继续前进。

### 完整推导：逐 tile 的 softmax 更新

假设分数矩阵按 tile 分成若干块 $S^{(1)}, S^{(2)}, \dots$。我们维护两个量：

- **运行最大值** $m_t = \max_{j \le t} \max(S^{(j)})$（迄今为止见过的最大分数);
- **运行归一化常数** $l_t = \sum_{j \le t} \sum_{i \in S^{(j)}} e^{S^{(j)}_i - m_t}$（相对于当前 $m_t$ 的指数和）。

处理新的一块 $S^{(t+1)}$ 时，先算它的局部最大值与局部指数和：

$$m^{(t+1)} = \max(S^{(t+1)}), \qquad l^{(t+1)} = \sum_{i} e^{S^{(t+1)}_i - m^{(t+1)}}$$

然后做**望远镜求和**(telescoping sum)式的纠正更新:

$$m_{t+1} = \max(m_t,\ m^{(t+1)})$$

$$l_{t+1} = l_t \cdot e^{\,m_t - m_{t+1}} + l^{(t+1)} \cdot e^{\,m^{(t+1)} - m_{t+1}}$$

第一项 $l_t \cdot e^{\,m_t - m_{t+1}}$ 把之前按旧最大值累加的指数和"缩放"到新的最大值下(如果 $m_{t+1} = m_t$,缩放假为 1,什么都不变);第二项把新块的指数和也统一到新最大值下。整个过程只需要 $O(1)$ 的额外状态($m_t$ 与 $l_t$),与序列长度无关。处理完全部块后,任意位置 $i$ 的 softmax 权重就是:

$$P_i = \frac{e^{S_i - m_{\mathrm{final}}}}{l_{\mathrm{final}}}$$

![](/lectures/05/slide-53.png)

这就是在线 softmax：用 $O(1)$ 的在线状态，把"全局归一化"拆成可以逐 tile 推进的增量更新。下面用一个 PyTorch 片段把它落到实处（对向量版本的在线 softmax，块大小为 `block_size`）：

```python
import torch

def online_softmax(x, block_size=8):
    """逐块计算 softmax,只维护 running max 与 running sum(O(1) 状态)。"""
    running_max = float("-inf")
    running_sum = 0.0
    for start in range(0, x.numel(), block_size):
        block = x[start:start + block_size]
        block_max = float(block.max())
        new_max = max(running_max, block_max)
        # 望远镜纠正:把旧累加与新块统一到 new_max 下
        running_sum = running_sum * torch.exp(running_max - new_max) \
                    + torch.exp(block - new_max).sum()
        running_max = new_max
    return torch.exp(x - running_max) / running_sum

x = torch.tensor([3.0, 1.0, 2.0, -1.0])
print(online_softmax(x))
print(torch.softmax(x, dim=0))   # 应与标准 softmax 一致
```

## 把一切拼起来：FlashAttention 前向

把上面的零件组合起来，FlashAttention 的前向就清楚了（Dao 2023）。这是一张来自 FlashAttention 2 的示意图，虚线块表示在 SRAM 中按 tile 计算的部分，蓝色块表示在 HBM（全局内存）中的部分：

![](/lectures/05/slide-54.png)

1. **逐 tile 计算内积 $S = Q K^\top$**(tiled 矩阵乘)，结果**留在 SRAM**;
2. **融合指数算子**：紧接着算 $P = \exp(S - m)$，不必物化完整的 $S$;
3. **逐 tile 计算 softmax**：用上面的在线望远镜求和技巧，维护 running max 与 running sum，跨 tile 传递部分和；
4. 同时以 **tiled 形式**把 $P$ 乘上 $V$，累积输出 $O$（未归一化的加权和)；最后在结尾统一除以归一化常数 $l_{\mathrm{final}}$。

整个过程，那张 $n \times n$ 的注意力矩阵**从不落盘到 HBM**——中间量要么在 SRAM，要么是 $O(1)$ 的在线状态，要么是最终除以归一化常数的那个小动作。注意力被切成小块，一块块推进，而不需要把巨大的中间结果来回搬运。

反向传播也一样优雅：Tatsu 没有展开讲，但指出 FlashAttention 的最后一个零件正是**重计算（recomputation）**——如果你保存了激活，你就仍要保存一个 $n^2$ 大小的注意力矩阵；与其保存它，**不如把它全部扔掉，在反向时逐 tile 重新计算**。于是我们学过的六个技巧，在 FlashAttention 里同时用上了：tiling（前后向）、算子融合（把指数融合进 tiled matmul）、在线 softmax（望远镜求和）、重计算（反向）、以及处处围绕"减少 HBM 搬运"的内存意识。

## 全课小结

把整讲收回来，Tatsu 的结语有三点：

1. **硬件驱动规模，底层细节决定什么能扩展（或不扩展）**。从 GPU 到 TPU，硬件是我们做的一切决策的基础。不要做"cargo cult"式的抄作业——比如盲目地让矩阵维度都是 32 的倍数——而要真正理解**为什么**要这样做，这需要一直追到硬件层面；
2. **当前基于 GPU 的计算，强烈鼓励我们把注意力放在 matmul + 数据搬运上**。矩阵乘是算术上最密集的核心操作；而由于算力与内存之间的差距，我们必须非常小心地对待**数据搬运（data movement）**;
3. **认真思考 GPU（合并访存、tiling、融合）会带来好的性能**。像 FlashAttention 这样的成果很酷，也有大量"硬件感知"的架构工作在推进；在设计未来系统时，**贴着硬件思考**是获得好性能的关键。

谢谢大家——下一讲开始进入 kernel 与 Triton，把这些思想真正写进代码。

<!-- lecture-nav -->

**→ 下一讲**:[06 Kernels, Triton](../06/)
