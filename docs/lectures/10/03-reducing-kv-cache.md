---
title: "10 · 减小 KV 缓存：GQA、MLA、CLA 与局部注意力"
lecture: 10
---

# 减小 KV 缓存：GQA、MLA、CLA 与局部注意力

现在我们已经建立了分析框架，可以开始让推理变快了。让推理更快的方法五花八门——从改模型架构到做系统优化，介于两者之间的还有很多。推理在某种意义上是一个相当"横切"（cross-cutting）的话题。

首先要做的、也是最显然的事（讲者说"希望已经刻进你们脑子里了"）：**内存是推理的瓶颈，而 KV cache 占了大量内存**——当 batch 足够大时，KV cache 甚至可以比参数量还大。那就试着**减小 KV cache 的大小**。当然要小心：不要因此损失太多精度。

## 分组查询注意力（GQA）

第一种做法我们其实已经讲过——**分组查询注意力（grouped-query attention，GQA）**。回顾一下：标准多头注意力（multi-headed attention，MHA）里，每个 token 有 query、key、value 各一份。GQA 则保持 query 头数不变，但只保留**更少的 key-value 头**：$N$ 个 query 头、$K$ 个 KV 头，每个 KV 头与 $N/K$ 个 query 头交互。三种极端情况：

- **MHA**：$K = N$，没有减少；
- **多查询注意力（multi-query attention，MQA）**：$K = 1$——几乎没人用，因为它真的效果很差；
- **GQA**：$K$ 取中间某个值，希望找到精度与速度之间的平衡。

GQA 的论文（2023）展示了**每样本时间（time per sample）**——它与延迟、吞吐都相关。MHA（完整注意力）的时间最高；把 $K$ 降到 1 会快很多；而在 $K$ 附近比如 $K = 8$，依然保持很快的速度，同时精度大幅回升。

![GQA 的速度收益：减少 KV 头数显著降低每样本时间](/lectures/10/gqa-speed.png)

### 为什么 GQA 能提升延迟与吞吐

原因很直接：**GQA 把 KV cache 缩小了 $N/K$ 倍**。而我们已经反复强调——推理是访存受限的，**减少内存占用就直接带来加速**。

回到我们熟悉的 Llama 2 13B。原始配置是 MHA（$K = N = 40$）。在 batch 为 64 时，我们得到：显存 79.7 GB、延迟 23.8 ms/token、吞吐 2,689 token/s（几乎占满 H100 的 80 GB）。现在做 GQA，采用 1∶5 的稀疏比（$K = 8$，即 $N/K = 5$）。KV cache 从每序列 839 MB 降到约 168 MB：

| 配置 | 显存 | 延迟 | 吞吐 |
|------|------|------|------|
| MHA，$K=40$，$B=64$ | 79.7 GB（逼近 H100 上限） | 23.8 ms/token | 2,689 token/s |
| GQA，$K=8$，$B=64$ | 36.8 GB | 11.0 ms/token | 5,831 token/s |
| GQA，$K=8$，$B=256$ | 69.0 GB | 20.6 ms/token | 12,434 token/s |

注意第一行到第二行：**延迟和吞吐同时变好了**——减少内存时，两者不冲突，都会改善；主要是 **batch 维**才是延迟与吞吐的张力的来源。而且 $B=64$ 时 MHA 已经贴着 H100 的 80 GB 上限，几乎没有余量。现在用 GQA 后显存空出来了，可以继续加大 batch：$B=256$ 时仍然放得进 80 GB，延迟略有上升（batch 变大），但吞吐按比例大幅上涨。这告诉我们：有时要**联合调节参数**——减小 KV cache 换来 batch 余量，再去做别的权衡。

用讲义里的符号系统写出来就是：

```python
# 用讲义中的性能统计函数做 GQA 核算
config_mha  = llama2_13b_config({K: 40, B: 64})   # 原始 MHA
config_gqa  = llama2_13b_config({K: 8,  B: 64})   # GQA, 1:5 比例
config_gqa_big = llama2_13b_config({K: 8, B: 256})  # 加大 batch

stats_mha     = compute_transformer_performance_stats(config_mha)
stats_gqa     = compute_transformer_performance_stats(config_gqa)
stats_gqa_big = compute_transformer_performance_stats(config_gqa_big)
```

### 精度检查

任何有损（lossy）改动最后都要检查一件事：**精度有没有掉**。GQA 论文在一堆评测上做了验证，效果基本都不错。

![GQA 的精度验证：各项评测基本不掉点](/lectures/10/gqa-accuracy.png)

不过讲者提醒：这类评测结果**总要带着怀疑去看**（"take it with a grain of salt"）。后面 DeepSeek 的论文就会展示，GQA 实际上确实是有损的——所以除了纯数学推导之外，论文里的经验结论都要谨慎对待。

## 多头潜变量注意力（MLA）

说到 DeepSeek，这里介绍另一个减小 KV cache 的想法。思路主线依然是：减小 KV cache → 提升延迟与吞吐。

回顾 MHA 与 GQA：MHA 对每个 token 有相同数量的 query、key、value；GQA 减少的是 key 与 value 的**数量**（头数）。**多头潜变量注意力（multi-head latent attention，MLA）**则完全保持 key、value 的数量不变（每个 token 各一份），转而把 KV cache 做**参数化压缩**。

正常情况下怎么算 key 和 value？把激活 $h$ 分别乘以一个矩阵：

$$K = W_K h, \qquad V = W_V h$$

它们的维度一般是 $N \cdot H$，等于模型维度，相当大。MLA 的做法是：先把激活**投影到一个压缩维度 $C$** 的潜向量

$$c = W_c h$$

再从压缩向量投影出 key 和 value：

$$K = W_K c, \qquad V = W_V c$$

这样 KV cache 里只需要存 $C$ 维的 $c$（小得多），真正需要 key、value 时再现场"物化"（materialize）出来。DeepSeek V2 把 KV 维度从 $N \cdot H = 16384$ 压到了 $C = 512$——非常激进的压缩。

![MLA 的示意图：把 KV cache 压缩为低维潜向量](/lectures/10/mla-schema.png)

有一个细节（wrinkle）：**MLA 与 RoPE 不兼容**——RoPE 直接作用在 key 上。所以 DeepSeek 额外加了几十维（64 维）专门处理 RoPE 的位置信息，实际存储是 $512 + 64 = 576$ 维。但无论如何，压缩幅度依然很大。

延迟与吞吐的收益不需要新论证：KV cache 变小，访存受限的推理自然变快——几乎是线性缩放，直到某个极限。

### 精度的两个对比

看精度。这里有两张表：

1. **MHA 优于 GQA**（虽然更贵）——所以 DeepSeek 的结论与 GQA 论文相抵触：GQA 并没有那么好；
2. **MLA 甚至比 MHA 略好**，而且便宜得多。

![MLA 与 GQA 的精度对比（一）：MHA 优于 GQA](/lectures/10/mla-accuracy.png)

![MLA 与 GQA 的精度对比（二）：MLA 优于 MHA 且更便宜](/lectures/10/mla-accuracy2.png)

> **课堂问答：MLA 与减小模型维度相比如何？**
>
> （同学：MLA 跟直接减小模型维度比怎么样？）
>
> **Percy**：好问题，但这些消融里没有展示。我的猜测是：盲目减小模型维度会让一切变差，因为你是不加区分地砍掉所有东西。这类工作的窍门在于**找到模型里可以安全压缩的地方**——而这事先并不一定能预知，只能靠大量实验试出来。

## 跨层注意力（CLA）

再介绍一个减小 KV cache 的思路：**跨层注意力（cross-layer attention，CLA）**。常规做法是每一层都维护自己的 key、value。CLA 说：不必这样——只在**一部分层**计算 key、value，其他层直接复用相邻层的 KV cache。

![CLA 示意图：跨层共享 KV](/lectures/10/cla-diagram.png)

这和 GQA 如出一辙：GQA 是**跨头**共享 KV，CLA 是**跨层**共享 KV。论文实验表明，这样做能改善**精度-开销的 Pareto 前沿**：给定同样的 KV cache 预算，CLA 能拿到更高的精度；或者说，给定同样的精度，CLA 的 KV cache（从而延迟、吞吐）更好。

![CLA 的结果：改善精度与 KV cache 大小的 Pareto 前沿](/lectures/10/cla-results.png)

## 局部（滑动窗口）注意力

接着是快速浏览一组减小 KV cache 的技巧。**局部注意力（local attention）/ 滑动窗口注意力（sliding window attention）**是很老、也很自然的想法：完整注意力的矩阵是 $S^2$ 大小；生成新 token 时，其实**只看最近的 $K$ 个 token** 就够了——对每个要生成的 token，只依赖它前面 $K$ 个 token。

![滑动窗口注意力：每个 token 只看最近的几个 token](/lectures/10/longformer-attention.png)

好处很大：

- **KV cache 与序列长度无关**！它只取决于 $B \times$（每序列固定的窗口大小 $K \times$ 头数等），对长上下文尤其友好；
- 而且由于有多层堆叠，信息的**有效上下文长度**可以比单层窗口更大——信息在层与层之间向下游传播，有效感受野随层数线性增长。

还可以做更花哨的变体：不是稠密地选择每一层，而是隔几层取一个；或者**全局（global）+ 局部（local）**混合——对一组固定的 token 点做全局注意力，再加上局部滑动窗口。

但问题是：**它仍然伤精度**，降低了表达力（expressivity）——没有免费的午餐，或者说这顿午餐很贵。人们的解决方案是：把局部注意力与全局注意力**交错（interleave）**——这些混合（hybrid）模型让一部分层用完整注意力、另一部分层用局部注意力，从而适度减小 KV cache，同时尽量平衡精度。

> **课堂问答：线性注意力变体与滑动窗口的取舍**
>
> （同学：线性注意力变体跟滑动窗口之间的权衡是什么？哪个更好？）
>
> **Percy**：我本来不打算细讲线性注意力，但可以快速说一下。有一类方法不存 KV cache，而是对全部历史计算某种**压缩表示**。最朴素的线性注意力就是把所有 KV 值累加成一个向量——这当然和序列长度无关了。更聪明的做法如 GateNet、DeltaNet、Mamba，允许"压缩但不遗忘"。它们也被用作滑动窗口注意力的替代品，结果不错。你也可以把完整注意力、滑动窗口注意力、线性注意力**组合**起来用，因为它们捕捉不同方面：如果你关心局部的、高分辨率的信息，滑动窗口更好；如果只是想要对过去的一个宽泛摘要，线性注意力可能更好。
>
> （同学：那对于长上下文，线性注意力是不是更好的设置？）
>
> **Percy**：没有免费午餐。比如一个很长的上下文、要解决"大海捞针"（needle in a haystack）问题——如果你必须把整个历史压缩进很小的表示，你就是在丢失信息，很可能捞不回那根针。
>
> （同学：我的直觉是，大家似乎在走向分层结构，但总是需要一些更长距离的注意力。我有点想搞清楚滑动窗口和一个 Mamba/DeltaNet 层的权衡到底是什么。用 DeltaNet 层是不是总是优于滑动窗口？）
>
> **Percy**：我大概会说，Mamba 和 DeltaNet 比滑动窗口注意力**更强**。你可以把 Mamba 想成至少能表达滑动窗口注意力的某些方面——做递归时它本来就能看最近的状态。所以线性注意力及其扩展**有更多的发挥空间**；一旦你用了滑动窗口注意力，也就到头了。

## DeepSeek v4 的注意力家族

顺带快速highlight一下：DeepSeek 持续在注意力机制上做创新。之前他们提出了压缩 KV 的 MLA；现在又有了一系列更激进的压缩方案。DeepSeek v4 支持 **100 万（1M）token 的上下文**，用到了几个互相配合的机制：

![DeepSeek v4 的注意力机制：CSA、DSA 与 HCA](/lectures/10/deepseek-v4-attention.png)

- **压缩稀疏注意力（Compressed Sparse Attention，CSA）**：把每 $m$ 个 token 压缩成 1 个 token；
- **DeepSeek 稀疏注意力（DeepSeek Sparse Attention，DSA）**：从压缩后的 token 里**选出 top-$k$** 个保留。选法很有意思：先用一组更轻量的 query 和 key 做一次小注意力，得到"索引分数（index scores）"，从而判断哪些 token 值得保留——一个"闪电般快"的方式来决定要保留哪些 token；
- **重度压缩注意力（Heavily Compressed Attention，HCA）**：在 DSA 基础上进一步压缩。

这些机制的总体目标都一致：在不大幅伤精度的前提下，把 KV cache 和注意力成本压下来。

## 本节小结

回顾一下这节的目标：**推理是访存受限的，所以减小 KV cache 就直接转化为吞吐与延迟的收益**，关键在于不伤精度。手段包括：

- **降低 KV cache 的维度**：跨头（GQA）、跨层（CLA）、压缩（MLA）；
- **局部注意力**：截断 KV cache（只在部分层用，与全局注意力交错）；
- 其他想法：**线性注意力 / 状态空间模型（Mamba 2、GatedDeltaNet 等）**、以及**扩散模型**——一种非自回归的生成方式，可能快得多。

别忘了，这里还有一个更激进的系统层面的做法：**量化**。我们下一节讲。

<!-- lecture-nav -->

**← 上一节**：[算术强度与推理核算：为什么推理是访存受限的](02-arithmetic-intensity.md)　**→ 下一节**：[量化、剪枝与蒸馏：有损捷径](04-quantization-and-pruning.md)
