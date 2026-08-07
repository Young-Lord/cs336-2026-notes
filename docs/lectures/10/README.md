---
title: "10 Inference"
lecture: 10
---

# Lecture 10: Inference

**讲师**：Percy Liang · **主题**：推理（Inference）

## 本讲内容

上一讲（第九讲）Tatsu 讲了缩放定律，这一讲换 Percy 登场，从"怎么训练"转向"怎么使用"——推理（inference）。这是本课程中唯一专门讲推理的一讲，但正如 Percy 开场强调的：训练是一次性成本，而推理是每天都要承担的重复成本；在 agent 时代，token 生成量就代表计算量，推理的重要性还在上升。整讲围绕一个核心事实展开：**训练时可以沿序列并行、一次性看到所有 token，而自回归推理只能逐 token 生成——这导致推理（尤其 generation 阶段）是访存受限（memory-bound）的**，于是所有提速手段归根结底都是在"减少要搬动的字节数、尤其是 KV cache 的大小，同时尽量不伤精度"。

讲次分为四大部分。**第一部分**是理解推理工作负载：推理出现的所有场景（聊天、代码补全、agent、批量处理、评测、强化学习）、商业格局与开源工具（vLLM、SGLang、TensorRT-LLM、llama.cpp）、三种"快"的度量（TTFT、延迟、吞吐）、训练与推理的根本差异，以及 Transformer 的形状记法（$B/T/D/H$、收缩维/批处理维、$F=4D$、$D=NH$、$N=KG$）。**第二部分**是本讲的数学核心——算术强度：矩阵乘法的 FLOPs/访存记账与算术强度推导（强度约等于 batch 大小 $B$）、H100 的加速器强度 295 与 Roofline 判断、朴素推理的 $O(T^3)$ 与 KV cache、prefill/generation 两阶段下 MLP 强度 $B\cdot T$ 与注意力强度 $ST/(S+T)$ 的完整推导、为什么注意力不受益于 batching；然后用这套公式核算 Llama 2 13B 在 H100 上的吞吐与延迟（KV cache 公式、延迟线性于 $B$、吞吐渐近、batch 的延迟-吞吐权衡、TTFT 即 prefill 时间）。**第三部分**是各种提速技术：减小 KV cache 的架构手段（GQA、MLA、CLA、滑动窗口/局部注意力、DeepSeek v4 的 CSA/DSA/HCA、线性注意力与状态空间模型）、量化（机制、精度谱、QAT/PTQ、GPTQ、AWQ）、剪枝与蒸馏（NVIDIA 15B→8B）、两条配方（从零训练 vs 蒸馏），以及**无损**的投机解码（草稿模型 + 目标模型、接受概率 $\min(1,q/p)$、残差分布、精确采样的按例证明、Medusa/EAGLE）。**第四部分**是实际部署：实时流量下批处理的三大难点、Orca 的连续批处理（迭代级调度）、选择性批处理（注意力分开、MLP 拼接成 mega-sequence）、vLLM 的 PagedAttention（内部/外部碎片化、分块、系统提示共享、写时复制），最后是全讲总结。

| 页面 | 内容 |
|------|------|
| [10 · 推理工作负载概览：重要性、度量与 Transformer 记法](01-inference-overview.md) | 推理的应用场景、训练一次性成本 vs 推理重复成本、8.6 万亿 token/天与 agent 时代、商业格局与开源工具（vLLM/SGLang/TensorRT-LLM/llama.cpp）、**TTFT/延迟/吞吐三种度量**、训练 vs 推理的根本差异（沿序列并行 vs 自回归逐 token）、Transformer 形状记法（$B/T/D/H$、收缩维/批处理维、$F=4D$、$D=NH$、$N=KG$ 及 GQA 记法课堂问答） |
| [10 · 算术强度与推理核算：为什么推理是访存受限的](02-arithmetic-intensity.md) | **矩阵乘法 FLOPs/访存记账与算术强度推导（$\to B$）**、H100 加速器强度 295 与 Roofline 判断、$B=1$ 的极端情形、**朴素推理 $O(T^3)$ 与 KV cache**、prefill/generation 两阶段、**MLP 强度 $B\cdot T$ 与注意力强度 $ST/(S+T)$ 完整推导**（prefill $S/2$、generation $<1$）、为什么注意力不受益于 batching、**Llama 2 13B 的吞吐/延迟核算**（参数量、KV cache 公式、延迟线性于 $B$、吞吐渐近、batch 权衡、并行化与 TTFT） |
| [10 · 减小 KV 缓存：GQA、MLA、CLA 与局部注意力](03-reducing-kv-cache.md) | GQA（MHA/MQA/GQA、速度收益、$N/K$ 倍 KV cache 缩减、Llama 13B 核算、精度验证与保留态度）、**MLA（压缩潜向量 $c=W_ch$、DeepSeek V2 16384→512、RoPE 细节、精度对比）**、跨层注意力 CLA（Pareto 前沿）、滑动窗口注意力（KV cache 与序列长度无关、有效上下文随层数增长、混合层）、线性注意力/Mamba/DeltaNet 课堂问答、DeepSeek v4 的 CSA/DSA/HCA |
| [10 · 量化、剪枝与蒸馏：有损捷径](04-quantization-and-pruning.md) | **量化的 scale/zero point 机制（示例代码）**、精度谱（fp32/bf16/fp8/int8/int4）、量化感知训练 QAT vs 训练后量化 PTQ、GPTQ（Hessian 信息与误差补偿）、**AWQ（激活感知、保留 0.1%–1% 高精度权重、4 倍内存下降/3.2 倍加速）**、NVIDIA 剪枝+蒸馏（识别重要部件→移除→蒸馏修复，15B→8B）、重要层识别的课堂问答、**从零训练 vs 蒸馏两条配方** |
| [10 · 投机解码：无损的捷径](05-speculative-decoding.md) | "检查比生成快"的不对称性、**草稿模型 $P$ + 目标模型 $Q$**、接受概率 $\min(1, q/p)$ 与残差分布 $\max(q-p, 0)$、**精确采样的按例证明（两词表 $\{A,B\}$ 的完整推导）**、加速效果与草稿长度甜点区（3–4）、实践配比（70B/8B、8B/1B）、与前面技巧的结合、Medusa 与 EAGLE |
| [10 · 动态工作负载：连续批处理、分页注意力与总结](06-dynamic-workloads.md) | 实时流量下批处理的三大难点、**Orca 连续批处理（迭代级调度）**、选择性批处理（注意力分开、MLP 拼成 mega-sequence）、**PagedAttention（内部/外部碎片化、分块、系统提示共享、写时复制）**、vLLM 其他优化、全讲总结与新架构展望 |

## 本讲要点

- **训练与推理的本质差异**：训练（监督式）能看到所有 token、沿序列并行；推理因自回归只能逐 token 顺序生成，无法沿序列并行——这是推理难以达到高算术强度、难以用满计算的根源；
- **三种度量**：TTFT（用户等待首 token 的时间，交互应用）、延迟（单查询的秒/token，交互应用）、吞吐（多查询的 token/秒，批量处理）；**延迟与吞吐在 batch 维上存在张力**；
- **算术强度与 Roofline**：矩阵乘法 FLOPs 是三次方、访存是二次方，强度 $\approx B$；H100 的加速器强度约 295，计算受限当且仅当 $B > 295$；$B=1$（矩阵-向量积）的强度只有 1，这正是推理的情形；
- **朴素推理是 $O(T^3)$**：每次生成重算整个历史；因果 Transformer 保证前缀激活不变，于是用 **KV cache** 复用——对每个序列、token、层、KV 头存一个 $H$ 维向量；
- **推理的算术强度核算**：MLP 强度 $\to B\cdot T$（prefill 容易计算受限）；注意力强度 $= ST/(S+T)$，prefill 为 $S/2$、generation 为 $S/(S+1) < 1$——**generation 的注意力是根本性瓶颈**，因为它每个序列有自己的 KV cache、增大 $B$ 只是并行更多独立的小点积（蓝色批处理维）；
- **结论：prefill 计算受限、generation 访存受限**——人们说"推理是访存受限的"，你因此知道为什么；
- **Llama 2 13B 核算**：KV cache 每序列约 839 MB（$S \cdot KH \cdot L \cdot 2 \cdot 2$）、参数约 26 GB；延迟 $= \text{memory}/\text{bandwidth}$ 线性于 $B$，吞吐 $= B/\text{latency}$ 随 $B$ 增长但渐近且会撞上显存墙（$B=64$ 时 79.7 GB 已贴近 H100 的 80 GB）；**小 batch 换延迟、大 batch 换吞吐**；TTFT 本质上就是 prefill 时间；
- **减小 KV cache 是一切的统一主线**：GQA（$N/K$ 倍缩减，同时改善延迟与吞吐，给 batch 腾出余量）、MLA（压到 $C=512$ 维潜向量，还比 MHA 略好）、CLA（跨层共享）、滑动窗口/局部注意力（KV cache 与序列长度无关，但与全局注意力交错以避免伤精度）——**一切以不伤精度为前提**；
- **量化**：降低精度（bf16→fp8/int8/int4）减少内存；QAT 训练时模拟误差（贵）、PTQ 事后量化（便宜）；GPTQ 用 Hessian 信息补偿误差；**AWQ 只对少数大激活通道（0.1%–1%）保留高精度**，fp16→int3 得 4 倍内存下降、3.2 倍加速；
- **剪枝+蒸馏**：小校准集上按激活幅度识别重要部件（层/头/隐藏维度），移除后蒸馏修复——15B→8B 精度损失很小；两条配方：从零训练更快架构，或用原模型初始化（Frankenstein）再蒸馏修复；
- **投机解码（无损）**：草稿模型 $P$ 猜 $K$ 个 token，目标模型 $Q$ 并行审查，以 $\min(1, q/p)$ 接受、拒绝时从残差 $\max(q-p, 0)$ 采样——**数学保证得到目标模型的精确样本**（两词表证明：$P[A]=p(A)\frac{q(A)}{p(A)}+p(B)\cdot 0 = q(A)$、$P[B]=p(B)+p(A)-q(A)=q(B)$）；草稿长度甜点区约 3–4；
- **动态工作负载**：连续批处理（Orca 的迭代级调度：完成即移出、新请求随时加入）、选择性批处理（注意力各算各的、MLP 拼成 mega-sequence）；**PagedAttention 用操作系统的分页思想**解决 KV cache 碎片化（内部/外部碎片化），支持系统提示共享与"多响应共享前缀 + 块级写时复制"；
- **新架构的潜力**：KV cache 与注意力让 Transformer 天生"对推理不友好"；为推理而生的新架构（状态空间模型、线性注意力、扩散模型）可能解锁巨大收益。

## 课程导航

- [上一讲：09 Scaling Laws](../09/)
- [下一讲：11 Scaling Laws](../11/)
