---
title: "08 Parallelism"
lecture: 8
---

# Lecture 8: Parallelism

**讲师**：Tatsu Hashimoto · **主题**：并行（Parallelism）

## 本讲内容

上一讲（第七讲）Percy 讲了并行的**底层机制**——collectives、硬件互联、torch.distributed，以及数据并行/张量并行/流水线并行的最初版本。这一讲由 Tatsu Hashimoto 继续并行的主题，但换了一个视角：不再是"实现一个并行训练"，而是**深入系统细节**——网络拓扑、all-reduce 的各种算法、ZeRO/FSDP 的显存与通信核算、Megatron 风格的各种并行组合，以及 TPU mesh 与 GPU fat-tree 的硬件差异。如果说第七讲教你"怎么把并行跑起来"，这一讲教你"在大规模下到底该选哪种并行、为什么"。

讲次分为三大部分。**第一部分**是 LLM 网络基础：多 GPU 的两个动机（算力与显存）、节点内/节点间通信、collective 复习（all-reduce = reduce-scatter + all-gather 这个对 ZeRO 至关重要的等价）、TPU 的 toroidal mesh 与 GPU 的 fat-tree 两种网络哲学、TPU 8i/8t 与 Virgo 网络的新趋势、华为昇腾 910 的"用功耗换连接性"。**第二部分**是标准并行化原语：朴素数据并行与显存记账（5 份拷贝、16 字节/参数）、ZeRO 三个 stage 的显存与通信核算（stage 1 为何"免费"、stage 2 的增量反向、stage 3 即 FSDP 的两次 all-gather + 一次 reduce-scatter 与通信重叠）、数据并行的两个天花板（critical batch size 与激活显存）；然后是模型并行——流水线并行（bubble 公式、micro-batch、DeepSeek 调度、zero-bubble）与张量并行（矩阵分解、f/g 对偶、列切/行切、通信量对比 $8bsh$ vs $bsh$）；再是激活显存（Korthikanti 公式）、序列并行、专家并行（EP 与 DP/TP 的组合约束、Megatron 的解耦、DeepSeek DPP 与 Nvidia Hybrid EP）、上下文并行/ring attention，最后是一张完整的并行策略对比总表。**第三部分**是把所有策略组合起来：算力/通信建模与利用率图、3D/4D 并行的简单处方（先 TP/EP 吃快互联，再 PP 或 FSDP 跨机，最后 DP 铺满）、Megatron 的实践指南、Narayanan 2021 的缩放规律，以及 Olmo、DeepSeek、Yi、Llama 3 405B、Gemma 2、Mixtral 8x22B、Nemotron、Qwen 3 等真实模型的并行配置与总表。

| 页面 | 内容 |
|------|------|
| [08 · 动机与网络：从单卡到数据中心](01-motivation-and-networking.md) | 两个瓶颈（算力/显存）、节点内 vs 节点间通信、collective 复习与 all-reduce = reduce-scatter + all-gather 等价、TPU toroidal mesh vs GPU fat-tree 两种网络哲学、TPU 8i/8t 与 Virgo 的趋势、华为昇腾 910 的功耗换连接性、Part 1 小结（计算的新单位是数据中心） |
| [08 · 数据并行与 ZeRO：把显存摊到所有 GPU](02-data-parallelism-and-zero.md) | 朴素数据并行与 SGD、显存记账（5 份拷贝 16 字节）、ZeRO 核心思想与显存构成图、**stage 1/2/3 的完整流程与通信/显存推导**（stage 1 免费、FSDP 两次 all-gather + 一次 reduce-scatter、通信重叠）、8×A100 实践核算（12/5/3.25/1.5 字节每参数）、critical batch size 与数据并行的天花板、模型并行"传激活而非传参数"的概念转变 |
| [08 · 模型并行：流水线与张量并行](03-pipeline-and-tensor-parallelism.md) | 层切的低利用率、**pipeline bubble 公式 $\frac{n_{\text{stages}}-1}{n_{\text{micro}}}$ 完整推导**、为什么流水线放最慢链路、Megatron 参数扫描、交错调度与 **zero-bubble（反向拆成传播激活梯度 + 计算权重梯度）**、张量并行的矩阵分解与 f/g 对偶、列切/行切、**TP 与 PP 通信量对比（$8bsh$ all-reduce vs $bsh$ 点对点）** |
| [08 · 激活显存、序列并行与专家并行](04-activation-sequence-expert.md) | 激活显存是动态的、Korthikanti 每层激活公式 $34bsh + 5as/h$、张量并行下剩余的 $10sbh$ 项、**序列并行（沿序列轴切逐点操作）**、专家并行（切专家而非矩阵）、EP 与 DP/TP 的组合约束、**注意力与 MLP 的解耦并行**（Megatron 的 ETP/EP/EDP）、DeepSeek DPP 与 Nvidia Hybrid EP、上下文并行/ring attention、**并行策略对比总表** |
| [08 · 组合策略与大规模训练案例](05-scale-and-case-studies.md) | 算力/通信建模与利用率图、**3D/4D 并行的简单处方**（TP/EP 吃快互联 → PP/FSDP 跨机 → DP 铺满）、Megatron 实践指南、Narayanan 2021 缩放规律、TP=8 最优与激活重计算、Olmo/DeepSeek/Yi/Llama 3 405B/Gemma 2/Mixtral/Nemotron/Qwen 3 的并行配置与总表、全讲收尾 |

## 本讲要点

- **两个瓶颈、一个分层**：并行的动机只有两条——算力不够（单卡离 exaflops 差太远）与显存放不下；由此引出贯穿全讲的**节点内（快）vs 节点间（慢）**之分，而所有策略的本质都是"把通信成本分配到合适的链路上"；
- **all-reduce = reduce-scatter + all-gather**：在带宽受限情形下这是最优分解，也是 ZeRO 系列"几乎免费"的数学基础——stage 1 与朴素 DDP 的通信量同为 $2\times\#\text{params}$，却把优化器状态显存从 $(4+K)$ 降到 $(4+K/N_{\text{gpu}})$；
- **显存记账是理解 ZeRO 的前提**：参数 2 字节 + 梯度 2 字节 + FP32 主权重 4 字节 + Adam 两个矩各 4（或 2）字节 ≈ 16 字节/参数、约 5 份拷贝，其中**优化器状态是大头**；纯 BF16 训练（配 Kahan summation）可压到 12 字节/参数；
- **ZeRO 三阶段**：stage 1 只切优化器状态（免费）、stage 2 再切梯度（增量反向、边算边发边释放，几乎免费）、stage 3 即 **FSDP** 切一切（每层两次 all-gather + 一次 reduce-scatter，通信 $3\times\#\text{params}$，靠"增量计算/通信 + 立即释放"与"**通信与计算重叠**"两个想法把开销几乎藏掉）；在 8×A100 80G 上，从 baseline 的 6.7B 参数一路到 stage 3 的 53.3B；
- **数据并行的天花板**：它消耗**全局 batch size**（batch = 8 时最多 8 卡），而 **critical batch size** 决定了无限 batch 不如无限步数；且 ZeRO 不减少激活显存——这正是模型并行的动机；
- **流水线并行**：沿深度切层、传激活；朴素切层利用率只有 $1/n$；micro-batch 流水线把**气泡占比**压到 $\frac{n_{\text{stages}}-1}{n_{\text{stages}}+n_{\text{micro}}-1} \approx \frac{n_{\text{stages}}-1}{n_{\text{micro}}}$，所以需要大 batch；它通信量小（$bsh$、点对点）因此**放在最慢链路上**；zero-bubble 把反向拆成"传播激活梯度（必须尽快）"与"算权重梯度（随时可做）"，几乎填满流水线；
- **张量并行**：沿宽度切矩阵、每块 block 四次 all-reduce；前向 $f$ 恒等/$g$ all-reduce、反向对调；列切 QKV 与 up-projection、行切注意力输出与 down-projection；通信量 $8bsh \cdot \frac{N-1}{N}$ 比流水线大得多，**只在 NVLink 域内（≤8 卡）用**；
- **激活显存**：每层存一切为 $34bsh + 5as/h$（二次项可重计算掉）；张量并行不切那 $10sbh$ 的逐点项（LayerNorm/Dropout/块输入），**序列并行**沿序列轴切这些项，两者组合才让激活显存真正线性缩放；
- **专家并行**：切专家、路由 token，对 MoE 比 TP 更高效；但要尊重"EP ≤ DP（共享副本）"的约束；注意力要高 TP、MLP 要低 TP，于是 Megatron 把注意力侧的 TP/CP/DP 与 MLP 侧的 ETP/EP/EDP **解耦**；实现极难（DeepSeek DPP 甚至用到未文档化 PTX 指令）；
- **组合的简单处方（3D/4D 并行）**：模型装不下时先用 TP/EP 吃满快速互联（8 卡/机 → TP/EP = 8），再上 PP 或 ZeRO-3 跨机器，装下之后剩下的 GPU 全用 DP，batch 太小就梯度累积；Narayanan 2021 验证了 TP 封顶 8、PP 负责装模型、DP 最后铺满的缩放规律，且"3D 并行"能带来平直的线性利用率；
- **真实案例的规律**：DP 用到极致（Llama 3 405B 主预训练 DP=128）、TP 几乎都 ≤ 8、MoE 时代 EP 可以很大（DeepSeek-V3 EP=64、Nemotron 长上下文 EP=64）、长上下文阶段普遍上调 CP；Llama 3 405B 训练期间 GPU 挂了 148 次，容错与并行同等重要。

## 课程导航

- [上一讲：07 Parallelism](../07/)
- [下一讲：09](../09/)
