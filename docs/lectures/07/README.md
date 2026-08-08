---
title: "07 Parallelism"
lecture: 7
---

# Lecture 7: Parallelism

**讲师**：Percy Liang · **主题**：并行（Parallelism）

## 本讲内容

这一讲是系统部分的第三站：上一讲（第六讲）我们学会让**一块** GPU 变快（写 kernel、tiling、算子融合），这一讲把视野从一块 GPU 扩展到**多块 GPU**——如何在四块、甚至上千块 GPU 上把训练搬起来。本讲是一堂"可执行讲座（executable lecture）"：课程讲义本身是一段 Python 程序，幻灯片内容、collective 示意图、基准测试、三种并行策略的朴素实现全部写在代码里，可以直接运行（直接运行会走 multiprocessing；逐行 trace 时则退化为单进程、把分布式调用替换为空操作）。

讲次分为两大部分。**第一部分**是分布式通信与计算的积木（building blocks）：先讲**集合通信原语**（collective operations）这个源自 1980 年代并行编程的编程模型（broadcast、scatter、gather、reduce、all-gather、reduce-scatter、all-reduce、all-to-all）；再讲**硬件**——GPU 究竟怎么连在一起（NVLink/NVSwitch、InfiniBand、Ethernet、RDMA，以及 NCCL 这个把 collective 落到包级别的库）；最后落到 **PyTorch**——用 `torch.distributed` 亲手实现这些操作，并像上一讲算 MFU 一样测量**有效通信带宽**。**第二部分**是三种分布式训练策略：**数据并行**（沿 batch 维切）、**张量并行**（沿宽度维切）、**流水线并行**（沿深度维切），全部用深度 MLP 演示（MLP 正是 Transformer 里真正的计算瓶颈，因此很有代表性），并分别讨论它们的**通信量**与**显存**代价。

| 页面 | 内容 |
|------|------|
| [07 · 开场与动机：为什么需要多 GPU 并行](overview.md) | 从"单 GPU 写 kernel"到"多 GPU 并行"、计算离数据远这一统一主题、内存/互联的广义层级（L1/shared → HBM → NVLink → InfiniBand/Ethernet）、多 GPU 的两大理由（放不下、想更快）、本讲两大部分路线图、可执行讲座的 multiprocessing 说明 |
| [07 · 集合通信原语：collectives 的编程模型](collectives.md) | rank 与世界大小（world size）、broadcast/scatter/gather/reduce 四个"热身"原语、all-gather/reduce-scatter/all-reduce 三个"主力"原语（含 all-reduce = reduce-scatter + all-gather）、all-to-all（MoE 的转置）、术语记忆法、课堂问答 |
| [07 · 硬件：GPU 如何连接](hardware.md) | 经典家庭拓扑（PCIe/Ethernet）vs 现代数据中心拓扑（NVLink/NVSwitch/InfiniBand/Ethernet）、关键带宽数字（NVLink 5.0 的 1.8 TB/s vs HBM 的 8 TB/s）、绕过 CPU 的 RDMA、NVL72 与 RoCE、NCCL 的职责、课堂问答 |
| [07 · torch.distributed：实现 collectives 与通信带宽](torch-distributed.md) | `torch.distributed` 的接口与后端（gloo/nccl）、spawn/setup/barrier/cleanup、all-reduce/reduce-scatter/all-gather 代码逐行讲解、基准测试的正确计时、有效带宽公式、**ring all-reduce 通信量完整推导**（$2\frac{N-1}{N}M$）、all-gather/reduce-scatter/all-to-all 通信量、课堂问答 |
| [07 · 数据并行（DDP）与张量并行](data-and-tensor-parallelism.md) | 数据并行：按行切数据、前向/反向、用 one-line 的 all-reduce（AVG）同步梯度——与普通训练的**唯一区别**；通信与显存分析；FSDP/ZeRO 预告；张量并行：按列切参数、每层 all-gather 激活、前向 all-gather 与反向 reduce-scatter 的对偶性；为何张量并行必须依赖快速互联；课堂问答 |
| [07 · 流水线并行与课程小结](pipeline-parallelism.md) | 流水线并行：按层切、micro-batch 与 send/recv、**pipeline bubble** 及其公式 $\frac{N-1}{N+K-1}$、通信/计算重叠为何关键；本讲"缺什么"（重叠、序列/专家并行、组合策略、硬件依赖、critical batch size）；PyTorch 与 JAX/TPU 的哲学对比；全课总结（recompute / store / communicate 三角） |

## 本讲要点

- **多 GPU 的动机只有两个**：① 模型（参数 + 梯度 + 优化器状态 + 激活）放不进单卡显存——B200 有 192 GB，但 1 万亿参数的模型远远放不下；② 即使放得下，也要用更多 GPU 的算力训练得更快。两者的权衡（少卡省通信 vs 多卡多通信）就是你要做的计算；
- **核心心智模型**：无论单卡还是多卡，"计算离数据远"这一本质没变——单卡时数据在 HBM，多卡时数据可能在另一块 GPU 上，都需要把数据搬来搬去。游戏始终是**编排计算以避开数据传输瓶颈**：上一讲靠融合与 tiling 减少访存，这一讲靠**复制（replicate）与切分**（shard）减少跨 GPU 通信；
- **广义内存层级**：L1 cache/shared memory（最快）→ HBM（上一讲嫌它慢，这一讲要把它当"快"的）→ NVLink/NVSwitch（单节点多卡，B200 的 NVLink 5.0 约 1.8 TB/s，约为 HBM 8 TB/s 的 1/4）→ InfiniBand（跨节点，约 0.05 TB/s）→ Ethernet（跨 pod，最慢）；"GPU 越多越慢"和"内存越大越慢"是同一件事；
- **collective 是 1980 年代的并行编程原语**，今天仍是大模型训练的基石："collective"意味着你只声明一种**通用的通信模式**，由系统去编排，而不是手动管理点对点通信。用三个词就能记住术语：**reduce** 做可结合可交换的运算（sum/min/max），**scatter** 是 **gather** 的逆（分发 vs 汇聚），**all** 表示目的地是所有设备；
- **三种主力原语**：**all-gather**（每块 GPU 持有参数分片，前向需要完整参数时把它汇聚到所有卡）、**reduce-scatter**（反向后把各卡不同数据算出的梯度按分片求和并分散存储）、**all-reduce = reduce-scatter + all-gather**（DDP 每步同步梯度用的就是这个；拆开做正是 FSDP/ZeRO 的灵活性来源）；**all-to-all** 是 MoE 里"把数据路由给专家"的操作，均衡切分时等价于矩阵转置；
- **ring all-reduce 的通信量**：设世界大小 $N$、每卡张量 $M$ 字节，环形拓扑上 all-reduce 分两阶段各 $N-1$ 步、每步搬 $M/N$，总时间 $T = \frac{2(N-1)M}{N \cdot B}$，每卡通信量 $2\frac{N-1}{N}M$；当 $N$ 很大时 $(N-1)/N \to 1$，于是 $T \to 2M/B$、通信量 $\to 2M$——**有效带宽与世界大小无关、与拓扑（ring/tree）无关**，这正是 NCCL 的"魔术"；
- **硬件三层结构**：每节点 8 卡经 NVLink 接到 NVSwitch（一个 NVLink 域），256 节点组成 pod 经 InfiniBand（PCIe → HCA → 光缆）相连，多个 pod 组成集群/数据中心经 Ethernet 相连；跨节点想绕过 CPU 需要 **RDMA**（Infiniband 原生支持，普通 Ethernet 不支持，**RoCE** 是它的以太网版本）；NVL72 把 72 卡（8 卡/托盘 × 9 托盘）放进同一个 NVLink 域；
- **NCCL** 把"我要 all-reduce"翻译成真正的包：探测拓扑 → 选择最优路径（ring 或 tree）→ 启动 GPU kernel 收发数据。PyTorch 的 `torch.distributed` 是对它的干净封装（gloo 后端跑 CPU、nccl 后端跑 GPU），课程刻意只用这些**原始原语**、不碰 FSDP 等高级封装，因为"from scratch"要让你看见机械原理；
- **数据并行（DDP）**：按行把 batch 切成 $N$ 份，每卡一份；每卡照常前向/反向，然后在 backward 之后插入**唯一的一行改动**——对每个参数做 `dist.all_reduce(op=AVG)` 同步梯度，再照常 `optimizer.step()`。损失各卡不同、梯度先不同后同步、参数永远一致；代价是每步一次全量梯度 all-reduce（通信量 $\approx 2|\theta|$）、每卡持有完整模型（显存 $N$ 份副本），这正是下讲 FSDP/ZeRO 要解决的；
- **张量并行**：把每层的参数矩阵**按列**切开（column tensor parallel），每卡只算 $X \cdot W_{\text{local}}$ 的 $B \times D/N$ 部分，再每层做一次 **all-gather** 把激活拼回 $B \times D$；前向 all-gather 与反向 reduce-scatter 是一对对偶操作。它的通信量正比于**激活**（每层都要搬），所以只在 NVLink 这种超快互联里用；
- **流水线并行**：按层切分，每卡只持有部分层、拿到全部数据；前向像接力一样 send/recv。朴素实现会产生 **pipeline bubble**——$N$ 个 stage、$K$ 个 micro-batch 的气泡占比 $\frac{N-1}{N+K-1}$，所以要把 batch 拆成 micro-batch；要真正高效还得**重叠通信与计算**（作业二会探索）；
- **"缺什么"**：通信/计算重叠、注意力等更一般的模型、序列并行与专家并行（MoE）、以及各种并行策略的组合——选择哪种并行强烈依赖硬件：张量并行必须用 NVLink 域内的超快互联，流水线并行能容忍很慢的互联（甚至跨半球），数据并行做到头会撞上 **critical batch size**（batch 再大也帮不上忙）；JAX/TPU 的做法是只定义模型与分片策略、让编译器负责编排，而本课用 PyTorch 手搓正是为了看清底层；
- 收官视角：**"重算，或存内存，或存在另一块 GPU 上再通信"**——数据并行本质上是冗余计算（每卡都更新整套参数），换来的是不用搬优化器状态；硬件再快，我们总是想要更大的模型，所以这种分层结构永远在。

## 课程导航

- [上一讲：06 Kernels, Triton](../06/)
- [下一讲：08 Parallelism](../08/)
