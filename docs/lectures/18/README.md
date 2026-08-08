---
title: "18 Guest Lecture: Dan Fu"
lecture: 18
---

# Lecture 18: Guest Lecture - Dan Fu（推理服务）

**讲师**：Dan Fu（UCSD / Together AI，客座讲授） · **主题**：推理服务全景——"一个 token 的一生"（调度、KV cache、prefill/decode 分离、跨节点并行、推理引擎、MegaKernel 与循环语言模型 Parcae）

## 客座讲次说明

这是本课程《Language Modeling from Scratch》的**最后一讲**，而且是一场**客座讲座（guest lecture）**：主讲人不是常规讲师 Percy 或 Tatsu，而是 **Dan Fu**——UCSD 助理教授、Together AI 研究成员（这一点以转录中的实际证据判断）。课程此前都在讲"如何训练语言模型"，本讲则换到"另一面"：**一旦你有了模型，服务（serving）它、做推理（inference）意味着什么**——把"电"变成 token、再把 token 变成"智能"。

Dan 的讲法很务实：先给出一张"**一个 token 的一生**"全景图——一个请求如何被调度到 GPU、如何在 KV cache 里命中复用、如何经历 prefill 与 decode、如何跨节点并行、最终把 token 送回用户；然后用三个视角把它讲透：**工作负载的多样性**（编码 agent、摘要、聊天、回合制 agentic 流程的 token 分布、节奏与 SLA）、**系统的复杂度**（连续批处理、KV cache 分层存储、模型切分、容错、只有大规模才会冒出的内核 bug），以及**两个具体的研究项目**——来自 Together 的 **MegaKernel / ThunderKittens**（用算子融合把 decode 逼近内存带宽极限）和来自 UCSD 实验室的 **Parcae**（用状态空间模型的数学稳定循环 Transformer，并给出"数据与循环应联合缩放"的初步缩放定律）。全课反复强调同一句话：**理解推理与 GPU kernel，就能在机器学习算法上做全栈创新**。

## 本讲内容

| 页面 | 内容 |
|------|------|
| [18 · 开场与动机：为什么推理值得研究](01-introduction-and-motivation.md) | 客座讲师 Dan Fu（UCSD/Together）介绍、模型能力的"新工业革命"、**规模曲线（2018 年 1 亿 → 今天万亿参数）**、1902–1912 曼哈顿马车与马粪的历史类比、"**GPU 是新的石油、推理是把电变成智能的引擎**"、模型即 DAG、**全栈创新**是本讲最重要的信息、本讲路线图与组织介绍 |
| [18 · 一个 token 的一生：推理引擎全景与工作负载](02-lifetime-of-a-token.md) | **推理引擎全部环节**（请求调度 → KV cache 命中检查 → 执行 ML 计算 → 跨节点/节点内并行 → 输出后处理）、生产负载的 token 分布（编码 agent 长输入 vs 摘要 vs 聊天）、**回合式 agentic 工作流**、不同应用的节奏与会话长度、**SLA（TTFT < 1s 等）**、单个请求的处理流程（分词、缓存检查、prefill/decode、停止符与安全检查、调度-执行-采样循环） |
| [18 · prefill 与 decode：连续批处理与访存分析](03-prefill-decode-and-batching.md) | prefill（**计算受限**）与 decode（**访存受限**）的算术强度推导（$\text{intensity}_{\text{decode}} \approx 1$ vs $\approx S$、70B 单步 decode 下限约 42ms）、时间结构（一次 vs 每 token 一次）、**连续批处理**（时间流下的动态调度、计算/内存两类资源、排队）、延迟-吞吐权衡、**KV cache 与前缀共享**（前缀树查表）与 KV cache 大小公式 |
| [18 · KV cache 与模型并行：切分、共享与大规模部署](04-kv-cache-and-parallelism.md) | KV cache 的**请求间共享**、为什么必须切分模型（单卡装不下万亿参数）、**张量并行**与 **MoE 专家并行**、张量并行的通信账（每层约 2 次 all-reduce、decode 通信/计算比高）、**NVL72 的 72 卡互连**、单卡故障与**容错**、百万 token 超长上下文 |
| [18 · 大规模推理服务：内核 bug 与缓存感知路由](05-serving-at-scale.md) | **分离式部署**（prefill/decode 各用各的 worker）与硬件专门化（NVIDIA 收购 Groq 做 decode、Cerebras、SambaNova）、**大规模下的恶性 bug**（NaN 导致复读、工具调用死亡循环、off-by-one 导致的"突然说中文"）、**KV cache 分层存储**（GPU → CPU → 磁盘）、驱逐与预取（LRU、预测未来）、**缓存感知的 prefill/decode 分离**（两行路由代码换来最多 40% 提速、研究仍处早期） |
| [18 · 研究前沿：MegaKernel 与循环语言模型 Parcae](06-megakernels-and-parcae.md) | **MegaKernel**（kernel-per-operation 的停机时间、把 GPU 当分布式系统调度、QKV+RoPE 与 KV cache 加载重叠、O 投影预取、**ThunderKittens**、H100 上 72% 带宽利用率、人力的代价）、**Parcae**（循环 Transformer 的动机、训练不稳定的动态系统分析 $x_{t+1}=\mathbf{A}x_t+\mathbf{B}u_t$、谱半径与 $2^{16}$ 爆炸、负对角参数化稳定化、质量提升、**数据与循环联合缩放的幂律**、所有现役模型都无循环）、七问七答（预训练模型循环、推理内存含义、MegaKernel 代价、硬件感知设计、计算最优、use case 架构差异、多 GPU 通信）、全讲总结 |

## 本讲要点

- **推理是把电变成智能的引擎**：模型只是运算的 DAG；推理引擎与 GPU kernel 才是真正把 GPU 从"沙子"变成可用东西的软件——**理解它们就能做机器学习算法的全栈创新**；
- **"一个 token 的一生"**：请求经历**调度 → KV cache 命中检查 → prefill/decode 执行 → 并行化选择 → 输出后处理**；引擎本质是"调度-执行-采样"的循环；
- **生产工作负载与训练完全不同**：编码 agent 是数万输入 token 的回合制流程，摘要/聊天/批处理各有各的 token 分布、节奏与 SLA（交互应用要 TTFT < 1s）；
- **prefill 计算受限、decode 访存受限**：decode 算术强度 $\approx 1$ FLOPs/byte（远低于 H100 的约 295），每步必须把全部权重读一遍（70B 单步下限约 42 ms）；prefill 强度 $\approx S$，容易算满 GPU；
- **连续批处理**：请求动态进出、短请求不被长请求拖住，同时占用计算资源与 KV cache 内存资源，内存不够就排队；
- **KV cache 与前缀共享**：用前缀树查"见过哪些 token"，复用激活；KV cache 大小 $= S \cdot KH \cdot L \cdot 4$ 字节/序列，是并发能力的瓶颈，也是"尽可能大"的优化目标；
- **模型必须切分**：张量并行（每个张量 4 等分）与 MoE 专家并行；切分方式决定系统瓶颈，**decode 阶段通信/计算比高、通信易成瓶颈**；
- **prefill 与 decode 可以分离部署**：两段计算特征不同，甚至可以硬件专门化——NVIDIA 买下 Groq 用 LPU 做 decode、Cerebras、SambaNova 都在下注；
- **大规模才会冒出的 bug**：$10^{-8}$ 量级的罕见触发（NaN 导致模型复读同一 token、工具调用处理的死亡循环、off-by-one 读到未初始化内存导致"突然说中文"）——小规模好使、大规模必坏；
- **KV cache 的分层存储是操作系统问题**：GPU → CPU DRAM → 磁盘；驱逐用 LRU（OS 论文证明 2 倍最优）、理想是预测未来（打开旧对话是预取信号）；
- **两行代码的路由优化**：把低缓存命中率的新请求与"热"请求分到不同 prefill 节点，服务速度最多提升 40%——这类研究还处于早期；
- **MegaKernel**：一个 kernel 覆盖多个运算，把 GPU 当作可调度的大规模分布式系统（QKV 未算完先加载 KV cache、注意力未结束先加载 O 投影权重）；ThunderKittens 用指令级抽象与虚拟化共享内存实现，H100 上达 72% 带宽利用率、30%–70% 加速，代价是极高的写 kernel 人力成本；
- **Parcae 让循环模型可训练**：循环 Transformer 的参数不变、FLOPs 可调、表达力更强，但一碰就炸；把非线性塞进盒子后得到线性动态系统 $x_{t+1}=\mathbf{A}x_t+\mathbf{B}u_t$，闭式解被 $\mathbf{A}^t$ 主导，谱半径 > 1 就指数爆炸——把 $\mathbf{A}$ 参数化为负对角矩阵、给 $\mathbf{B}$ 加线性归一化，谱半径 < 1，训练稳定（$6\times10^{-4}$ 学习率也稳定）；
- **数据与循环应联合缩放**：等参等 FLOP 曲线一致指向"固定参数时数据越多、循环也应越多"；今天的模型全都没有循环、却用了海量数据——预训练也许存在更好的做法；
- **循环模型对推理友好**：参数更少意味着更多 KV cache、更少 GPU 通信，甚至可能设计出恰好装进 Groq LPU（约 250 MB 内存）的模型；
- **架构与服务互相塑造**：agentic 工作流要 KV cache 热（MLA、FP8/FP4 缓存重要），批处理用双向注意力（BERT）一次编码即可，聊天则永远有一段 decode。

## 课程导航

- [上一讲：17 Alignment - Multimodality](../17/)
- 本讲为课程最后一讲
