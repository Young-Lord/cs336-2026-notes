---
title: "04 Attention Alternatives"
lecture: 4
---

# Lecture 4: Attention Alternatives and Mixtures of Experts

**讲师**: Tatsu Hashimoto · **主题**：注意力替代方案与混合专家模型（Attention Alternatives and Mixtures of Experts）

## 本讲内容

这一讲是 2026 版课程最重要的一讲之一。前半讲是完全新增的内容——**线性时间注意力（linear-time attention）与状态空间模型（SSM）/混合架构**：从控制长上下文成本的动机讲起，然后由**乘法的结合律**这一个核心思想出发，推导出线性注意力的复杂度优化（$O(n^2) \to O(nd)$）、它与 **RNN 的等价形式**（固定大小的状态、训练/推理"对偶性")，再一路推广到 **Mamba-2、Gated DeltaNet** 这些如今已在生产环境大规模验证的混合架构，并讨论混合架构的**比例设计**；最后介绍另一个替代方案——**稀疏注意力（DSA）**。后半讲是 **MoE（混合专家模型)**：它本质上是一个"更高效的 MLP"，先讲清动机与工业现状，再深入 **Top-K 路由公式、共享/细粒度专家、负载均衡损失（含导数推导）、稀疏门控不可微的困境**，以及系统侧（专家并行、稀疏矩阵乘、token dropping 随机性）、稳定性（router 的 float32 与 z-loss）、微调与 upcycling，最后完整走一遍 **DeepSeek MoE v1 → v2 → v3** 的演进与 **MLA、MTP** 两个额外组件。

| 页面 | 内容 |
|------|------|
| [04 · 开场与动机：为什么需要注意力替代方案](overview.md) | 上下文窗口竞争、FFN vs 注意力的成本对比、局部+全局注意力混合、FlashAttention 与常数因子的重要性、通向线性时间注意力 |
| [04 · 线性注意力：从结合律到 RNN 等价](linear-attention.md) | 注意力回顾、$QK^\top V = Q(K^\top V)$ 的结合律重排、$O(n^2) \to O(nd)$ 复杂度推导、RNN 等价形式与状态携带、训练/推理对偶性、Minimax M1 的 7:1 混合 |
| [04 · 状态空间模型与混合架构：Mamba-2 到 Gated DeltaNet](ssm-hybrid.md) | Mamba-2 的输入相关门控 $\gamma_t$、Nemotron 3、Gated DeltaNet 的第二门控 $\beta_t$ 与擦除投影子、Qwen 3.5/Qwen Next 的 3:1 混合、混合架构比例设计研究 |
| [04 · 另一种替代方案：稀疏注意力 DSA](sparse-attention.md) | DeepSeek Sparse Attention 的轻量 indexer + Top-K 机制、事后（post-hoc）适配、DeepSeek V3.2 与 GLM 5 的证据、以及"常数因子很重要"的系统教训 |
| [04 · MoE 基础与路由：什么是 MoE、为什么流行](moe.md) | MoE = 更高效的 MLP、同样 FLOPs 更多参数的直觉、四大流行原因、西/中工业现状、为什么 MoE 曾不流行、路由三范式与 Top-K 公式、共享/细粒度专家、配置表 |
| [04 · 训练 MoE 与系统稳定性：从负载均衡损失到 DeepSeek 演进](moe-training.md) | 稀疏且不可微的门控困境、RL 与随机扰动、Switch 平衡损失的导数推导、DeepSeek 逐专家/逐设备/偏置平衡、专家并行与稀疏矩阵乘、token dropping、稳定性、微调、upcycling、DeepSeek v3 与 MLA/MTP |

## 本讲要点

- 本讲两大主题对应 transformer 的两个块：**注意力替代方案**改的是注意力块（用线性时间注意力/状态空间模型处理长上下文），**MoE** 改的是 MLP 块（用稀疏激活换取更多参数）；
- 长上下文成本的核心矛盾：**FFN 成本随序列长度线性增长，而注意力是位置间的全对全交互，呈二次方增长**——序列一长，注意力就迅速反超并主导成本；
- 控制成本有两种基本手段：**局部注意力 + 全局注意力的混合**（如每 8 层才做一次全局注意力）与**系统工程**——**FlashAttention 这类常数因子改进带来的收益常被低估**：它只重排运算、不改变二次方本质，却能把吞吐翻倍，甚至在注意力矩阵放不进内存时继续训练；
- 线性注意力只需一个核心思想——**乘法的结合律**：丢掉 softmax 后，$QK^\top V = Q(K^\top V)$，复杂度从 $O(n^2 d_k + n^2 d_v)$ 降到 $O(2n d_v d_k)$——把"对序列长度 $n$ 的二次方依赖"换成"对隐藏维度的线性依赖";
- 线性注意力的另一个惊人性质是 **RNN 等价**:$S_t = S_{t-1} + k_t v_t^\top,\ y_t = q_t^\top S_t$，状态 $S$ 大小固定、随 token 依次推进；于是可以**训练用并行的密集矩阵乘形式、推理用串行的 RNN 形式**——"鱼与熊掌兼得";
- **从线性注意力到 Mamba-2 只是一小步**：给状态更新加一个只依赖当前输入的门控 $\gamma_t = f(x_t)$($S_t = \gamma_t S_{t-1} + k_t v_t^\top$)，表达力大增且对偶性保持；Gated DeltaNet 再加一个"无输入操作"门 $\beta_t$ 与擦除投影子 $(I - \beta_t k_t k_t^\top)$——**只要门控只依赖输入、不依赖状态，对偶性就还在**;
- 目前**没有任何人证明过纯线性时间注意力能在大规模上工作**:Minimax M1（7:1）、Nemotron 3（~3:1）、Qwen 3.5/Qwen Next（3:1）全是"线性层 + 少量全注意力"的混合，而混合架构的低比例段几乎没有性能损失、高比例段才明显退化；
- **DSA（稀疏注意力）是线性注意力的替代路线**：先用轻量 indexer 做 Top-K 选出少数 token，再在小子集上做全注意力——它不是线性时间（indexer 仍做全对全 QK 内积），但常数因子极好，且可以在**长上下文扩展阶段事后接入**、不必从头预训练；
- **MoE 本质上是"更高效的 MLP"**：把一个大 FFN 换成 $N$ 个 FFN 与一个路由器，参数乘 $N$ 而每次前向/反向只付 $K$ 个专家的成本——**增加参数而不增加 FLOPs**；同样的证据（参数量↑ loss↓）与训练加速在 Fedus 2022、OlMoE、DeepSeek 中反复出现；
- 几乎**所有 MoE 都收敛到 token 选择的 Top-K 路由**：亲和度 $s_i(t) = \mathrm{softmax}_i(u_t^\top e_i)$,softmax 放在 Top-K 之前还是之后只是"美学选择";DeepSeek 发扬光大的**细粒度专家 + 共享专家**（共享专家永远在线、绕过路由器）如今是事实标准；
- 训练 MoE 的核心矛盾：**训练时也要稀疏（否则 FLOPs 爆炸），但稀疏门控不可微**;RL 与随机扰动都不划算，实践几乎只用**启发式负载均衡损失**——Switch 的 $\alpha N \sum_i f_i P_i$ 对概率的导数正比于专家使用频率 $f_i$,"越常用压得越狠";
- **移除负载均衡损失是灾难性的**(OlMoE 消融)：几乎所有 token 涌向一两个专家，其余专家"死亡"，白扔参数；DeepSeek v3 的"无辅助损失平衡"（逐专家偏置 + 在线学习）也并非真的完全无辅助损失；
- MoE 的系统代价：专家并行依赖 all-to-all 通信（可与稀疏矩阵乘、MegaBlocks 结合），负载不均导致 **token dropping**——你的推理结果甚至可能受同一 batch 里别人的查询影响；路由 softmax 需要 **float32 + z-loss** 来稳定；
- **upcycling（从稠密模型"升级"成 MoE)** 是一条省钱的训练路径：复制 MLP、随机初始化路由器、继续训练即可——MiniCPM 2.4B→13.4B、Qwen 1.8B→2.7B 都是成功案例，但如今大家更倾向直接从头训练 MoE。

## 课程导航

- [上一讲：03 Architectures, Hyperparameters](../03/)
- [下一讲：05 GPUs, TPUs](../05/)
