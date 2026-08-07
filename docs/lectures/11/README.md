---
title: "11 Scaling Laws"
lecture: 11
---

# Lecture 11: Scaling Laws（进阶）

**讲师**：Tatsu Hashimoto · **主题**：缩放定律（进阶）——案例研究与细节

## 本讲内容

上一讲（第十讲）Percy 讲了推理，这一讲 Tatsu 回到缩放定律，但这次是**进阶版**：不再是最基础的幂律与计算最优，而是“把缩放定律用在真实构建开源语言模型上”的那些细节。两讲之前的第九讲（基础缩放定律）讲的是 Kaplan/Hestness/Chinchilla 这版“经典正典”（知识大致停在 2022 年）；本讲则从 2022 年之后、主要来自中国开源社区的一批缩放论文里精选内容，目标是让你从 Chinchilla 一路“速通”到最新的、带缩放细节的模型（比如 Kimi K2）。这一讲的核心问题是：**你在小规模上把超参调好之后，怎么保证放大到真实模型规模时不翻车？** 答案围绕两条路线展开——要么用 μP 这类参数化让超参“对尺度不变”，要么在小规模上拟合出 batch/学习率的缩放律直接外推；而这一切都离不开对初始化、优化器、batch size 这些**尺度敏感**对象的仔细思考。

讲次分为两大块。**第一块是详细、公开的缩放配方**：MiniCPM（2024，用 μP 稳定学习率 + 缩放梯子 + Kaplan 式临界 batch 分析）、WSD 学习率（warmup-stable-decay，让 Chinchilla 式扫描从“二次方代价”变成“反复衰减”的廉价操作）、DeepSeek（2024，不用 μP、直接网格搜索并拟合 batch/LR 的缩放律，再复现 Chinchilla 的 isoFLOP 分析）、以及更近期的 Qwen、Kimi K2（MoE 稀疏度缩放，选出稀疏度 48）、Hunyuan（96∶1 数据/激活参数比）、LLaMA 3（isoFLOP 与“损失→下游”的 sigmoid 映射）、MiniMax-01（架构缩放定律）。**第二块是优化器与初始化**：StepFun 的大规模超参网格搜索（损失关于 batch/LR 是凸的、最优 batch 几乎只依赖数据量、学习率随模型增大而减小/随数据增多而增大的反直觉趋势）；优化器尺度依赖的三大问题（不同优化器需要不同超参、计算量与 Chinchilla 比两条缩放轴、建立缩放本身的困难——Cautious AdamC 的“好看缩放突然爆炸”例子）；Muon（基于 Newton-Schulz 迭代的矩阵正交化优化器，nanoGPT speedrun 上的大收益与大规模证据）；以及最后的数学核心——**μP 的完整推导**（条件 A1/A2、初始化与学习率缩放规则的逐步推导、与标准参数化的对比、RMSNorm gain/符号优化器/强 weight decay 三种失效模式），收尾总结“scaling in the wild”的三个挑战与三类对策。

| 页面 | 内容 |
|------|------|
| [11 · 引言与动机：缩放的“正典”与进阶](01-introduction-and-motivation.md) | 本讲三个核心问题（Chinchilla 在实际规模上有效吗、能否省计算、是否该选特定参数化）、经典缩放正典的边界（Kaplan/Hestness/Chinchilla 停在 2022）、2022 后开源社区的新局面、本讲两大部分的地图、初始化与尺度敏感性的总起 |
| [11 · MiniCPM：用 μP 稳定缩放](02-minicpm-scaling-recipe.md) | MiniCPM（2024）与缩放梯子（最大梯子模型与发布模型差约 5 倍）、**μP 的具体清单**（embedding 输出缩放、按 $\sqrt{\text{层数}}$ 缩放残差、fan-in/fan-out 初始化、逐张量学习率、LM head）、最优学习率实验（最小损失稳定在 $10^{-2}$）、**Kaplan 式最优 batch 分析**（临界 batch 随目标损失呈幂律） |
| [11 · WSD 学习率与 DeepSeek](03-wsd-and-deepseek.md) | cosine 学习率的“二次方代价”（每改数据量都要从头重训）、**WSD 学习率详解**（固定步数预热 + 稳定 + 10%–20% 快速衰减，可随时回滚续训）、WSD vs cosine 的性能对比、用 WSD 廉价地做 Chinchilla 方法一/三、**DeepSeek 路线**（不用 μP、网格搜索并拟合 batch/LR 缩放律、两次 10% 衰减的 WSD 变体、isoFLOP 方法二、缩放律预测最终损失） |
| [11 · 近期的开源缩放配方](04-recent-scaling-recipes.md) | 为什么 2024 后论文“细节变少”、Qwen 2.5/3 的标准配方、**Kimi K2 的 MoE 稀疏度缩放**（选稀疏度 48）、Hunyuan（96∶1）、LLaMA 3（isoFLOP 与“损失→下游”sigmoid）、MiniMax-01（架构缩放定律）、**DeepSeek 配方 vs MiniCPM 配方总结表**、后训练的课堂问答 |
| [11 · StepFun 超参研究与优化器缩放](05-stepfun-and-optimizer-scaling.md) | StepFun 大规模超参网格搜索：三种“batch/LR 缩放函数形式”之争、**观察 1 损失凸性**、**观察 2 最优 batch 几乎只依赖数据量 $D$**（$B_{opt} \sim \sqrt{D}$）、学习率的反直觉趋势、**观察 3 对 MoE 与数据集的鲁棒性**、学习率的普遍鲁棒范围（$10^{-3}$～$10^{-4}$）；优化器尺度依赖的三大问题（超参调优跑偏、计算量与 Chinchilla 比两条轴、建立缩放的不平凡性——Cautious AdamC 的缩放爆炸）；“直接用缩放律还是自己网格搜索”的课堂问答 |
| [11 · Muon 与 μP 深入推导](06-muon-and-mup.md) | nanoGPT speedrun 里的 Muon、**Muon 与 Newton–Schulz 正交化的完整推导**（$B = USV^{\top} \to UV^{\top}$、逆平方根的 Newton 迭代与收敛性、PyTorch 示意代码）、Muon 的缩放证据（小规模/缩放研究/Kimi K2）、**μP 的完整推导**（条件 A1/A2、初始化缩放 $\sigma = \Theta(\frac{1}{\sqrt{n_{l-1}}}\min(1,\sqrt{n_l/n_{l-1}}))$、SGD 学习率 $\eta_l = \Theta(n_l/n_{l-1})$、Adam 学习率 $\Theta(1/n_{l-1})$）、mini recap 对照表、CerebrasGPT 证据、μP 的鲁棒性与三种失效模式（RMSNorm gain、Lion、强 weight decay）、收尾“scaling in the wild” |

## 本讲要点

- **两条处理“敏感超参”的路线**：MiniCPM 路线用 μP 把学习率“稳定”下来（最优学习率不随规模漂移）；DeepSeek 路线不靠 μP，直接在小规模上网格搜索最优 batch/LR，再拟合它们的缩放律并外推——两者都是正经做法；
- **μP 是“先断言不变量、再解出超参约束”的推导风格**：条件 A1（初始化激活 $\Theta(1)$）给出初始化 $\sigma = \Theta\left(\frac{1}{\sqrt{n_{l-1}}}\min\left(1,\sqrt{\frac{n_l}{n_{l-1}}}\right)\right)$；条件 A2（一步更新后激活变化 $\Theta(1)$）在“每步损失更新 $O(1)$”的假设下给出 SGD 学习率 $\eta_l = \Theta(n_l/n_{l-1})$、Adam 学习率 $\Theta(1/n_{l-1})$——fan-in 大的层用更小的学习率；
- **WSD 学习率是廉价做 Chinchilla 分析的关键**：cosine 需要预先知道总预算、改数据量就得从头重训（近似二次方代价）；WSD 的稳定阶段结束后可随时回滚续训，只重复“快速衰减”（约 10% 成本）——稳定阶段看起来落后，衰减阶段一举追回甚至反超 cosine；
- **临界 batch 与学习率是最敏感的两个缩放对象**：MiniCPM 用 Kaplan 式分析得到“最优 batch 随目标损失呈幂律”；StepFun 更进一步发现，Chinchilla 式联合缩放下**最优 batch 几乎只依赖数据量 $D$**（$B_{opt} \sim \sqrt{D}$），而学习率随模型增大而减小、随数据增多而增大（反直觉且可能脆弱）；
- **优化器是尺度依赖的**：比较优化器时必须警惕两条轴——计算量轴与 **Chinchilla 比（token/参数）** 轴（常常是性能的大混杂因素）；Muon 在小规模（nanoGPT speedrun）收益巨大，但缩放研究中增益缩水——不过它已经进入 Kimi K2 这样的大规模训练；
- **建立缩放本身不平凡**：Cautious AdamC + $\sqrt{\text{batch}}$ 学习率缩放这条“很漂亮”的缩放律，在 $10^{20}$ 量级 FLOPs 后突然爆掉——好看的缩放趋势也可能在多个数量级后咬你一口；做算法开发时永远检查相对计算量与 Chinchilla 比的缩放；
- **μP 的鲁棒性与失效**：SwiGLU、大小 batch、零注意力初始化等大多与 μP 兼容；RMSNorm 可学习 gain、基于符号的优化器（Lion）会破坏 μP；**强解耦 weight decay（0.1）是 μP 最显著的失败**；
- **缩放定律像科学、实为艺术**：核心缩放机制（Chinchilla、LR 缩放）如今人人会做、论文不再详述；但外推是否成立永远是“信念”问题——用小规模搜索最优 LR/batch（保持固定或预测缩放）、用 μP 或假设稳定性、用 WSD 类调度，是控制超参漂移的现实手段，目前没有银弹。

## 课程导航

- [上一讲：10 Inference](../10/)
- [下一讲：12 Evaluation](../12/)
