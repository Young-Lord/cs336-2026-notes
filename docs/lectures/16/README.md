---
title: "16 Post-Training - RLVR"
lecture: 16
---

# Lecture 16: Post-Training - RLVR（基于可验证奖励的强化学习）

**讲师**：Tatsu Hashimoto · **主题**：RLVR（PPO 与 GRPO、RLVR 案例研究、DeepSeek R1、Kimi K1.5、Qwen 3、智能体 RL）

## 本讲内容

上一讲（第十五讲）Tatsu 讲了后训练的 SFT 与 RLHF，结尾留下一个略显沮丧的注脚：**RLHF 因为过度优化（overoptimization）无法被干净地缩放**——奖励模型只是人类偏好的近似，往同一个奖励模型里不断投入计算，迟早会过拟合它。这一讲（第十六讲）紧接着解决"那 RL 到底还有没有用"这个问题：答案是 **RLVR（reinforcement learning from verifiable rewards，基于可验证奖励的强化学习）**。在数学、编码这类"可验证"的窄领域里，我们优化的恰恰就是想要的东西——就像 AlphaGo 优化围棋的输赢一样——因此可以无限地投入计算。OpenAI 用思考模型解出重大公开数学难题的新闻，就是 RLVR 最直接的注脚。

这一讲的结构分两半：**第一部分是核心算法**。先复习 PPO：策略梯度（REINFORCE 技巧）是一切的核心，PPO 就是在复用 rollout 的同时把比率裁剪到 $[1-\varepsilon,1+\varepsilon]$ 附近；但"概念上简单"是一回事，"实现上 37 个细节"是另一回事——语言模型上的 PPO 需要价值模型、经验缓冲区、逐 token 的 KL 项，AlpacaFarm 的实现里处处是 KL 裁剪、$\gamma=\lambda=1$ 退化为赌博机之类的"脏细节"。于是研究社区极其渴望替代品：**GRPO** 去掉价值函数，把优势算成**组内 z-score**——减组均值、除组标准差。在线情形下 GRPO 退化成"优势减 KL 惩罚"的极简目标，半个幻灯片就能写完实现（nano-aha-moment，std 里加 $10^{-4}$ 防止除零）。但 GRPO 并不是无懈可击的策略梯度：除以标准差**不是合法 baseline**（它放大过易/过难的问题），长度归一化引入**长度偏差**（长序列拿到的梯度总量更小）——Dr. GRPO 论文把这两处修正后，就非常接近 REINFORCE with leave-one-out。

**第二部分是案例研究**，看上述算法如何体现在开源模型的技术报告里。**DeepSeek R1/R1-Zero**：R1-Zero 是极简配方的极致——基础模型 + GRPO + 准确率奖励 + 格式奖励，只比 o1 差一点；训练中"CoT 越长"和"aha 时刻"两个现象后来被证明大概率是 GRPO 长度偏差的自然副作用、以及预训练里就有的东西。R1 生产化时叠上 SFT 初始化、语言一致性奖励（防止 CoT 语言混杂）、600k 推理数据与 200k 非推理数据的 SFT/RLHF，并放弃过程监督、证明 PRM/MCTS 并非必需；最后用 R1 的 800k CoT traces 蒸馏出会推理的小模型。**Kimi K1.5**：与 R1 同时期、也用 RL 击败 o1。数据上排除选择题/判断题（假阳性）、只保留 best-of-8 都做不对的题；RL 用 DPO 式推导——KL 正则化目标经非参数假设解出闭式解 $\pi^{*}\propto\pi_{\mathrm{ref}}e^{r/\tau}$，再用平方损失代理（baselined policy gradient w/ regularization）；长度上用组内长度奖励进一步压缩 CoT。此外还讲了 RL 基础设施为什么难做（on-policy 意味着慢推理、训练/推理框架切换、长 CoT 让批次不均），以及 RL 一致地优于专家迭代的消融。**Qwen 3 与 Qwen 3 Coder Next**：Qwen 3 采用经过验证的 RLVR 剧本（难度过滤、去污染、手动过滤 CoT），只用约 4000 个例子做 GRPO；特有的"思考模式融合"让思考/非思考共处一个模型、用特殊字符串早停思考，测试时缩放曲线优雅退化；Qwen 3 Coder Next 则展示了智能体 RL 的完整图景——仓库级长上下文 mid-training（6000 亿 token）、四个专家模型（web dev/UX/单轮 QA/SWE）训练后蒸馏回单一模型、自动构造 80 万个 SWE-bench 式环境做 RL，以及"git 历史作弊"这个奖励可验证性问题的绝佳例子。

| 页面 | 内容 |
|------|------|
| [16 · 引言：从 RLHF 到 RLVR](01-introduction-rlvr.md) | OpenAI 解决公开数学难题的新闻、课程位置（ChatGPT → o1/r1）、**过度优化与标注瓶颈**、AlphaGo 对比（搜索问题 vs 学习问题）、可验证领域（形式数学/自然语言数学）、本讲两大部分结构 |
| [16 · PPO 复习：从策略梯度到实现细节](02-ppo-revisited.md) | **策略梯度/REINFORCE 恒等式**（带权重的 SFT 更新）、复用 rollout（TRPO/PPO）、PPO 的历史（OpenAI Gym、OpenAI Five）、概念层面很简单（Spinning Up 伪代码）、**37 个实现细节**博客、语言模型的 PPO 理想化（token 动作 + 序列级稠密奖励）、AlpacaFarm 实现逐层剖析（外层循环没问题、损失计算标准、KL 裁剪破坏 KL 意义、$\gamma=\lambda=1$ 让 GAE 退化为赌博机）、期待看到的训练曲线、为什么还需要新算法（PPO 实现复杂/价值模型吃内存、DPO 数据不成对/离线） |
| [16 · GRPO：去掉价值函数的组内相对策略优化](03-grpo.md) | GRPO 的动机（研究社区对 PPO 的恐惧）、**组内 z-score 优势**（直觉：与同组另外 G 条 rollout 比较）、**GRPO 完整目标函数**（clip 项 + KL 项，DeepSeekMath 第 (3) 式）、在线情形退化（clip 消失，只剩优势 − KL）、极简实现四步骤与 nano-aha-moment 代码（$10^{-4}$ 稳定性因子、KL 用 $e^x-1-x$ 估计、stop gradient）、GRPO 优于 RFT、**REINFORCE with baseline 定理**（状态相关 baseline 合法）、**GRPO 的基线不合法**（除以标准差破坏无偏性）、两个偏差来源（标准差放大过易/过难问题、长度归一化带来长度偏差）、Dr. GRPO 的修正（接近 REINFORCE with leave-one-out） |
| [16 · DeepSeek R1 与 R1-Zero：极简 RLVR 配方](04-deepseek-r1.md) | R1 的社会现象（性能超 o1、开放简单配方、终结 MCTS/PRM 猜测）、起点 DeepSeekMath 与**放弃过程监督**（outcome vs process supervision）、**R1-Zero 受控设定**（DeepSeek-V3 + GRPO + 准确率/格式奖励，只比 o1 略差）、有趣现象（长 CoT、aha 时刻）与**它们是否被夸大**（GRPO 长度偏差的副作用、基础模型里本就有 aha）、R1 生产化流水线（DeepSeek-V3 → 推理 SFT → RL → SFT/RLHF）、SFT 初始化（少量长 CoT 数据、读技术报告的弦外之音、1k 样本就够）、语言一致性奖励、600k 推理数据 + 200k 非推理数据、R1 的效果、800k CoT 蒸馏 Qwen 2.5、**RL 作为监督来源 vs 模仿**、不成功尝试章节（PRM、MCTS） |
| [16 · Kimi K1.5：DPO 式目标与长 CoT](05-kimi-k15.md) | 为什么要研究（与 R1 同期、用 RL 击败 o1、细节互补）、长 CoT 策略三步（数据/ SFT / RL）、数据整理（排除选择题判断题的假阳性、只留 best-of-8 失败题）、SFT 就是"prompt engineering"（规划/评估/反思/探索）、**Kimi 的 RL 目标**（KL 正则化优化问题、非参数假设闭式解 $\pi^{*}\propto\pi_{\mathrm{ref}}e^{r/\tau}$、平方损失代理、经验均值近似 $\tau\log Z$、baselined policy gradient w/ regularization、去掉价值网络的理由）、**长度控制**（组内长度奖励公式 $\lambda=0.5-\frac{\mathrm{len}-\min}{\max-\min}$、正确压短/错误只罚超中点、后期才启用）、课程表（难度标签、按 $1-\text{success\_rate}$ 采样）、奖励构造（代码生成测试用例、数学用 800k 样本训 CoT 奖励模型做答案等价）、**RL 基础设施**（on-policy 慢推理、框架切换、长 CoT 批次不均、黎曼猜想例子、off-policy 的诱惑）、缩放结果（Omni-MATH 不靠加长也变强）、**RL vs 专家迭代**消融 |
| [16 · Qwen 3 与智能体 RL](06-qwen3-and-agentic-rl.md) | Qwen 3 总体图景（base → SFT → 推理 RL → RLHF → 蒸馏）、**经过验证的 RLVR 剧本**（难度过滤、去污染、手动过滤 CoT、只用 3995 个例子）、思考模式融合（思考/非思考同处一模型、特殊字符串早停）、**测试时缩放优雅退化**、各阶段贡献构成（一般任务显著提升、数学/STEM 略降、3.5 放弃 hybrid）、**Qwen 3 Coder Next 智能体 RL**（mid-training 数据：6000 亿 token 仓库级数据、Common Crawl 文本+代码、合成数据与智能体轨迹）、四个专家模型蒸馏回单一模型、自动构造 80 万 SWE-bench 式环境、**git 历史作弊与奖励可验证性**（"RLVR 只与你的奖励一样稳健"、Lean 编译器也不对抗稳健）、SWE-bench 70.6%（3B 参数）、课堂问答（thinking mode 是 prompt 切换、mid-training 的作用、专家蒸馏的数据混合、长 CoT 与长上下文扩展、RL 串行 vs 并行）、本讲回顾 |

## 本讲要点

- **RLHF 无法干净缩放，RLVR 是出路**：RLHF 的奖励模型只是人类偏好的近似，往同一个奖励模型里无限投入计算会过拟合它（标注瓶颈）；而在围棋、数学、编码这类**可验证**领域，我们优化的恰恰是想要的东西，可以无限投入计算——"搜索问题"与"学习问题"之分是理解 RLVR 的钥匙；
- **策略梯度是一切的核心**：$\nabla_\theta\mathbb{E}_{p_\theta}[R] = \mathbb{E}_{p_\theta}[R\,\nabla_\theta\log p_\theta]$ 相当于带正负权重的 SFT 更新；PPO 只是在此之上加了"裁剪比率到 $[1-\varepsilon,1+\varepsilon]$"以安全地复用 rollout；
- **PPO 概念简单、实现狰狞**：价值模型（吃内存、要调参）、经验缓冲区、逐 token KL 项让语言模型上的 PPO 极其敏感（"37 个实现细节"）；常见的怪癖包括把 KL 裁剪到 0（破坏 KL 意义）和设 $\gamma=\lambda=1$（把 GAE 退化成赌博机）；
- **GRPO 用组内 z-score 换掉价值函数**：优势 $\hat{A}_i = \frac{r_i-\mathrm{mean}(r)}{\mathrm{std}(r)}$，在线情形下目标退化为"优势 − KL 惩罚"，半个幻灯片就能实现（std 加 $10^{-4}$、KL 用 $e^x-1-x$）；它明显优于 RFT（拒绝微调）；
- **GRPO 不是"合法"的策略梯度**：除以标准差不是合法的状态相关 baseline（REINFORCE with baseline 只允许减法），它会**放大过易/过难的问题**；按序列长度平均则引入**长度偏差**（长序列梯度总量更小）——这正是 R1-Zero 长 CoT 现象的可能来源；Dr. GRPO 去掉这两处后接近 REINFORCE with leave-one-out；
- **R1-Zero 证明 RLVR 可以极简**：基础模型 + GRPO + 准确率/格式奖励，只比 o1 略差；"aha 时刻"在基础模型里就有、"CoT 变长"可能是目标偏差的副作用——宣传的现象未必等于模型的顿悟；
- **R1 展示生产化配方**：推理 SFT → GRPO RL（+ 语言一致性奖励）→ SFT/RLHF；**过程监督、PRM、MCTS 都不是必需的**（有专门的"不成功尝试"章节）；800k 条蒸馏 CoT 就能让不会推理的模型学会推理——RL 是自我生成监督的途径，但模仿有时也够用；
- **Kimi K1.5 用 DPO 式推导得到平方损失目标**：KL 正则化优化经非参数假设给出 $\pi^{*}\propto\pi_{\mathrm{ref}}e^{r/\tau}$，平方损失代理本质上就是"带 reward 均值 baseline 的策略梯度 + $L_2$ 正则"；数据上排除选择题/判断题（假阳性）、只留 best-of-8 失败题；长度奖励压缩 CoT（正确压短、错误只罚超中点）；
- **RL 基础设施是隐藏的大坑**：on-policy 意味着慢推理、训练/推理框架切换、一个超长 CoT 拖垮整个批次（黎曼猜想例子）；复用 rollout 的诱惑会把你推向 off-policy 的不稳定；
- **RL 一致地胜过专家迭代**（Kimi 的消融），**RLVR 的稳健性只与奖励的稳健性相当**：模型会 hack git 历史、绕过"不许用 git log"的约束，连 Lean 编译器都不是对抗稳健的；
- **Qwen 3 与 Coder Next**：只用约 4000 个例子做 GRPO 也能走很远；思考模式融合（思考/非思考同处一模型 + 特殊字符串早停）让测试时缩放优雅退化；智能体 RL 的完整图景是 mid-training（6000 亿 token 仓库级数据）→ 四个专家 → 蒸馏回单模型 → 80 万个 SWE-bench 式环境做 RL（SWE-bench 70.6%，仅 3B 参数）——**数据始终是真正重要的东西**。

## 课程导航

- [上一讲：15 Mid/Post-Training](../15/)
- [下一讲：17](../17/)
