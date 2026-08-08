---
title: "16 · Kimi K1.5：DPO 式目标与长 CoT"
lecture: 16
---

# Kimi K1.5：DPO 式目标与长 CoT

## 为什么要研究 Kimi K1.5

**Kimi K1.5** 与 DeepSeek R1 几乎同时发布，也有两个值得研究的理由：

- 它也**只用 RL 就击败了 o1**；
- 它在**数据、目标函数、长度控制**上提供了与 R1 **互补的细节**——如果说 R1 展示了"极简配方"，Kimi 则展示了"精细打磨"的那一面。

![Kimi K1.5：与 R1 同时发布，也用 RL 击败 o1](/lectures/16/slide-39.png)

它的整体策略是一条**长 CoT 推理**路线，三个关键步骤是：

1. **数据构造**（难度过滤）；
2. **SFT**（为长 CoT 做准备）；
3. **RL**（用他们自己的策略梯度损失）。

![Kimi 的长 CoT 推理策略：数据构造（难度过滤）→ SFT（长 CoT）→ RL（自己的策略梯度损失）](/lectures/16/slide-40.png)

## 数据整理与 SFT：怎么保证"可验证"

### 数据整理

Kimi 在数学类设定里做了标准的整理：平衡不同主题的题目分布。但更重要的是两条反直觉的排除规则：

- **排除选择题与判断题**——它们的验证容易产生**假阳性（false positives）**：模型可能靠猜答对，而验证器无法察觉；
- **只保留"best-of-8 都答不对"的问题**——即让模型自己采样 8 次都失败的问题。这保证了数据的"可学性"与难度：如果模型已经会做，RL 就没有增益；如果模型完全做不了，RL 也学不到。

### SFT：几乎没有描述的"prompt engineering"

Kimi 对长 CoT SFT 的描述少得可怜——只用"**prompt engineering**"一笔带过（Tatsu 怀疑那本质上是蒸馏）。做法是构造一个**小而高质量的长 CoT 热身数据集**，其中封装了人类推理的关键认知过程：**规划（planning）、评估（evaluation）、反思（reflection）、探索（exploration）**。对这个热身集做一次轻量 SFT，模型就"预热"出了这些推理策略。

## Kimi 的 RL：DPO 式推导 + 平方损失代理

Kimi 的 RL 部分最有意思。他们有一个**基于参考模型的奖励模型（reference-based reward model）**，所以优化问题是：

$$
\max_{\theta}\ \mathbb{E}_{(x,y^*)\sim\mathcal{D}}\ \mathbb{E}_{(y,z)\sim\pi_\theta}\left[ r(x,y,y^*) \right]
- \tau\, \mathbb{D}_{\mathrm{KL}}\left[ \pi_\theta(\cdot \mid x) \,\|\, \pi_{\theta_i}(\cdot \mid x) \right],
$$

其中 $\tau > 0$ 控制正则化的强度，$\pi_{\theta_i}$ 是当前迭代的参考模型。这和我们上一讲在 DPO 推导里见过的是同一个 KL 正则化目标——**Kimi 的 RL 算法正是受 DPO 式推导启发的**。

**第一步：非参数假设 + 解出 $r$。** 把 $\pi_\theta$ 当作任意策略（非参数假设），这个问题有闭式解：

$$
\pi^{*}(y,z \mid x) = \frac{1}{Z(x)}\, \pi_{\theta_i}(y,z \mid x)\, \exp\!\left( \frac{r(x,y,y^*)}{\tau} \right),
$$

其中 $Z(x) = \sum_{y',z'}\pi_{\theta_i}(y',z'\mid x)\exp(r(x,y',y^*)/\tau)$ 是归一化因子。两边取对数，就得到对任何 $(y,z)$ 都成立、且能利用离策略数据的约束：

$$
r(x,y,y^*) - \tau \log Z(x) = \tau \log \frac{\pi^{*}(y,z \mid x)}{\pi_{\theta_i}(y,z \mid x)}.
$$

**第二步：平方损失作代理。** Kimi 不把 $\pi^{*}$ 换成参数化策略去构造成对目标（那是 DPO 的路子），而是直接用平方损失，让策略的对数比率去拟合"奖励减去 $\tau\log Z$"：

$$
L(\theta) = \mathbb{E}_{(x,y^*)\sim\mathcal{D}}\ \mathbb{E}_{(y,z)\sim\pi_{\theta_i}}\left[
\left( r(x,y,y^*) - \tau \log Z(x) - \tau \log \frac{\pi_\theta(y,z \mid x)}{\pi_{\theta_i}(y,z \mid x)} \right)^2
\right].
$$

**第三步：用经验均值近似 $\tau\log Z$。** 对每个问题 $x$ 从 $\pi_{\theta_i}$ 采样 $k$ 条响应，用样本均值近似：

$$
\tau \log Z(x) \approx \bar{r} = \frac{1}{k}\sum_{j=1}^{k} r(x,y_j,y^*).
$$

这之所以合理，是因为当 $\tau \to \infty$ 时 $\tau\log Z$ 趋近于 $\pi_{\theta_i}$ 下的期望奖励。最终梯度为：

$$
\nabla_\theta L(\theta) = \frac{1}{k}\sum_{j=1}^{k}\left[
\nabla_\theta \log \pi_\theta(y_j,z_j\mid x)\, \big( r(x,y_j,y^*) - \bar{r} \big)
- \frac{\tau}{2}\, \nabla_\theta \left( \log \frac{\pi_\theta(y_j,z_j\mid x)}{\pi_{\theta_i}(y_j,z_j\mid x)} \right)^2
\right].
$$

![Kimi 的 RL 目标：参考模型正则化的策略优化，DPO 式推导 + 平方损失代理](/lectures/16/slide-42.png)

对熟悉策略梯度的人来说，这个梯度**本质上就是"用样本奖励均值做 baseline 的策略梯度"**，区别只在于：响应是从 $\pi_{\theta_i}$ 采样的（离策略），并且加了一个 $L_2$ 正则项。所以它可以说是"通常的在线正则化策略梯度"向离策略情形的自然推广——**baselined policy gradient w/ regularization**。另外，Kimi 也完全去掉了价值网络，并给出了一个很有意思的理由：如果给每个中间步骤都分配价值（credit assignment），那么探索那些"最终能纠正回来"的错误分支反而会被惩罚，这不利于模型学会长 CoT 里的试错（trial and error）。

## 长度控制：进一步压缩 CoT

Kimi 的目标本身没有 GRPO 那种长度偏差问题，但他们**仍然想把 CoT 压缩得更短**——模型在 RL 训练中会出现"过度思考"（overthinking），token 数疯长。于是他们为每个 batch 设计了一个**长度奖励（length reward）**。

设一个问题 $x$ 采样的 $k$ 条响应长度为 $\mathrm{len}(i)$，组内最长为 $\mathrm{max\_len}$、最短为 $\mathrm{min\_len}$（若相等则长度奖励全为 0），定义：

$$
\lambda_i = 0.5 - \frac{\mathrm{len}(i) - \mathrm{min\_len}}{\mathrm{max\_len} - \mathrm{min\_len}},
\qquad
\mathrm{len\_reward}(i) = \begin{cases}
\lambda_i, & r(x,y_i,y^*) = 1\ \text{（正确）}\\
\min(0,\ \lambda_i), & r(x,y_i,y^*) = 0\ \text{（错误）}
\end{cases}
$$

怎么理解这条规则（幻灯片上的解读）：

- $\lambda$ 落在 $[0.5, -0.5]$ 之间，**组内越长的序列 $\lambda$ 越负**；
- **正确的回答被激励写短**：比组内均值短的拿正奖励，比均值长的挨罚；
- **错误的回答**只有 $\min(0,\lambda)$——只惩罚那些比"组内最长与最短的中点"还长的，比中点短的不给正奖励。

这条长度奖励以某个权重加进总奖励里。Kimi 还特别提到：**长度惩罚在训练初期会拖慢收敛**，所以他们先跑一段不带长度惩罚的标准策略优化，再在训练后期开启并保持恒定的长度惩罚——这解释了为什么"只在后期启用"。

![Kimi 的长度控制：组内长度奖励把正确回答压短、惩罚过长的错误回答](/lectures/16/slide-43.png)

## 更多细节：课程表与奖励构造

**课程表（curriculum）**。Kimi 给数据打了难度标签，从易到难上课；同时按 $1-\text{success\_rate}$ 的概率采样问题——**越答不上的问题采样越多**，避免反复做已经会做的题。这两个信号（数据自带的难度标签、训练中跟踪到的每题成功率）都不需要额外标注。

**奖励构造**。

- **代码**：许多网上的编码题没有测试用例，Kimi 的做法是——对有标准答案的问题**自动生成新的测试用例**来当奖励信号；
- **数学**：用 **800k 个样本训练一个 CoT 奖励模型**，专门做"答案等价性"（answer equivalence）检查——因为同一道题可以有多种等价答案，需要一个模型来判断"这个答案算不算对"。

## RL 基础设施：RL 为什么难做高效

系统的利用效率（utilization）是 RL 里极其重要的方面。为什么 RL 这么难做高效？

- **On-policy 意味着 rollout**：rollout 是（很慢的）**推理**，而推理在系统上远比训练难伺候；
- **训练与 rollout 之间的切换**：通常需要**不同的框架**（训练框架 vs 推理框架），来回切非常昂贵；
- **长 CoT 让批次严重不均**：设想一批 rollout 里有一道极难的题（比如黎曼猜想），模型在那儿"吭哧吭哧"算很久。如果做朴素的批处理推理，**其他所有 rollout 都得等这道题完成才能进入下一阶段**——一个极长的 rollout 就能拖垮整个批次。要不要截断？要不要把它挪到别的机器？这些都是你必须做的决策。

还有那个痛苦的两难：**on-policy 在数学与训练动态上非常好**（作业里你会体会到 GRPO 简单 on-policy 形态的优雅），但系统利用率低。你于是变得"贪婪"：想复用 rollout、想重叠推理与计算、想做各种聪明的事——**然后你就引入了 off-policy 问题，导致训练不稳定**，等等。

所以如今大多数开源技术报告都会有一节讲他们的 RL 基础设施：既要有训练部分（图里的蓝色框），也要有推理部分（右侧的绿色框）；**权重要在训练与推理之间搬来搬去**，两者必须密切协调，有时甚至共享同一批机器（推理跑的时候训练侧可能正闲着）。

![Kimi 的 RL 训练设置：训练（蓝）与推理（绿）两部分紧密协调，权重需要在两者间搬运](/lectures/16/slide-46.png)

## 缩放结果：不只是"token 越长越好"

我们已经知道 Kimi 大致匹配甚至超过了 o1，但还有别的有趣结果：

- **随着 RL 推进，模型想得更长、性能上升**；
- 但并不是在无限地堆 token——**Omni-MATH 是个好例子**：思考时间没有显著变长，性能却持续上升。这正好说明**长度控制（length control）在起作用**，而不是简单地把预算全部花在更长的 CoT 上；
- 即便是**只用数学数据训练的小模型**，也展示了很有意思的缩放行为。

![Kimi 的缩放结果：RL 进行时思考变长、性能上升；但 Omni-MATH 说明长度控制可以带来"不加长也变强"](/lectures/16/slide-47.png)

## 消融：RL 对专家迭代

最后一个问题：**能不能不做 RL 式的负梯度、只从正样本学习？** 这就是**专家迭代（expert iteration）**——在不少过去的论文里它效果很好，在处理非常不稳定的设定时你可能更愿意用它。但 Kimi 有大规模的消融实验：**这类 RL 方法在一致性地优于专家迭代**（图里的橙色压过蓝色）。所以如果你想榨出全部性能，**RL 是躲不掉的**。

![Kimi 与专家迭代的消融对比：RL（橙色）始终优于专家迭代（蓝色）](/lectures/16/slide-48.png)

## 小结

Kimi K1.5 与 R1 互补地展示了 RLVR 的另一面：**同样的 GRPO 式地基，但目标函数换成了 DPO 式推导出的平方损失代理**（baselined policy gradient w/ regularization），并且极其重视数据（难度过滤、假阳性排除、课程表）与长度控制（组内长度奖励、后期启用）。它还提醒我们，**RL 的系统问题（推理/训练协同、批次不均）与算法问题同样重要**。下一节也是最后一节：Qwen 3 与智能体 RL。

<!-- lecture-nav -->

**← 上一节**：[DeepSeek R1 与 R1-Zero：极简 RLVR 配方]（04-deepseek-r1.md）　**→ 下一节**：[Qwen 3 与智能体 RL](06-qwen3-and-agentic-rl.md)
