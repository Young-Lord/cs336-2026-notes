---
title: "16 · GRPO：去掉价值函数的组内相对策略优化"
lecture: 16
---

# GRPO：去掉价值函数的组内相对策略优化

## 为什么是 GRPO：研究社区对 PPO 的"恐惧"

与 DPO 类似，研究社区对"不用 PPO"有极大的渴望。**DPO 和 GRPO 能够被广泛采用这件事本身，就说明了 PPO 在很多情况下有多痛苦。** 上一节讲过，DPO 的替代路径是从偏好数据出发、直接优化策略；而 GRPO（Group Relative Policy Optimization，组内相对策略优化）则是**可验证任务（verifiable tasks）上更简单的 RL 方式**，它已经在很大程度上接管了开源社区的 RLVR。它和 PPO 在精神上非常接近，只是剥离掉了 PPO 里最复杂的几个部分，从而得到一个简单得多的算法。

## GRPO 的核心想法：用组内 z-score 代替价值函数

GRPO 出自 DeepSeek 的 **DeepSeekMath** 论文。它的起点是 PPO："PPO 是个好主意，但我只想改几件事。"改的是什么呢？——**PPO 里最复杂、最烦人的部分：价值函数（value function）**。

价值函数是那个被当作 baseline 减掉、用来降低梯度更新方差的东西。它本身是一个完整的神经网络，会让训练不稳定，我们根本不想要它。但**去掉价值函数之后，仍然需要优势（advantage）**——又不想用 vanilla REINFORCE，因为它的方差实在太高。怎么办？答案：**把优势算成"组内 z-score（z-score within a group）"**。

直觉是这样的。通常你会得到一个奖励，如果有价值函数，就把它和预测值比较："我的神经网络说我应该得 5 分，结果我得了 6 分，所以这是个好 rollout。"GRPO 换了一种方式：**你得到一个得分 5 的 rollout，就再采样 10 个别的 rollout，然后问"我相对于这 10 个 rollout 表现如何"**。如果你比组内均值好，优势就高——这在"能从同一个 prompt 采样多次"的设定里是一种非常自然的优势计算方式。

具体地，对每个问题 $q$，从旧策略 $\pi_{\theta_{\mathrm{old}}}$ 采样一组输出 $\{o_1, \dots, o_G\}$，用奖励函数打分得到 $G$ 个奖励 $r = \{r_1, \dots, r_G\}$，然后：

$$
\hat{A}_{i,t} = \tilde{r}_i = \frac{r_i - \mathrm{mean}(r)}{\mathrm{std}(r)}.
$$

GRPO 优化的目标函数（DeepSeekMath 论文中的定义）是：

$$
\mathcal{J}_{\mathrm{GRPO}}(\theta) = \mathbb{E}\left[ q \sim P(Q),\ \{o_i\}_{i=1}^{G} \sim \pi_{\theta_{\mathrm{old}}}(O \mid q) \right]
\frac{1}{G}\sum_{i=1}^{G} \frac{1}{\lvert o_i \rvert}\sum_{t=1}^{\lvert o_i \rvert}
\min\left[
\frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t} \mid q, o_{i,<t})}\hat{A}_{i,t},\;
\operatorname{clip}\!\left(\frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t} \mid q, o_{i,<t})},\ 1-\varepsilon,\ 1+\varepsilon\right)\hat{A}_{i,t}
\right]
- \beta\, \mathbb{D}_{\mathrm{KL}}\left[ \pi_\theta \,\|\, \pi_{\mathrm{ref}} \right],
$$

其中 $\varepsilon$、$\beta$ 是超参数。这个目标完全就是 PPO 的更新：min 掉裁剪后的优势，再加上一个 KL 项让自己别离参考策略太远；KL 用一种特殊方式计算，细节不是特别重要（DeepSeekMath 用的是 $e^{x} - 1 - x$ 形式的无偏估计，其中 $x = \log\frac{\pi_{\mathrm{ref}}}{\pi_\theta}$，保证恒为正）。关键的差别只在于：**优势 $A_i$ 是在每组输出内部算 z-score**——减组内均值、除以组内标准差。

![GRPO 的定义：去掉价值函数，用组内 z-score 作为优势（右侧为 PPO 作对照）](/lectures/16/slide-18.png)

这里有一个重要的区分：**在线（online）情形**。在很多情况下你可能就是在线的（rollout 完立刻更新）。这时裁剪操作基本消失——因为 $\pi_{\theta_{\mathrm{old}}}$ 和 $\pi_\theta$ 相同，比率恒为 1，clip 永远不会起作用，目标退化成"**优势减去 KL 惩罚**"：

$$
\mathcal{J} \approx \hat{A}_i - \beta\, \mathbb{D}_{\mathrm{KL}}[\pi_\theta \,\|\, \pi_{\mathrm{ref}}].
$$

所以在在线 rollout 情形下，GRPO 是一个**非常简单、意义清晰**的 RL 目标。这个目标值得好好理解——它是后面所有令人兴奋的数学能力的基石，你们作业里也能用它复现这些结果。

## 极简实现：半个幻灯片就够

GRPO 非常简单（多亏了没有价值函数）。你可以——而且人们真的会——写出很小的 GRPO 实现。步骤只有四步：

1. 为每个 rollout 计算奖励；
2. 按组做均值/方差归一化；
3. 计算 KL 项；
4. 在损失上做梯度更新。

作业里你们会亲手实现它。要用 autodiff 做这件事，你需要在一两处做 **stop gradient**（把归一化用的统计量当作常数，不让梯度穿过它们），但整体上并不复杂。Tatsu 展示了一个参考实现——McGill 的 **nano-aha-moment** 项目，它有一个很漂亮的玩具版实现，组索引的构造、标准差的计算等全部塞进半页幻灯片。它和论文字面写法唯一的区别是：**在标准差计算里加了一个 $10^{-4}$ 的稳定性因子**，防止在只有一个样本、或者样本奖励完全相同（比如都得了 0 分）时标准差 blow up——这在"解不出数学题就得数值零奖励"的领域里是真实会发生的事。

优势计算的代码大致是：

```python
rewards = np.array(rewards)                       # 组内 G 个 rollout 的奖励
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
```

损失部分（策略梯度项 + KL 惩罚）大致是：

```python
ref_logratio = ref_logps - logps                 # log(pi_ref / pi_theta)
kl_penalty = torch.exp(ref_logratio) - 1 - ref_logratio   # 逐 token KL 估计
policy_loss = -logps * advantages                # 加权"负 SFT"梯度
loss = (policy_loss + KL_COEFFICIENT * kl_penalty).sum() / total_response_len
```

这个实现之所以值得强调，是因为它说明了 GRPO 有多么直截了当，也解释了为什么几乎所有开源工作都建立在这个算法上：**容易实现、容易理解**，而且如我们即将看到的，它在复现闭源实验室（那些可能用 PPO 的地方）的结果上交出了相当有说服力的答卷。

## 效果：GRPO 对 RFT 的胜出

GRPO 在 DeepSeekMath 原始论文里的结果如何？如果你对数学 AI 感兴趣，这篇论文值得一读，它有一组漂亮又引人入胜的实验。要点是：**GRPO 明显好于 RFT（rejection fine-tuning，拒绝微调）**——RFT 就是把你模型生成的正确答案拿来训练、丢掉其他一切，这也是你们作业里要实现的 baseline；同时论文似乎表明**过程监督（process supervision，不只给最终答案打分，还给中间步骤打分）带来一些增益**。Tatsu 说后面会再回到过程监督这个话题，因为它是这类 RL 问题里一个重大的设计决策。现在先记住结论：GRPO 有效，蓝线和黄线都压在别的曲线上面。

![GRPO 在 DeepSeekMath 中的结果：优于 RFT，过程监督带来部分增益](/lectures/16/slide-21.png)

## 认真审视 GRPO 的目标：这是"合法"的优势吗？

现在退一步，认真想想 GRPO 目标里到底发生了什么——我们真的在取策略梯度吗？GRPO 与 PPO 的关键区别就是优势：PPO 里价值函数喂给优势，GRPO 把它换成了 z-score 的家伙 $\hat{A}_i = \frac{r_i - \mathrm{mean}(r)}{\mathrm{std}(r)}$。**这是一个好的优势吗？**

上过 RL 课的人大概已经知道答案。翻开 Sutton and Barto 那本经典教材，里面有 **REINFORCE with baseline** 算法，它告诉你策略梯度（也就是本讲第一个方程）**允许你从奖励里减去任何"状态相关基线"（state-dependent baseline）**——在赌博机世界里，我们的状态就是 prompt，所以任何"prompt 相关基线"都可以减。只要你这么做，你仍然沿同一个方向下降；而 $b$ 的不同选择只会让这个梯度下降过程方差更低或更高。

![REINFORCE with baseline：可以减去任何状态相关基线而不改变梯度的期望（Sutton and Barto）](/lectures/16/slide-22.png)

你应该已经注意到，**GRPO 做的不是这件事**。它不只是减掉一个常数，**还除以了标准差**。这就有问题了——或者说，"在某种意义上是个问题，在另一种意义上又不是"。如果你想要一个概念上干净、真正"表里如一"的算法——真的在下山奖励——**GRPO 并不满足**。因为除以标准差破坏了我们上面说的"基线契约"：我们不只减基线，还对奖励做了归一化。

另外 GRPO 还有一个前面没提的实现细节：**它几乎按 token 平均**——把整个序列的长度作为归一化因子除进去。但如果从第一性原理出发，老老实实按照策略梯度定理和基线定理推导 GRPO，**你得到的东西不一样**：既没有长度归一化，也没有标准差归一化。

GRPO 出来之后不久，就有人注意到这件事，写了一篇论文（**Dr. GRPO**，Liu et al. 2025）：如果你把这两处"额外动作"去掉，会得到非常不同、而且很可能是更好的行为。Tatsu 说这个话题到 DeepSeek R1 那部分还会再讲。但这里的核心是：**GRPO 不是这个想法的最朴素推导，它实际做的是稍有不同的东西，有优点也有缺点。**

## GRPO 的两个偏差来源

既然知道 GRPO 并不直接在下降奖励目标，它带有的这两个"修正因子"到底在干什么？

**标准差归一化**。除以标准差，等于**放大标准差小的题**。对二元奖励问题，什么时候标准差小？——**当题太容易或太难的时候**。如果题非常简单、你 100% 做对，奖励完全没有变化，你除以（几乎）零，权重被显著放大；如果题难到 100% 做错，同样是零变化，也会被显著放大。所以**标准差项同时上调了"太简单"和"太难"两端的问题**。这显然不一定是我们要的——我们希望模型在"它解决得了的范围"内学习。

**长度归一化**。这个稍微复杂一点。GRPO 按序列长度平均，意味着每个序列对梯度的总贡献被 $\frac{1}{\lvert o_i \rvert}$ 缩放。在优势对整条序列恒定的情况下，**长序列获得的梯度总量更小**——这就产生了一个"长度偏差"：模型学到"想获得更大的梯度，就把回答写短"，从而人为地压低 CoT 长度。后续分析指出，R1-Zero 训练中"CoT 越来越长"的现象，很可能就是这种目标偏差的自然结果，而非模型"学会思考更久"。

![GRPO 的长度偏差：标准差项放大过易/过难的问题，长度归一化项带来长度偏差](/lectures/16/slide-24.png)

**修复是什么？** 去掉这两处归一化——即用"逐问题"（而不是逐 token）的策略梯度、并用 leave-one-out 式基线代替组均值除标准差，就得到一个无偏的梯度版本。这个版本与 REINFORCE with leave-one-out 非常接近，也是 Dr. GRPO 论文提出的修正。

## 小结

GRPO 是这一讲的第一个"新"算法，也是后面所有案例（R1、Kimi、Qwen）的共同地基：**没有价值函数、组内 z-score 优势、极简实现、效果很好**。但它并非无懈可击——标准差归一化放大过易/过难问题、长度归一化引入长度偏差，这两点在后来的分析与改进中都被拿掉了。带着这些理解，我们进入第二部分：看这些算法如何真实地出现在开源模型的技术报告里。

<!-- lecture-nav -->

**← 上一节**：[PPO 复习：从策略梯度到实现细节]（02-ppo-revisited.md）　**→ 下一节**：[DeepSeek R1 与 R1-Zero：极简 RLVR 配方](04-deepseek-r1.md)
