---
title: "16 · PPO 复习：从策略梯度到实现细节"
lecture: 16
---

# PPO 复习：从策略梯度到实现细节

## 一切的起点：策略梯度与 REINFORCE 技巧

要讨论语言模型的强化学习，绕不开 **PPO**（Proximal Policy Optimization，近端策略优化）。即使上一讲已经讲过一遍，PPO 足够令人困惑，值得再讲一遍。而所有内容里最重要、最需要记住的，是**策略梯度（policy gradients）**，尤其是其中的 **REINFORCE 技巧**：

$$
\nabla_\theta \mathbb{E}_{z\sim p_\theta}\left[ R(z) \right]
= \mathbb{E}_{z\sim p_\theta}\left[ R(z)\, \nabla_\theta \log p_\theta(z) \right].
$$

推导与上一讲相同：把梯度推进期望里，利用 $\nabla_\theta p_\theta(z) = p_\theta(z)\,\nabla_\theta \log p_\theta(z)$ 即可。我们**一直在做的事情，就是对奖励做梯度下降**，具体方式相当于**带权重的 SFT 更新**——每个样本的权重是它的奖励，权重可正可负。$R$ 的具体定义我们后面再说，但这个方程很重要，**因为它会在本讲反复出现——从某种意义上说，它是一切推导的核心**。

在此基础上，回忆上一讲的直觉：我们想一次迈出多步。**朴素的策略梯度要求每次取梯度步都要从当前策略采样**——能不能复用已有的 rollout？可以，方法就是 TRPO、PPO 这一类。Tatsu 假设不是每个同学都上过强化学习课程，所以他会把 PPO 当作"陌生算法"再介绍一遍。

![PPO 在理论上的三步演进：策略梯度（方差太高）→ TRPO（在当前策略附近线性化问题）→ PPO（把比率裁剪到某个 $\varepsilon$ 内）](/lectures/16/slide-05.png)

Tatsu 把 PPO 的理论谱系概括成三步：**策略梯度（variances are too high）→ TRPO（在当前策略附近线性化问题）→ PPO（把比率裁剪到某个 $\varepsilon$ 内）**。这条演进路径上一讲已经推导过，今天只需在心里装着它，然后去看 PPO 在真实工程里的样子。

## PPO 是什么：RL 的主力算法

如果你不熟悉 PPO，它是强化学习界真正的"主力"（workhorse），被用在大量非常困难的 RL 设定里。OpenAI 在早期极度"RL 化"的年代做了不少这类工作：比如 OpenAI Gym 里让人形机器人走路；而更戏剧性的演示是 **OpenAI Five**——用 PPO 训练出的、在 Dota 2 里打赢人类高手的队伍。这类问题的特点是在高维动作与状态空间上做深度 RL，比通常的 RL 复杂得多。

![PPO 的历史：2017 年的 PPO 公告博客与 2019 年用 PPO 训练的 OpenAI Five](/lectures/16/slide-06.png)

## 概念层面：PPO 其实很简单

概念上，PPO 非常简单。很多人一直在试图"干掉 PPO"，这本身就是一个动机；但如果你去看 OpenAI Spinning Up 文档里的 PPO 伪代码，会觉得"这没什么可怕的，一次就能实现"：

1. 采样一些轨迹（trajectories）；
2. 用任意优势估计方法（advantage estimation）计算一个优势；
3. 用那种"稍微有点奇怪但可以接受"的方式**裁剪优势**，在裁剪后的优势下更新策略；
4. 再拟合一个**价值函数（value function）**。

![PPO 的概念层面：目标函数基本就是全部（Spinning Up 伪代码）](/lectures/16/slide-07.png)

正因如此，大家的反应往往是"PPO 完全没问题"。然后你就会看到这样的博客文章——**《PPO 的 37 个实现细节》**。这类文章应该让你心生敬畏：**如果某个算法有一篇讲"37 个实现细节"的博客，说明它对你的实现决策极其敏感。** 你有各种各样的库、各种各样的实现，它们给出的数字完全不同；而且事实证明**很多人把它实现错了**——有论文指出，某些人用的 PPO baseline 根本不是 baseline，从根本上改变了优化问题。

![如果你看到讲 PPO 实现细节的博客，就该知道情况不妙](/lectures/16/slide-08.png)

## 语言模型中的 PPO：理想化图像

语言模型设定下的 PPO，与通用 RL 设定相当接近：**动作（actions）作用在 token 上，而在序列最后有一个大的稠密奖励（dense reward），作用于完整序列**。

![语言模型中的 PPO 理想化：动作是 token，最终奖励作用于整个序列（Zheng et al. 2023）](/lectures/16/slide-09.png)

但实际实现远没有这么干净。这张图（来自 AlpacaFarm 的实现）是 PPO 在语言模型上的很好写照：中间那一大块是**优势估计（advantage estimation）**；有**经验缓冲区（experience buffer）**，因为要保留一部分旧数据；要训练**价值模型**，而价值模型又反过来用于优势估计的计算——注意图中绿色方框出现了两次；更重要的是，目标里的 **KL 项是逐 token 计算的**，所以这**并不是单纯的赌博机问题（bandit problem），而是一个多步 RL 问题**。所有这些都让 PPO 的实现变得又难又复杂。

## 走进一个真实实现：AlpacaFarm 的 PPO

为了看清楚"混乱"到底发生在哪，Tatsu 带我们看了一个真实且相当稳健的参考实现——**AlpacaFarm 的 PPO**（他们的学生之前用它做 RLHF 项目，花了很多时间才让它跑通）。逐层往下看：

**外层循环没有问题**：通过 rollout 取数据，做若干步 PPO 更新，计算损失、裁剪梯度、前进几步。非常合理。

**内层损失计算也基本标准**：按照 PPO 更新计算优势、计算裁剪比率（clipping ratios）、据此更新模型。下面是从 AlpacaFarm 的 `ppo_trainer.py` 里摘出的核心（`cliprange = 0.2`）：

```python
def compute_ppo_loss(...):
    log_probs = ...            # 当前策略下的 log 概率
    old_log_probs = ...        # 采样时策略的 log 概率
    ratio = torch.exp(log_probs - old_log_probs)
    # 目标：min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1.0 - cliprange, 1.0 + cliprange
    )
    pg_loss = torch.max(pg_losses1, pg_losses2).mean()
    ...
```

但再往下，就会碰到比较"脏"的部分：

**奖励塑形（reward shaping）**。高层想法是：加一个逐 token 的 KL 惩罚，最后的 token 给完整奖励。但在实践中，**只有在新策略的对数概率小于参考策略（new policy logp < reference logp）时，才把 KL 裁剪掉**。为什么？如果你把模型"训爆"了，这个操作能防止 KL 发散。可问题在于：**把 KL 裁剪到 0，完全破坏了 KL 散度的意义**——KL 是正负值都会被求和的对象，一旦移除裁剪，损失立刻 blow up。诸如此类的细节层出不穷。

**广义优势估计（GAE）**。原始 PPO 论文里有广义优势估计（Generalized Advantage Estimation）：用折扣奖励 $\gamma$ 与价值函数在每个 token 生成步上估计奖励：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t),
\qquad
\hat{A}_t = \sum_{\ell=0}^{T-t-1} (\gamma\lambda)^{\ell}\, \delta_{t+\ell}.
$$

但实际中**人们常常直接设 $\gamma = \lambda = 1$**——这是一个退化设定，把 GAE 直接变回赌博机问题（用"奖励减价值"替代优势）。换句话说，**你亲手丢掉了 PPO 带来的大部分结构**。Tatsu 强调，这不是他学生的实现才有的怪癖，而是非常普遍的做法。

## 你期待看到什么：PPO 的训练曲线

如果一切正常（经过大量工程调试），PPO 的训练曲线是这样的：

- **总体奖励不断上升**；
- 如果是 RLHF，**奖励模型的分数上升**；
- **KL 奖励为负**（模型在靠近或偏离参考模型）。

这是一个赌博机设定，所以你会期待合理、单调的训练曲线。你们做作业跑 RL 时，应该能看到类似的现象。

![PPO 中你期待看到的东西：整体奖励上升（含奖励模型），KL 奖励为负](/lectures/16/slide-16.png)

## 为什么还需要新的 RL 算法？

既然 PPO 能用，为什么还要新算法？

**为什么不用 PPO？**

- 实践中**实现非常复杂**；
- 需要**价值模型**——它很吃内存（memory hungry），训练时还要额外的调参。

**为什么不用 DPO？**

- **数据天然不是成对的**——DPO 是为成对反馈（Bradley–Terry 比较）设计的，而数学题的数据并不是天然成对的；
- **离线（offline）**——不过这个区分被严重夸大了，因为通过反复迭代 DPO，完全可以让它变成在线的。

所以 PPO 是那个更通用的"锤子"，什么都能敲；DPO 是特定问题的特定解法。如果你要解数学题，用 DPO 就是"用错了锤子"。下一节的主角 **GRPO**，正是为可验证任务准备的、更简单的替代方案。

<!-- lecture-nav -->

**← 上一节**：[引言：从 RLHF 到 RLVR]（01-introduction-rlvr.md）　**→ 下一节**：[GRPO：去掉价值函数的组内相对策略优化](03-grpo.md)
