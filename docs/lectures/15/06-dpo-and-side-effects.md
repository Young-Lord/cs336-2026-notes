---
title: "15 · DPO 与 RLHF 的副作用"
lecture: 15
---

# DPO 与 RLHF 的副作用

## DPO：没有眼泪的 RLHF？

上一节我们看到，PPO 又复杂又挑剔。于是 DPO（Direct Preference Optimization，直接偏好优化）的目标很明确：**把 PPO 简化**——

- **去掉奖励模型（reward model）**；
- **去掉一切 on-policy 的东西**（rollout、外层循环等）。

取而代之的是一个非常简单的直觉：**深度学习里的一切，都是朝"好东西"的方向走梯度步**。所以我们就：

- 朝"好样本"的**对数损失方向**走梯度步；
- 朝"坏样本"的方向走**负**梯度步（适当加权）。

换句话说：对一边做 SFT，对另一边做"负 SFT"。这几乎是你能想到的最天真的事情——但事实证明，**只要把这两个方向加权得当，你就能得到一个相当好的算法**。

## DPO 的完整推导

Tatsu 说推导很简单，他会在黑板上走一遍。我们从 RLHF 的目标出发：

$$
\max_{\pi}\ \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi(y\mid x)}\left[ r(x,y) \right]
- \beta\, \mathbb{D}_{\mathrm{KL}}\left[ \pi(y\mid x) \,\|\, \pi_{\mathrm{ref}}(y\mid x) \right].
$$

第一项是在策略 $\pi$ 下的期望奖励；第二项是让策略 $\pi$ 保持靠近参考策略 $\pi_{\mathrm{ref}}$（也就是我们预训练出来的那个）的 KL 距离。

### 第一步：非参数假设，得到闭式解

现在做**唯一一个强假设**：**策略 $\pi$ 不是一个神经网络，而是"所有可能策略的集合"——一个非参数的（nonparametric）东西**，可以逼近任何函数。

如果 $\pi$ 什么都能是，上面这个问题就可以**闭式求解**——我们真的知道最大值点是什么。对每个 prompt $x$，我们用拉格朗日乘子法解约束最大化：把 $\pi(y\mid x)$ 视为变量，在约束 $\sum_y \pi(y\mid x) = 1$ 下最大化

$$
\sum_y \pi(y\mid x) r(x,y) - \beta \sum_y \pi(y\mid x) \log \frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}.
$$

对 $\pi(y\mid x)$ 求导并令其为零，得到

$$
r(x,y) - \beta\left( \log \frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)} + 1 \right) + \lambda = 0,
$$

解出

$$
\pi(y\mid x) \propto \pi_{\mathrm{ref}}(y\mid x)\, \exp\left( \frac{r(x,y)}{\beta} \right).
$$

加上归一化常数后，最大值点（闭式解）就是：

$$
\pi^{*}(y\mid x) = \frac{1}{Z(x)}\, \pi_{\mathrm{ref}}(y\mid x)\, \exp\left( \frac{1}{\beta} r(x,y) \right),
\qquad
Z(x) = \sum_{y} \pi_{\mathrm{ref}}(y\mid x)\, \exp\left( \frac{1}{\beta} r(x,y) \right).
$$

这个结果非常直观：**每个回答按照 $r$ 的好坏被指数级地上调或下调权重**——奖励很差就指数级压低，奖励很好就指数级抬高。这正是"如果 $\pi$ 可以任意"时的完美解。

### 第二步：解出"隐含奖励"

既然我们知道了这个完美解 $\pi^{*}$，就可以反过来**解出"隐含奖励"（implied reward）**：什么样的 $r$ 会诱导出这个 $\pi^{*}$？把上式改写，奖励就"现出原形"：

$$
r(x,y) = \beta \log \frac{\pi^{*}(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)} + \beta \log Z(x).
$$

（这个"策略 ⇄ 奖励"的等价关系，也正是 kimi-think 那篇论文里用到的等价性。）

### 第三步：把隐含奖励代入奖励模型的训练目标

接下来，我们把隐含奖励当作奖励模型，代入 Stiennon 那套**成对（Bradley–Terry）目标**：

$$
\mathcal{L}(r) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}
\left[ \log \sigma\big( r(x,y_w) - r(x,y_l) \big) \right].
$$

把隐含奖励代进去，注意 $Z(x)$ 在相减中**恰好抵消**：

$$
r(x,y_w) - r(x,y_l)
= \beta \log \frac{\pi^{*}(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
- \beta \log \frac{\pi^{*}(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}.
$$

最后把 $\pi^{*}$ 换回我们实际要训练的参数化策略 $\pi_\theta$，就得到 **DPO 目标**：

$$
\mathcal{L}_{\mathrm{DPO}}(\pi_\theta; \pi_{\mathrm{ref}})
= -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}
\left[
\log \sigma\left(
\beta \log \frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
- \beta \log \frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right)
\right].
$$

![DPO 的推导：从 RLHF 目标出发，经非参数闭式解与"隐含奖励"，代入成对偏好目标，得到 DPO 目标](/lectures/15/slide-56.png)

关键步骤就三步：

1. **做非参数假设**——把 $\pi_\theta$ 和 $r$ 以闭式形式联系起来；
2. **用策略来参数化奖励 $r$**；
3. **用监督损失优化这个奖励**（而这反过来就优化了策略）。

概念上讲：这就是在**非参数假设 + 换一种参数化**之下的、对成对奖励的**最大似然（MLE）**。

### 第四步：DPO 的更新形式——"对好的正梯度、对坏的负梯度"

把 DPO 目标的梯度写出来，是最直观的形式。记

$$
u = \beta \log \frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
- \beta \log \frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)},
$$

那么 $\mathcal{L}_{\mathrm{DPO}} = -\mathbb{E}[\log\sigma(u)]$。利用 $\frac{d}{du}\log\sigma(u) = 1 - \sigma(u) = \sigma(-u)$，有

$$
\nabla_\theta \mathcal{L}_{\mathrm{DPO}}
= -\beta\, \mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\Bigg[
\sigma\!\left(
\beta \log \frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
- \beta \log \frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
\right)
\cdot
\Big( \nabla_\theta \log \pi_\theta(y_w\mid x) - \nabla_\theta \log \pi_\theta(y_l\mid x) \Big)
\Bigg].
$$

可以看到两部分：

- **$\nabla_\theta \log \pi_\theta(y_w\mid x)$（正项）**：提高被偏好样本的对数似然；
- **$-\nabla_\theta \log \pi_\theta(y_l\mid x)$（负项）**：降低被拒绝样本的对数似然。

而前面的 sigmoid 权重扮演了**步长缩放**的角色——它正是"隐含奖励模型的预测误差"：如果我的模型已经给赢家分配了很高的概率（隐含奖励判断正确），$\sigma(\cdot)$ 里 $y_l$ 项减去 $y_w$ 项是个很大的负数，sigmoid 接近 0，**步长很小**；如果我很"错"，把两者判得几乎相等，sigmoid 接近 $1/2$，**步长就大**。所以基于概率差异，这个"差分对数概率"目标会走更大或更小的步。

## DPO 在 LLaMA 及其他开源模型中的应用

DPO 比 PPO 简单太多——它只是取梯度，而我们都知道怎么用标准工具做梯度。而且它**效果确实不错**。有一阵子大家非常执着于"DPO 比 PPO 好还是反过来"，现在答案大概是：**除非你在前沿训练最好的模型，否则区别没那么大**。DPO 相当好——"对 LLaMA 足够好，对我也足够好"（Tatsu 原话）。

如果去翻 LLaMA 技术报告，你会看到它本质上是一个漂亮的外层循环训练：SFT → DPO → 用 DPO 模型生成候选 → 拒绝采样（rejection sampling）→ 重复。外层有循环，但**LLaMA 的核心 RLHF 原语是 DPO**。

## 变体：SimPO 与长度归一化 DPO

之后出现了一大堆 DPO 变体，其中两个值得一提（来自 Tulu 3 论文）：

- **SimPO**：**去掉参考模型（no ref）**，把 $\pi_{\mathrm{ref}}$ 换成一个由长度归一化的项（类似 $\gamma / L$ 之类的形式）；
- **长度归一化 DPO（length-normalized DPO）**：按回答长度做归一化，以避开前面讲过的那些长度黑客问题。

但说真的，**这些变体似乎都没有太大差别**。

### PPO vs DPO：结果高度依赖实验设定

Tatsu 觉得很有意思的一点是：**这类结果对实验设定的细节极其敏感（highly contingent）**。即使在 AI2 内部：

- 有一篇论文说"从 DPO 换成 PPO 你会得到更好的结果"；
- 但 Tulu 2 的论文又说"如果你把 DPO 做对，DPO 可以胜过 PPO"。

所以**取决于你怎么执行，谁更好是不确定的**。你大概有读深度学习论文时那种经验：结果往往很脆弱。也许这里的结论是：**这些 DPO 变体其实已经足够接近"正确的事情"**，在很多情况下都能给你相当好的性能——"朝对的方向走正梯度、朝坏的方向走负梯度"这个核心想法，只要你把步长设对，就相当管用。

## 副作用一：过度优化（overoptimization）

最后几页讲 RLHF 要当心的东西。**过度优化（overoptimization）可能是最大的一桩**。给一点历史背景：InstructGPT 刚出来时大家非常兴奋，当时有一个非常现实的问题——"我们能不能靠 RLHF 一路通到超级智能？也许只要收集足够多的赞/踩就行？"事实证明这**相当困难**，因为：

> 如果你真的把 RLHF 过程使劲往前推，你会开始**过拟合你学到的奖励模型**。

也就是说，奖励模型只是人类偏好的一个近似；优化得太狠，模型就会**去钻奖励模型的空子**，而不是真正变好。这正是前面那个 KL 正则项存在的意义——**KL 正则化在很多情况下至关重要**，它能防止优化过程对奖励模型过拟合（至少在你的优化过程非常强的时候是这样）。

![过度优化：对很多不同的 RLHF 式优化器来说，优化奖励会越过某个点之后过拟合——对人工偏好（左）与带噪声的 LM 偏好（中）成立，但对无噪声的 LM 偏好（右）不成立](/lectures/15/slide-63.png)

幻灯片上的曲线来自相关研究：横轴是 KL（离参考策略有多远），纵轴是"真实"奖励。三种设定分别是**人类偏好**（左）、**带噪声的 LM 偏好**（中）与**无噪声的 LM 偏好**（右）：前两种都能看到"先升后降"的过拟合曲线，而第三种（无噪声，比如验证器给出确定正确的奖励）不会。这提示我们：**偏好的噪声越大，过度优化越严重**。

## 副作用二：模式坍缩与校准

第二个要当心的是**模式坍缩（mode collapse）**。人们很多次看到：RL 模型的**多样性低得多**——输出集中在少数几个可能的结果上。这正好接回前面讲的概念：**RLHF 之后的模型不再建模一个分布**（分布天然自带多样性），它是一个**只要拿到好奖励就可以坍缩的策略**。这是 RLHF 模型近几年一直挣扎的问题。

![RLHF 之后模型不再"概率化"：默认不校准——GPT-4 时代 OpenAI 公开承认的未解决问题之一](/lectures/15/slide-64.png)

Tatsu 因为时间关系把这块快速带过，但强调**熵与模式坍缩的问题到今天依然相当重要**。GPT-4 时代是 OpenAI 少有的公开列出"还剩几个开放问题"的时候，其中就包括：**"我们的模型在做完 RLHF 之后是不校准（uncalibrated）的"**——幻灯片上可以看到那张图。**至今没人真正解决它**。Anthropic 的人则论证说：模型是"天然不校准"的，你可以有时重新校准、但并非总是可以。

这一点在下一讲讲 RLVR（基于可验证奖励的强化学习）时**非常关键**——在那类设定里，**熵与探索**对模型"探索所有可能的解法、在难题上取得进展"至关重要。

## 全讲总结

把整堂课收个尾，RLHF 的要点是：

1. **RLHF 数据收集（同样）很难**——混杂因素非常多（风格、长度、人口统计、注释者质量……）；
2. **RLHF 算法比 SFT 复杂**——尤其 PPO；下一讲还会多讲一点，而且你们作业里会用一个更简单的变体 **GRPO**，效果也相当好；
3. **要当心（过度）优化奖励的影响**——过度优化与模式坍缩都是真实存在的风险。

最后是通往下一讲的过渡：**有没有"我们不会过度优化"的奖励？** 即那种"你只管往里面堆计算，模型性能就单调变好"的奖励——这正是 RLVR 如此有影响力的原因之一。谢谢大家，周四见。

<!-- lecture-nav -->

**→ 下一讲**:[16 Post-Training - RLVR](../16/)
