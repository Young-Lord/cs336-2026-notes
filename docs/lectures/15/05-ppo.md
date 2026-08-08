---
title: "15 · PPO：从策略梯度到裁剪"
lecture: 15
---

# PPO：从策略梯度到裁剪

## 目标函数：最大化奖励，但别走太远

现在进入算法部分。还记得我们处在 RLHF 的设定里：**目标是在我们的策略下，最大化"从该策略采样"所能得到的奖励**。这其实是"婴儿级"强化学习——有人说我们只是在玩**赌博机（bandits）**，这并不是真正的多轮 RL，大致没错。所以我们的算法也会相当简单。

这个"最大化奖励"的目标几乎原封不动地出现在 InstructGPT 论文里：如果你翻到它的第（2）个公式，会看到正是我们写下的东西。用我们的记号，RLHF 的目标是：

$$
\max_{\pi_\theta}\ \mathbb{E}_{x\sim\mathcal{D},\ y\sim\pi_\theta(y\mid x)}\left[ r(x,y) \right]
- \beta\, \mathbb{D}_{\mathrm{KL}}\left[ \pi_\theta(y\mid x) \,\|\, \pi_{\mathrm{ref}}(y\mid x) \right],
$$

其中第一项是从 RL 策略 $\pi_\theta$ 采样、最大化奖励 $r(x,y)$；第二项是**KL 散度**——它只是在说：**我想离我的参考策略（通常是 SFT 模型）近一点**，因为我不想到处乱跑、变成退化模型。

![PPO 在语言建模中的应用（InstructGPT）：目标 = 期望奖励 − KL 散度，看起来人畜无害](/lectures/15/slide-51.png)

同样的结构也出现在 Stiennon 等人的论文（*Learning to summarize from human feedback*）里。在那个设定中，奖励 $r(x,y)$ 是一个**成对反馈模型（pairwise feedback model）**：先在"一对例子中哪个更好"上训练一个二分类器，然后在那个奖励上做爬山（hill-climbing）。非常简单的一套结构。

奖励模型的具体训练方式是**Bradley–Terry（成对）目标**：给定 prompt $x$、被偏好的回答 $y_w$（winner）与被拒绝的回答 $y_l$（loser），我们希望被偏好者的奖励尽量高于被拒绝者的奖励：

$$
\mathcal{L}(\phi) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}
\left[\log \sigma\big( r_\phi(x,y_w) - r_\phi(x,y_l) \big)\right],
$$

其中 $\sigma$ 是 sigmoid 函数。这就是"训练一个验证器"的核心步骤。

## PPO 的三步演进：策略梯度 → TRPO → PPO

那么具体怎么爬山？用的是叫 **PPO**（Proximal Policy Optimization，近端策略优化）的算法。如果你上过强化学习课，当然知道 PPO；如果没上过，Tatsu 会给一个"婴儿版"描述——而且课程作业里你也会需要理解它。

### 第一步：策略梯度（方差太高）

一切的起点是**策略梯度恒等式（policy gradient identity）**。我想最大化"从我的策略采样得到的奖励"：

$$
\nabla_\theta \mathbb{E}_{z\sim p_\theta}[R(z)] = \mathbb{E}_{z\sim p_\theta}\big[ R(z)\, \nabla_\theta \log p_\theta(z) \big].
$$

推导并不难：把梯度推进期望里，利用 $\nabla_\theta p_\theta(z) = p_\theta(z) \nabla_\theta \log p_\theta(z)$ 即可。它的形式**看起来就像 SFT，只不过每个例子带了权重**——用奖励给每个样本加权。问题在于：**策略梯度的方差太高**。要估计这个梯度，你需要不停地从当前策略采样。

### 第二步：TRPO（线性化 + 约束）

你可能会说："策略梯度很棒，但问题是每做一次优化步都要采样——而采样（生成）很贵。"我们在系统部分讲过：**推理（inference）常常又复杂又难，而训练是算数密集型、很"好"的那种计算**。所以你想**滚动（roll out）一次、复用很多次**——这就是**离策略（off-policy）**。

要做离策略，你可以一次迈多步，但**不能走太远**：走太远的话，你对局部奖励的估计会爆炸。这就引出了 **TRPO**（Trust Region Policy Optimization）：基本想法是照常做策略梯度，但要**保持在当前位置附近**——通过一个重要性权重（importance weighting）修正来实现。Tatsu 不打算深入细节。

### 第三步：PPO（裁剪比率）

PPO 说：**TRPO 是个好主意，但它那个"距离约束"很难处理**。于是 PPO 提出了一个**启发式的裁剪（clipping）**机制：干脆**抑制 RL 算法跑到离原策略太远的地方**。具体地，定义比率

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)},
$$

PPO 的目标是：

$$
L^{\mathrm{CLIP}}(\theta) = \mathbb{E}_t\left[
\min\Big( r_t(\theta)\, \hat{A}_t,\;
\operatorname{clip}(r_t(\theta),\, 1-\varepsilon,\, 1+\varepsilon)\, \hat{A}_t \Big)
\right],
$$

其中 $\hat{A}_t$ 是优势函数（advantage）的估计。直觉是：当比率超出 $[1-\varepsilon, 1+\varepsilon]$ 时，目标被"裁剪"，不再鼓励策略继续往那个方向漂移——一个廉价、启发式但有效的替代 TRPO 约束的办法。

Tatsu 把这三步的进度画在幻灯片上：**策略梯度（方差太高）→ TRPO（在当前策略附近线性化问题）→ PPO（把比率裁剪到某个 $\varepsilon$ 内）**。下一讲他会把细节讲完整，今天只需要大致理解这条演进路径。

## 能不能把 PPO 干掉？

PPO 的公式已经有点狰狞了。于是很多人多年来都在问同一个问题：**能不能不做任何"真正的 RL"（即 on-policy 的 RL 算法）？** 这是很多人认真想过的事，Tatsu 把那些"合理的尝试"列了出来——免得你在自己的研究里再踩一遍：

1. **用控制 token 训练模型**：对成对数据做 SFT，给被选中的回答前面加一个 `[GOOD]` token、给没被选中的加 `[BAD]` token。生成时只要在前缀写上 `[GOOD]`，模型就只会生成好东西。这是**把 RL 归约为 SFT**——**不行**；
2. **只训练被偏好的输出**：也**不太行**；
3. **训练奖励模型，得到 LM 输出，然后只选奖励模型选中的输出进行训练**：**不如 PPO**，但**部分有效**；
4. **训练奖励模型，采样 1024 个 LM 输出，取最好的一个**：也是类似的思路。

很多方法都试过了，最终我们确实有了一个比 PPO 简单得多、效果不错、而且在很多方面看起来就像 SFT 的东西——那就是 **DPO**。这是下一节的内容，也是"又酷又好玩"的部分。

## 小结：PPO 的定位

PPO 是 RLHF 的"原始且非常挑剔（finicky）"的方法：目标函数本身简单（期望奖励 − KL 散度），但实现上要处理策略梯度的高方差、离策略带来的偏差，以及各种工程细节。它是当前语言模型对齐事实上的标准 RL 构件之一；而 DPO 则试图把它的一切复杂性剥掉。下一节我们就看 DPO 是怎么做到这一点的。

<!-- lecture-nav -->

**← 上一节**：[RLHF 的数据与注释者经济]（04-rlhf-data-and-annotation.md）　**→ 下一节**：[DPO 与 RLHF 的副作用](06-dpo-and-side-effects.md)
