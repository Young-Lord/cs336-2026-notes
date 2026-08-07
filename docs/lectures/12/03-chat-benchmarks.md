---
title: "12 · 聊天类基准：人类偏好、LLM 评委与 Arena"
lecture: 12
---

# 聊天类基准：人类偏好、LLM 评委与 Arena

## 开放回答的评估难题

到目前为止，我们评估的都是定义良好的多选题任务。但**大多数人并不会向他们的 AI 助手问多选题**——除非是想让助手替自己做作业，而那是绝对没有人会做的事。真实的用户提问是开放式的。比如这样一个半真实的例子：

> **提示词**：*I would like to make a beet salad with goat cheese. What kind of herbs would work well and what would not work well?*（我想做一道配山羊奶酪的甜菜沙拉。哪些香草合适、哪些不合适？）
>
> **回答**：*Here's a breakdown of herbs that work well (and some that don't) in a beet + goat cheese salad, based on how their flavors interact with the sweet-earthiness of beets and the tangy creaminess of goat cheese...*（以下是关于香草搭配的分析：有些香草和甜菜的甜、山羊奶酪的酸奶油感很搭，有些则不行……）

![Chatbot Arena 上的甜菜沙拉问答示例](/lectures/12/arena-beets.png)

这是个非常开放的问题，回答也是开放式的。**怎么评估这样的回答？** 你不能用字符串精确匹配（exact match），因为根本不存在唯一的"标准答案"。这里有几个候选思路，我们逐一过一遍。

## Chatbot Arena：让真实用户做评委

第一个思路由 **Chatbot Arena**（现名 Arena AI）开创，核心是**问人类**。但它的"问法"很巧妙：它搭了一个网站，任何来自互联网的随机路人都可以进来聊天；与普通助手只给一个回答不同，这里你会拿到**两个来自不同模型、被匿名化的回答**，然后你要判断哪个更好——选项是"A 更好 / 两个都好 / 两个都差 / B 更好"。

![Arena 的成对比较界面](/lectures/12/lmarena-leaderboard.png)

通过这个过程，你能收集到大量"模型 A vs 模型 B，哪个更好"的**成对比较（pairwise comparison）**数据，然后用它们计算 **Elo 排名**。下面把 Elo 的数学完整展开。

### Elo 评分：从成对比较到评分

假设每个模型有一个评分（rating），记模型 A、B 的评分分别为 $R_A$、$R_B$。**评分模型（rating model）**定义 A 击败 B 的概率为

$$p(A \succ B) = \frac{1}{1 + 10^{(R_B - R_A)/400}}$$

直觉上：A 的评分越高，这个概率越大；分母里的 400 是控制"灵敏度"的尺度常数。取个对数就能看清它的结构——**胜率之比（odds）是评分差的对数线性函数**：

$$\frac{p(A \succ B)}{1 - p(A \succ B)} = 10^{(R_A - R_B)/400}$$

即评分每差 400 分，胜率之比就变成 10 倍。这其实就是统计里的 **Bradley–Terry 模型**：令尺度参数 $s = 400 / \ln 10 \approx 173.7$，上式可以改写成逻辑斯蒂（sigmoid）形式

$$p(A \succ B) = \frac{e^{R_A/s}}{e^{R_A/s} + e^{R_B/s}} = \sigma\!\left(\frac{R_A - R_B}{s}\right)$$

其中 $\sigma(z) = 1/(1+e^{-z})$ 是逻辑斯蒂函数。

**拟合**：我们有一堆观察到的成对比较结果 $\{(A_i, B_i, y_i)\}$，其中 $y_i = 1$ 表示 A 赢、$y_i = 0$ 表示 B 赢（平局/双好/双差可以记 $y_i = 0.5$）。把评分当作参数，用**最大似然**拟合：

$$\mathcal{L}(R) = \prod_i p(A_i \succ B_i)^{\,y_i}\,\big(1 - p(A_i \succ B_i)\big)^{1 - y_i}$$

最大化 $\mathcal{L}$（等价于最小化负对数似然；通常还要加一点 L2 正则——因为把所有评分同时加一个常数不会改变任何胜率，必须固定这个"平移自由度"）。这正是讲者在课上说的"**拟合这个模型，使成对比较的（联合）概率最大化**"。

一个有用的联系：**在线 Elo 更新其实就是对这个对数似然的梯度上升**。对单个对局 $(A, B)$，A 的期望得分就是 $E_A = p(A \succ B)$；对数似然对 $R_A$ 的偏导为

$$\frac{\partial}{\partial R_A}\Big[y\ln E_A + (1-y)\ln(1-E_A)\Big] = \frac{y - E_A}{s}$$

于是按学习率 $K \cdot s$ 做一步梯度上升，就得到经典的 Elo 更新公式：

$$R_A \leftarrow R_A + K\,(y - E_A), \qquad R_B \leftarrow R_B - K\,(y - E_A)$$

其中 $K$ 是所谓的 K 因子。用代码表示：

```python
import math

def expected_score(rating_a: float, rating_b: float) -> float:
    """A 对 B 的期望得分（即 A 获胜概率）。"""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

def elo_update(rating: float, opponent_rating: float,
               outcome: float, k_factor: float = 32.0) -> float:
    """经典 Elo 在线更新：outcome ∈ {1, 0.5, 0}。"""
    expected = expected_score(rating, opponent_rating)
    return rating + k_factor * (outcome - expected)

rating_a, rating_b = 1500.0, 1500.0
rating_a = elo_update(rating_a, rating_b, outcome=1.0)   # A 赢了
rating_b = elo_update(rating_b, rating_a, outcome=0.0)   # B 输了
```

（现代 Arena 的榜单通常是离线地、用最大似然批量拟合全部对局，而不是逐步在线更新，但两者共享同一个模型。）

### Arena 的优点

讲者随后点评了这个设定的几件好事。第一，**提示词是真实世界的**：任何人都能上这个网站，而用户来这里的动机是**免费使用语言模型**，所以隐含地假设他们真的是想做点有用的事——于是你拿到的是真实的使用提示。第二（也是 Elo 的妙处），**你不需要把相同的提示词喂给所有模型**。就像下棋时你不需要跟所有人都对弈一样，只要对局图（把模型当节点、比较当边）是连通的，你就能从稀疏的成对比较中推出排名。这对"人类当评委"的设定至关重要——你不能指望一个人去评所有的模型，甚至不能期望他一次评超过两个。

此外这个设定是**动态**的：随着新的提示词和新的模型进来，自然有办法随时间更新排名。

### Arena 的顾虑

但也要警惕很多问题。**"这些评委是谁"**？来 Arena 的"互联网随机路人"到底是什么分布，没人说得清——论文里有一些人口统计信息，但统计信息并不能说明全部。你还要担心来自不同模型的偏差、垃圾信息（spammer）、以及试图**作弊**的人——比如有人提交了自己的模型、想让它的分数好看。总之这里有点像"狂野西部"。

另一个问题是：**二值偏好（哪个更好）非常简洁、很适合喂给评分模型，但它把风格（style）和正确性（correctness）混在了一起**。Elo 用在象棋上非常合理，因为那里唯一重要的就是"你赢没赢"；但"哪个回答更好"就远没有那么清晰了。而且，**人类自己就是评委**——输入提示词的人同时是回答质量的判断者，这有一定道理，因为提问者带有意图、能判断意图是否被满足；但他显然是因为**不知道答案**才问的，那他又如何判断哪个回答是对的？最后还有**谄媚（sycophancy）**问题：更"讨喜"的回答可能比"正确但诚实"的回答得到更高的评价。总之这个设定潜在的问题不少。

## AlpacaEval：LLM 当评委与指标的自证

**AlpacaEval**（2023）换了一个思路：**让语言模型来当评委**。它有 **805 条来自各种来源的指令**；评估指标是**对基准模型（baseline）的胜率**——给定一条提示，用你的模型生成回答、用当时的基准模型（GPT-4 preview）生成回答，然后**让 GPT-4 preview 判断哪个更好**。这里立刻会有一个疑问：评委和基准是同一个模型，会不会有偏差？先按下不表。

![AlpacaEval 的榜单](/lectures/12/alpacaeval-leaderboard.png)

AlpacaEval 的确出了问题：**LLM 评委偏好更长的回答**，导致大家通过让模型"话痨"来刷榜。AlpacaEval 2.0 用**回归（regression）去偏**来修正这个指标。

但这里引出一个更普遍的问题——**如何评估一个指标本身**？我们一直在讨论怎么评估模型；如果你提出一个新指标，怎么知道它好不好？这个问题很难，没有标准答案。一个合理的**合理性检查（sanity check）**是：看它和其他指标的相关性——前提是你并不是在所有情况下都想要"更高更好"，除非你就是在模仿那个指标。AlpacaEval 与 Chatbot Arena（人类判断）的**相关性是 0.98**，非常高。这意味着，如果你想让模型在 Arena 上表现好、但又不愿（或等不及人类来评）把自己的模型放上去，你可以用 AlpacaEval 来代替。

不过讲者也提醒：这个相关性是**相对于某个特定模型集合**算出来的，对"比 GPT-4 preview 更强的模型"未必成立。另外这个榜单已经一年多没有维护了——上面就是它当时的模样。

## WildBench：真实聊天记录 + checklist 评委

**WildBench** 是另一个 LLM 评委思路的基准。像 Arena 一样，他们也搭了一个免费服务让人聊天，**从 100 万条真实的人机对话里筛出了 1024 个示例**作为评估集；又像 AlpacaEval 一样，用 LLM 当评委。它最主要的创新是引入了一张**checklist（清单，即针对该提示生成的评分标准/rubric）**：

![WildBench 的提示与 checklist 示例](/lectures/12/wildbench.png)

想法是这样的：直接问一个语言模型"这个回答好不好"，本质上是**非常良定义不足（ill-defined）**的任务——好不好取决于你在乎什么。而给评委一张 checklist 或 rubric，可以极大地**限定任务范围、让评估任务更良定义**。WildBench 同样报告了与 Chatbot Arena 的相关性——讲者坦承这有点循环论证（毕竟凭什么 Arena 就是 ground truth？），但至少"大家都在同一条船上"。

## 小结

怎么评估开放式的回答？没有干净的答案，但有一些行之有效的思路：

- **成对比较（pairwise comparison）**通常比绝对打分更可靠，尤其是两个回答很相似时，评委能给出方向性的信息（"这个比那个稍微好一点"）；而绝对打分（"这是 7/10 还是 8/10？"）的信息量要低得多；
- **始终警惕偏差**——人类评委和 LLM 评委各有各的偏差，都必须留意；一个务实的做法是**同时用多个评委（人类和 LLM）**：如果所有评委都说你的模型更好，那它大概真的更好；
- 开放回答的评估**本质上不是良定义的问题**，所以越来越重要的是**定义 rubric/checklist** 来提高评估的可靠性与良定义性——这对人类评委和 LLM 评委都成立。任何做过众包（crowdsourcing）的人都知道：不给评分标准就让人打分，得到的几乎肯定是胡言乱语。

<!-- lecture-nav -->

**← 上一节**：[考试类基准：MMLU 到 Humanity's Last Exam]（02-exam-benchmarks.md）　**→ 下一节**：[智能体基准：SWE-bench 到 MLE-Bench](04-agentic-benchmarks.md)
