---
title: 03 · 归一化:pre-norm、LayerNorm 与 RMSNorm
lecture: 3
---

# 归一化：pre-norm、LayerNorm 与 RMSNorm

第一板块是**常见架构变体**，我们从归一化（normalization）讲起。关于架构，人们可以争论很多事情，但**有一件事几乎所有人都同意**——就是归一化放在哪里。如果拿原始 transformer 论文说事，很多人会说"transformer 作者大部分东西都做对了，**唯独这一处没做对**"：那就是 **LayerNorm 的位置**。这一节就把整个归一化的故事讲完，它几乎是一条**铁律**:

- **基本上每个人都用 pre-norm**，或者至少把 LayerNorm 放在残差流（residual stream）之外；
- **几乎所有人都用 RMSNorm** 而不是 LayerNorm;
- 推广开来，**大多数现代 transformer 没有 bias 项**。

## Pre-norm 与 post-norm：唯一人人同意的选择

先看术语。原始 transformer 的 LayerNorm 放在**残差路径（residual path）里**——也就是残差流的末端。回想一下 transformer 的结构：有一条残差流 $x$ 贯穿整个网络，每个子层（注意力、FFN）把它们的输出 $\Delta$ 加回这条残差流；原始论文在每个子层**之后**、加回残差流的位置放一个 LayerNorm，来保证梯度在层与层之间保持稳定。这种放法被称为 **post-norm**（或者叫 **residual norm**，因为归一化被放进了残差层里）。

与之相对的 **pre-norm** 则把 LayerNorm 移到**残差流之外、每个计算之前**：在多头注意力之前放一个，在 FFN 之前再放一个。

$$\text{block}(x) = x + \text{Attn}(\text{LN}(x)), \qquad x \leftarrow x + \text{MLP}(\text{LN}(x))$$

![](/lectures/03/slide-10.png)

**几乎所有现代语言模型都把 LayerNorm 推到残差流之外**。这基本上是人人遵守的惯例,不过也要注意:**BERT 用的是 post-norm**(它训练的是编码器,当时就是这么定的)。真正"滑稽"的例外是 **OPT-350M**——一个自回归解码器却在残差流里用 post-norm,Tatsu 也搞不懂为什么偏偏是它,只能猜测这个模型在训练时出了什么岔子、后来就被"遗弃"了。

### 最初的动机:去掉 warm-up

你可能会好奇:为什么这个选择如此统一?回到早期的归一化位置研究,你会发现最初的动机是**训练时要做学习率预热(warm-up)**。现代 transformer 训练仍然需要 warm-up,但当时的问题是:**能不能把 warm-up 去掉?** 这就是早期一堆研究(LayerNorm 位置)的出发点。

很快人们就意识到,去掉 warm-up 会带来严重的稳定性问题。如果你用"post-norm + LayerNorm"(即原始 transformer 的配置),即使在无 warm-up 的设置下也能训练,但收敛效果远不如 pre-norm——pre-norm 可以在**没有 warm-up** 的情况下获得漂亮得多的收敛曲线。所以最初的理由是"pre-norm 可以省掉 warm-up"。

### 真正的解释:梯度衰减与梯度尖峰

但随着网络变深、稳定性问题凸显,人们意识到把 LayerNorm 移出残差流有更重要的含义。Tatsu 认为最清晰的是**梯度衰减**(gradient attenuation)问题。做架构设计的人常挂在嘴边的一句话是:**"保持你的残差流干净(keep your residual stream clean)"**。在 pre-norm 下,输入 $x$ 从网络底部一路直通到顶部输出,这条**恒等(identity)通路**让梯度在反向传播时可以"直线穿透",于是:

- **梯度传播非常漂亮**:pre-norm 下,从初始化开始,各层的梯度大小基本保持不变;而 post-norm 下,每穿过一个 transformer block 都要做一次 LayerNorm,这会不断改变梯度的范数,带来各种复杂效应;
- **稳定性更好**:实验显示,pre-norm 相比 post-norm,梯度尖峰(gradient spikes)的**大小和频率**都显著改善。这张图来自 Salazar 与 Nguyen(2019),他们是最早仔细研究这一现象的团队之一。

正是**稳定性 + 能训练得更深**这两条,让"把 LayerNorm 移出残差流"成为几乎所有模型采纳的做法。

### 新东西:double norm(非残差 post-norm)

既然 LayerNorm 放进残差流是坏事,那**为什么 LayerNorm 非得放在 block 前面?** 放在计算**之后**也一样是残差流之外,至少从这个逻辑看同样成立。事实正是如此:

- **Grok** 和 **Gemma 2** 采用了这种结构:把 LayerNorm 移到计算**之后**(也是一种 post-norm,但在残差流之外);
- 更进一步,有些模型干脆**前后各放一个**(double norm);
- **OLMo 2** 则只在注意力块和 FFN 块**之后**做非残差 post-norm。

Tatsu 还透露了另一个"有点离谱但屡试不爽"的经验:**如果遇到稳定性问题,就往模型里到处撒 LayerNorm**——放进注意力里,放进各种地方,几乎总是能提升稳定性。这句话听起来很荒谬,但一次又一次被验证。这一点在后面讲稳定性技巧时还会再见到。

## LayerNorm 与 RMSNorm

原始 transformer 用的是 **LayerNorm**:对激活 $x$ 减去经验均值 $\mu$、除以标准差(加一个小常数 $\varepsilon$ 防止除零),再乘可学习的缩放参数 $\gamma$、加可学习的偏移参数 $\beta$:

$$y = \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} \cdot \gamma + \beta, \qquad \mu = \frac{1}{d}\sum_i x_i,\ \ \sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2$$

其中 $d = d_{model}$，即对整个模型维度做归一化。

**RMSNorm** 则**不减去均值、也不加 bias 项**：只按均方根（RMS）缩放，再乘 $\gamma$:

$$y = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \varepsilon}} \cdot \gamma$$

按使用的归一化可以把模型分成两派:

- **LayerNorm**:GPT-3/2/1、OPT、GPT-J、BLOOM;
- **RMSNorm**:LLaMA 家族、PaLM、Chinchilla、T5。

### 为什么 RMSNorm?——"更快,而且一样好"

论文里给出的理由通常是:**RMSNorm 更快,而且效果一样好**。快在哪里?

- **更少的操作**:不用计算均值;
- **更少的参数**:不用存储 bias 项 $\beta$,也就是从内存搬运到计算单元的参数更少。

**但这个解释真的说得通吗?** 你可能立刻会反驳:上一讲(第 2 讲)不是刚讲过,**除了矩阵乘法,什么都不重要**吗?RMSNorm 又不是矩阵乘法,为什么要关心它?

这个反驳有道理,**前提是只盯着 FLOPs**。有一篇 2023 年的论文(Ivanov 等人,《Memory Movement Is All You Need》)对 transformer 各组件做了剖析:矩阵乘法(张量收缩)占了 transformer 中约 **99.8% 的 FLOPs**。照这么说,省下 RMSNorm 那一点点 FLOPs 似乎毫无意义。

### 关键教训:FLOPs 不等于运行时间(runtime)!

**FLOPs 是浮点运算的次数,不是墙钟时间。** 运行时间是一个复杂得多的东西。统计归一化(如 LayerNorm)虽然只占约 **0.17% 的 FLOPs**,但在某些 workload 和配置下**可能占到 25% 的运行时间**——尤其是小模型上,这一比例可以非常夸张,因为做这些操作时你仍然要把所有参数在快慢存储之间搬来搬去。**数据搬运(data movement)极其重要**,RMSNorm 正是靠减少数据搬运来省时间的。

![](/lectures/03/slide-16.png)

(图中的白色部分是算术强度、黑色部分是 FLOPs:可以看到 LayerNorm 的算术强度非常低——这正是我们最想删掉的那种操作。)

这正是**系统与架构协同设计**(co-design)开始介入的地方:Percy 在上一讲提到的**算术强度**(arithmetic intensity)概念——我们想让 GPU 保持"火热",尽量多做矩阵乘法这类高强度运算,而不是浪费在搬来搬去小块内存上。按这个观点:**既然减均值、加 bias 并没有带来多少表达力,那就把它删掉。**

### 问答:为什么归一化的数据搬运如此不成比例?

(有同学提问:归一化的数据搬运量为什么和计算量这么不成比例?)

Tatsu 解释:像张量收缩(即矩阵乘法)这类操作,工作量的绝大部分花在**乘法本身**;而统计归一化这类操作,工作量的绝大部分花在**内存移动**上,而内存移动很慢。想象一个操作里"搬东西"几乎就是全部工作,即使激活规模很大你也得照付这笔钱。他还补充:幻灯片里那个"25% 运行时间"是**小模型**的极端情况(那些矩阵规模在现代 workload 里其实不太常见),但它传达的道理是成立的——**这是一次免费的优化(free optimization)**。

### 验证:Narang 等人的消融

上面说的不是空谈。Narang 等人 2020 年(Google 的论文)系统评估了各种架构干预:对一个约 **2 亿参数**的小 transformer,**换用 RMSNorm 后每秒能跑更多步数**(幻灯片第三列),而且**性能反而更好**——后者并不是必然的,但作为"附赠"很 nice。也就是说:**白捡一个系统层面的胜利**。这正是大家纷纷转向 RMSNorm 的原因。

## 推广:去掉 bias 项

与 RMSNorm 一脉相承的最后一点是:**大多数现代 transformer 没有 bias 项**。原始 transformer 的 FFN 长这样(每个线性层都带 bias):

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

而大多数现代实现（如果不是门控单元）长这样——**直接把 bias 扔掉**:

$$\text{FFN}(x) = \sigma(xW_1)W_2$$

理由和 RMSNorm 是同一套:

- **系统视角**:bias 也是"算术强度不高、但相对而言内存密集"的操作,删掉就是免费的系统优化;
- **优化稳定性**:Tatsu 顺带提到,有些情况下 bias 项还会**诱发稳定性问题**。但主因还是系统层面——**去掉 bias 只是为了让事情更简单**。

Tatsu 坦承:**这类事情没法事先推理出来**。我们不可能预先知道"去掉 bias 也没事",这些结论完全来自大量实验和集体积累的经验——至少在典型语言建模 workload 上,去掉线性层和 RMSNorm 的 bias 是安全的。

## 小结:归一化的"铁律"

归一化部分的故事非常清晰,你只需要记住:

1. **基本上人人做 pre-norm**,或者至少把 LayerNorm 放在残差流之外——这几乎是铁律。直觉是"保持残差流干净",观察是"梯度传播更漂亮、尖峰更少";有些人还会在残差流之外**再加第二个 norm**(double norm);
2. **几乎人人用 RMSNorm**:实践上它和 LayerNorm 效果几乎一样,但需要搬运的参数更少、操作更少,省墙钟时间;
3. **去掉 bias 项**广泛适用:很多模型大部分地方都没有 bias,因为"计算/参数交换比"不划算。

带着这些背景,下一节看**激活函数与 FFN**——从 ReLU 一路走到 SwiGLU。
