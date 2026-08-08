---
title: "17 · SigLIP：用二分类损失训练图像编码器"
lecture: 17
---

# SigLIP：用二分类损失训练图像编码器

## 动机：CLIP 的两个技术缺点

现在我们已经有了 CLIP 这样的图像编码器：它把一张固定尺寸（比如 336×336）的图像映射成一个向量，向量里带着图像的一些**语义**。但在深入之前，值得把 CLIP 的两个技术缺点再强调一遍，因为下一节的主角——来自 Google 的 **SigLIP**（Sigmoid Loss for Language Image Pre-Training，用于语言-图像预训练的 sigmoid 损失）——正是冲着它们来的：

1. **需要非常大的 batch size**（比如 3 万）。batch size 为 1 显然不工作，甚至 10 都不工作；
2. **softmax 要在整个 batch 上操作**，所以损失很难"分解"——没法像普通语言模型训练那样，把一个 batch 里的各条序列完全并行、最后只做一次聚合。

对比一下普通语言模型训练：一个 batch 里所有序列是互相独立的，各自算损失、最后聚合一下就行。而 CLIP 的损失本质上是一个横跨整个 batch 的全局结构，这给并行化带来了麻烦。

## SigLIP 的目标：从多分类到二分类

SigLIP 可以看作是 CLIP 的改进版。核心区别在于损失的"语义"：

- **CLIP** 做的是**多分类（multiclass classification）**：对于对齐的（文本，图像）对，要把"这一对"对**所有其它图像**、**所有其它文本**区分开来——也就是"我的配对是正的，其它全都是负的"；
- **SigLIP** 要简单得多：对**任意给定的（图像，文本）对**，只回答一个问题——**"它们对齐了吗？"（aligned or not？）**

回到 CLIP 那张 $N\times N$ 矩阵图：对角线上的配对是正样本，非对角线上的都是负样本。CLIP 对每行每列各做一次 $N$ 分类；而 SigLIP 把**每个矩阵元素都当成一个独立的二分类问题**——匹配了给 $+1$，没匹配给 $-1$，如此而已。

### 形式化与代码

设图像与文本嵌入归一化后为 $\hat I_i, \hat T_i$，定义 logit

$$
z_{ij} = \gamma\, \langle \hat I_i, \hat T_j \rangle + b,
$$

其中 $\gamma$ 是（对数尺度的）温度、$b$ 是偏置。标签取

$$
y_{ij} = \begin{cases} +1 & \text{若 } i = j \text{（匹配对）}\\[2pt] -1 & \text{若 } i \neq j \text{（不匹配对）} \end{cases}
$$

于是每个元素都是一个带 sigmoid 的二分类项，总损失为：

$$
\mathcal{L}_{\text{SigLIP}} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{N}\log\sigma\!\left(y_{ij}\, z_{ij}\right),
$$

其中 $\sigma$ 是 logistic sigmoid。注意：当 $y_{ij}=-1$ 时，$\log\sigma(-z_{ij}) = \log(1 - \sigma(z_{ij}))$，正好是对"负例"的标准二分类损失。实现起来非常直接：

```python
def siglip_loss(image_embeddings, text_embeddings, logit_scale, bias):
    # 归一化后计算相似度矩阵
    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
    text_embeddings  = text_embeddings  / text_embeddings.norm(dim=-1, keepdim=True)

    logits = logit_scale * image_embeddings @ text_embeddings.t() + bias

    # 标签：对角线为 +1（匹配），其余为 -1（不匹配）
    labels = -torch.ones_like(logits)
    labels.fill_diagonal_(1.0)

    # 逐元素二分类损失：-log sigmoid(logits * labels)
    loss = -F.logsigmoid(logits * labels).mean()
    return loss
```

![SigLIP 的目标：对每个（图像，文本）对做独立的二分类——匹配（+1）还是不匹配（-1）](/lectures/17/siglip-code.png)

> **课堂问答：负样本需要专门的采样策略吗？**
>
> （同学：一般的对比学习方法里，负样本的采样结构很重要——大多数样本不是那么随机的，损失里可能需要注意负样本的平衡之类的？）
>
> **Tatsu**：问的是需不需要复杂的采样策略。至少在最初那篇论文里，他们就是**直接在同一个矩阵上操作**，没有搞任何花哨的东西。你可以想象，一般来说这类对比方法确实需要考虑负样本的平衡、或者用"硬负样本（hard negatives）"之类的手段避免偏差——但至少在最初版本里，SigLIP 就是相当朴素的。

## 数据：WebLI

SigLIP 是在 Google 做的，数据用到了 **WebLI** 数据集——这是另一篇 2022 年的图文模型论文使用的数据集，规模是**十亿（order of billion）量级的（图像，文本）对**。几个关键点：

- 数据是**从互联网上抓取（scraped）**的；
- 他们还额外做了一件事：对**图像里含有文字**的情况做**自动 OCR**，把识别出的文字当作文本，从而形成更多的（图像，文本）对；
- 做了一定的过滤，最后**保留质量最高的 10%**；
- 数据集是**多语言的，支持约 100 种语言**。

## 效率：为什么 SigLIP 训练起来快得多

SigLIP 论文最主要的价值点，是**训练效率远高于 CLIP**。直观对比一下：

| 方法 | 硬件 | 训练时长 |
|------|------|----------|
| CLIP | 256 个 TPUv3 | 10 天 |
| SigLIP | 32 个 TPUv4 | 5 天 |

你可能会想：TPUv4 应该比 TPUv3 快吧？**其实不是**。从单芯片的 FLOP/s 来看，TPUv4 **反而更慢**——大约慢 60% 左右。TPUv4 的优势在于：一个 pod 里可以塞进更多的芯片、互连更好。所以 SigLIP 之所以快，**不是靠更快的芯片，而是靠更好的并行化**。

当然，CLIP 当年大概也没有为最大化吞吐去优化代码——"能把模型训出来"就是目标。而 SigLIP 论文展示了可以这样并行化：

![SigLIP 的并行化：把图文对分布到各设备，像 DDP 一样各自算本地损失，再轮转交换文本嵌入来覆盖非对角块](/lectures/17/siglip-parallelism.png)

回忆一下系统课的**数据并行（data parallelism，DDP）**：每个设备存一部分（图像，文本）对。但这里的麻烦是——**样本之间存在交互**，不像语言模型训练那样一切都可以分解。于是做法是：**第一轮**，每个设备在本地持有的图文对上计算所有（本地）损失；然后**把文本嵌入发出去轮转**——比如设备 1 先收到 $T_5$ 到 $T_8$，从而能算这一块的非对角负样本；接着收到 $T_9$ 到 $T_{12}$，再算下一块……如此循环，直到覆盖矩阵里所有的非对角块。这样就把"跨设备的全矩阵损失"变成了可以流水化的通信-计算模式。

## Batch size：把 batch size 从损失中解耦

SigLIP 的另一个优点：**有效解耦了 batch size 与损失**。

- 在 CLIP 里，损失**绑定**在 batch size 上：你改 batch size，就等于改了损失函数——因为 $N$ 分类问题的类别数就是 batch 里的样本数；
- 在 SigLIP 里则不然。因此他们可以放心地实验**小得多的 batch size（小于 16K）**，在这个区间 SigLIP **明显优于 CLIP**——CLIP 在 batch 太小时损失函数本身会退化，而 SigLIP 只是**方差变大**，但**期望意义下的损失是一样的**；
- 反过来，batch 可以推到很大（他们实验过**高达 100 万**的 batch），但**并没有额外的收益**；
- 他们发现存在一个**临界 batch size（critical batch size）**，大约在 **32K** 附近——再大就没什么用了。这个"临界 batch size"的概念，正是第 9 讲讲缩放定律时出现过的那个。

## 小结

CLIP 与 SigLIP 都是"图像编码器"：输入一张固定尺寸的图像，输出一个携带语义的向量。它们构成了后面所有 VLM 的"视觉半边"。接下来，就是本讲真正的主题——**如何把这些嵌入注入语言模型，构建视觉语言模型（VLM）**。

<!-- lecture-nav -->

**← 上一节**：[CLIP：对比语言-图像预训练]（02-clip.md）　**→ 下一节**：[LLaVA 与 LLaVA OneVision：把图像注入语言模型](04-llava.md)
