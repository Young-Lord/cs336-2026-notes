---
title: "17 · CLIP：对比语言-图像预训练"
lecture: 17
---

# CLIP：对比语言-图像预训练

## 历史背景：回到 2021 年

我们把时钟拨回 **2021 年**。**CLIP**（Contrastive Language-Image Pre-training，对比语言-图像预训练）——你们中的一些人可能已经很熟悉它——是现代视觉语言模型（VLM）的一大块地基，所以我们值得把它讲得细一些。

当时的局面是这样的：GPT-3、GPT-2 已经出现，语言模型整体已经进入了"基础模型（foundation model）"时代——互联网上有海量文本，虽然非常嘈杂，但只要模型足够大，就能从中"悟"出有用的东西。然而**视觉**领域还停留在老一套：在**大规模人工标注数据集**（比如 ImageNet）上训练模型（比如 ResNet），再辅以各种数据增强（data augmentation）技巧来拿到好成绩。于是 OpenAI 的研究者们开始琢磨一个问题：**能不能利用互联网上那海量、免费的（图像，文字描述）配对？**

这个想法是受语言模型启发的：语言那边，直接爬取互联网文本、任它嘈杂，把模型训大就能产出有用的东西。那么对图像而言，等价物是什么？CLIP 因此诞生。

## CLIP 的方法：对比学习的核心思想

CLIP 的想法相当简单直接。假设我们有一大堆（图像，文本）配对，比如 **32,000** 个这样的配对。做法如下：

- 对每张图像，用一个**图像编码器（image encoder）**（后面再细讲是什么）编码成一个向量，得到 $I_1, I_2, \dots, I_n$；
- 对对应的文本做同样的事，得到 $T_1, T_2, \dots, T_n$。

这些都是文本的嵌入（embedding）和图像的嵌入。现在设定目标：以 $I_1$ 为例，它和 $T_1$ 是**对齐**（aligned）的——我们想要

$$
\langle I_1, T_1 \rangle \gg \langle I_1, T_j \rangle \quad (\forall j \neq 1),
$$

也就是 $I_1$ 与它自己的文本之间的**点积**（dot product）要远大于它与批内其它所有文本之间的点积。反过来，对于文本这一侧也一样：

$$
\langle T_1, I_1 \rangle \gg \langle T_1, I_j \rangle \quad (\forall j \neq 1).
$$

这就是 CLIP 目标的全部。你可以把它看作<strong>$2n$ 个 softmax 分类问题</strong>：对每一张图像做一次 $n$ 分类（哪个文本和它匹配），对每一段文本也做一次 $n$ 分类（哪张图像和它匹配）。

### 形式化：InfoNCE / 对比损失

更形式化地说，设批量大小为 $N$，图像嵌入与文本嵌入分别归一化后为 $\hat I_i, \hat T_i \in \mathbb{R}^d$（即 $\|\hat I_i\| = \|\hat T_i\| = 1$），并引入一个可学习的温度参数 $\tau$，定义成对 logit

$$
\ell_{ij} = \frac{\langle \hat I_i, \hat T_j \rangle}{\tau}.
$$

那么 CLIP 的损失就是两个方向的 InfoNCE 损失（Oord et al., 2018）的平均。对图像侧：以图像 $i$ 为锚点，$T_i$ 是正样本、批内其余 $N-1$ 段文本是负样本：

$$
\mathcal{L}_{\text{image}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(\ell_{ii})}{\sum_{j=1}^{N}\exp(\ell_{ij})}.
$$

对文本侧对称：以文本 $i$ 为锚点，$I_i$ 是正样本、其余图像是负样本：

$$
\mathcal{L}_{\text{text}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(\ell_{ii})}{\sum_{j=1}^{N}\exp(\ell_{ji})}.
$$

总损失取两者平均：

$$
\mathcal{L}_{\text{CLIP}} = \frac{1}{2}\left(\mathcal{L}_{\text{image}} + \mathcal{L}_{\text{text}}\right).
$$

这正是 InfoNCE 的结构：给定一个"上下文"（锚点），把正样本与一批负样本区分开——只不过这里上下文与正样本互为对方，形成了 $N$ 对互为正负样本的结构。这也是为什么它天然需要**大批量**：批内的负样本越多，对比信号越强。

### 代码

从实现上看，你拿到的是一张 $N\times d$ 的图像编码矩阵和一张 $N\times d$ 的文本编码矩阵；归一化之后做矩阵乘法、乘上温度（对数尺度），再算交叉熵。本质上就是一个把样例排列成 $N\times N$ 矩阵形式的多分类问题：

```python
def clip_loss(image_embeddings, text_embeddings, logit_scale, labels=None):
    # image_embeddings / text_embeddings：形状 (N, d)，先做 L2 归一化
    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
    text_embeddings  = text_embeddings  / text_embeddings.norm(dim=-1, keepdim=True)

    # logits[i, j] = <image_i, text_j> * exp(logit_scale)，温度是可学习的
    logits = logit_scale * image_embeddings @ text_embeddings.t()

    if labels is None:
        # 对角线上的 (image_i, text_i) 才是正样本
        labels = torch.arange(len(logits))

    loss_image = F.cross_entropy(logits, labels)        # 图像侧：N 分类
    loss_text  = F.cross_entropy(logits.t(), labels)    # 文本侧：N 分类（转置）
    return (loss_image + loss_text) / 2
```

![CLIP 的算法图：批量编码图像与文本，构造 N×N 的相似度矩阵，对角线为正样本](/lectures/17/clip.png)

![CLIP 的 PyTorch 实现：归一化 → 温度缩放的点积 → 两个方向的交叉熵](/lectures/17/clip-code.png)

## 数据从哪来

CLIP 论文里关于数据的细节其实不多。大致流程是：**拿一堆查询词（queries），在网上搜索，挖掘出一大堆（图像，文本）配对**。幻灯片上给出的数字是：大约搜索了 **50 万个查询词**，平均每个查询收集约 2 万个（图，文）对，最终形成 **4 亿个图文对**的数据集。

值得注意的几点：

- **数据集没有公开**；
- 但有一个复现工作叫 **OpenCLIP**，它复现并进一步扩展了 CLIP 的想法；
- 配套公开了一个数据集 **LAION-5B**——50 亿张带文字描述的图像，经历了一系列处理；
- 有意思的是，**OpenCLIP 自己就用 CLIP 做了数据过滤**，然后再训练 OpenCLIP——也就是说这里面存在一点"自举（bootstrapping）"的味道。

不过无论如何，OpenCLIP 是一个你能**指着数据集和代码**去复现的模型，这是它很大的价值。

## 数据处理：固定尺寸的预处理

还有一个细节。互联网上的图像分辨率千奇百怪：有的又长又瘦，有的又高又宽（或者说反过来的那种），总之是**任意尺寸（arbitrary W×H）**。而神经网络有个"毛病"：它们不喜欢动态的东西，想要**固定大小**。所以 CLIP 用了有点"启发式"的预处理：

1. 先把图像**缩放**（resize，用双三次插值 bicubic interpolation），让**较短的那条边**变成目标尺寸——336 像素（也可以是 224）；
2. 再**中心裁剪**（center crop）成 336×336 的正方形。

这显然是为了方便。后面我们会看到，其实有比这好得多的做法；但当时这样做是出于**省事（expediency）**，而且那时 CLIP 的作者们满脑子还是 ImageNet 分类——ImageNet 里物体通常在画面中央，裁掉一些背景无关紧要。

## 视觉编码器：Vision Transformer（ViT）

CLIP 论文里其实实验了很多种视觉编码器：ResNet 和各种**视觉 Transformer（Vision Transformer，ViT）**——ViT 当时才刚刚问世。结果发现 **ViT 表现最好**。所以当人们说"CLIP"时，通常默认指的是 **ViT 版**的 CLIP。

ViT 大致做这样的事：

1. 把图像**切分成一个个 patch**（图像块）。原始 ViT 论文用的是 16×16，CLIP 用的是 **14×14**；
2. 每个 patch 展开成一个向量——从某种意义上说，**每个 patch 就是视觉 Transformer 的一个 token**；
3. 像训练语言模型那样，**加上位置嵌入（positional embedding）**；
4. 送进一个**标准的 Transformer 编码器**。

![Vision Transformer：图像 → 切分为 patch → 加位置嵌入 → 标准 Transformer 编码器](/lectures/17/vit.png)

Transformer 编码器输出的是一串向量。如果要拿来做分类，通常的做法是把这些向量**平均**成一个向量 $u$。但 CLIP 论文做了一个略有不同的操作——**attention pooling（注意力池化）**：先用所有激活的**全局平均**得到一个向量，然后**再把这个向量作为 query，对每个位置的 key 和 value 再做一轮注意力**，得到另一个向量。这个向量比"直接平均"更有信息量——平均是所有 patch 等权，而 attention pooling 让模型自己决定哪些位置更重要。

用公式写清楚。设 Transformer 输出的位置向量为 $h_1, \dots, h_L$，先算全局平均作为查询：

$$
u = \frac{1}{L}\sum_{\ell=1}^{L} h_\ell,
$$

再用 $u$ 与每个位置做一轮缩放点积注意力（$W_q, W_k, W_v$ 为投影矩阵）：

$$
\hat{u} = \sum_{\ell=1}^{L} \mathrm{softmax}_\ell\!\left(\frac{\left(W_q u\right)^{\top}\left(W_k h_\ell\right)}{\sqrt{d}}\right) W_v h_\ell.
$$

这个 $\hat{u}$ 就是整张图像最终的表示。

### 最佳配置：ViT-L/14@336px

CLIP 论文里最好的模型是 **ViT-L/14@336px**。翻译一下这个代号：

- **L** = Large，大的 ViT——大约有 24 层左右（讲者自己也不太确定这个数）；
- **14** = 使用 **14×14 的 patch**；
- 每个 patch 是 RGB 三通道的；
- **336px** = 训练时使用 336×336 分辨率。

有个小细节：他们其实**先在较低分辨率上训练**，训练的后半段才切换到高分辨率——因为高分辨率图像计算更慢，先用低分辨率训练快一些，最后再"上强度"。

> **课堂问答：图像的位置嵌入需要更复杂吗？**
>
> （同学：对文本来说位置嵌入是一维的，但图像有二维结构，位置嵌入会不会应该更复杂？）
>
> **Tatsu**：问的是能不能在位置嵌入上做更聪明的事——我们这里用的就是简单的线性 0 到 9 这样的编号。这篇论文里他们其实尝试过某种 2D 版本的位置嵌入，结果发现**差别不大**。不过我总是带着"这些结论是在分类任务上得出的"这个前提去看待结果——对分类来说，位置嵌入怎么设计可能确实无所谓。后面我们会看到真正**考虑空间结构**的更花哨的位置嵌入（比如 Qwen2-VL 的 MRoPE），那才是体现空间信息的地方。

## 文本编码器

文本编码器是一个标准 Transformer——具体说是 **GPT-2 风格的 Transformer**（因为 CLIP 来自开发 GPT-2 的同一批人），约 63M 参数、12 层。为了从一整个序列中取出**单个向量**，做法是：在序列前面加一个 **`[BOS]`**，后面加一个 **`[EOS]`**，然后取 **`[EOS]` 在最高层的激活**作为整段序列的表示。

## 训练过程

现在你有了视觉编码器和文本编码器，训练就顺理成章了：挑一个 batch，编码所有文本，编码所有图像，形成上面说的两个 $N$ 分类交叉熵损失，然后反向传播。

## 头条结果：零样本 ImageNet 超越 ResNet-50

2021 年让很多人对 CLIP 兴奋起来的**头条结果**是：在 **ImageNet 基准**上，**零样本（zero-shot）的 CLIP 击败了在 120 万张 ImageNet 图像上训练出来的 ResNet**。

想想这 120 万张 ImageNet 图像意味着什么：那是 Amazon Mechanical Turk 工人无数小时的标注时间。而 CLIP 训练用的是更"有机"的、来自网络的数据——当然，网络数据的产生也凝聚了很多人的劳动，但**如果你能利用已经存在的数据，就可以直接零样本**地工作。

零样本预测的做法（幻灯片上其实有）：拿一张图像，再拿一堆文本标签，用图像嵌入与各个标签嵌入做点积，**看哪个得分最高**，那就是预测结果。完全不需要在 ImageNet 上做任何训练。

> **课堂问答：这种"弱监督"会不会把模型搞糊涂？**
>
> （同学：如果一张图像是狗，但训练数据里其它很多 caption 也提到狗，会不会把模型搞糊涂？）
>
> **Tatsu**：问的是这种带噪的监督信号会不会混淆模型。总体上这个流程是**很嘈杂**（noisy）的。就算出现另一只狗也没关系——因为**平均而言**，其它 caption 里不会总是出现狗，有时是苹果、有时是猫。而且要知道，这些图像基本上是从网页上抓来的——图像配着旁边的文字，或者 `alt` 属性里的文字——所以**极其嘈杂**。有研究专门分析过：图像的文字描述并不一定逐字描述图像里有什么——如果图里是只狗，你根本不需要写下"狗"这个字。所以整个过程非常吵，它居然能工作，多少有点出人意料。这背后需要**大量的数据过滤（data filtering）**——如果你直接拿网上任意的图文配对这种原始数据，很可能会太吵而无法训练。

## 消融：排序目标 vs 直接预测文本

还有一个与本讲结尾呼应的要点。CLIP 作者还尝试了**另一种目标**：不搞这种"排序"式的对比学习，而是**直接从图像预测文本**——把目标设成"根据图像预测文字"，可以是**词袋**（bag of words）式的预测，也可以是**语言模型**式的逐 token 预测。

结果多少有点出人意料：**使用更强的模型（语言模型式的逐 token 预测）效果反而更差、至少是效率更低**，还不如词袋预测，更不如 CLIP 的排序式目标。这张效率图展示了不同方法在相同算力下的 ImageNet 零样本准确率——CLIP 式的排序方法在相同的算力预算下走得最远。

![消融：从图像直接预测文本（词袋或语言模型）远不如 CLIP 式的排序目标算力高效](/lectures/17/clip-efficiency.png)

这说明：对"获得图像的粗略表示"这个目标来说，**精确建模 caption 的 token 序列并不重要**——重要的是把图像与文本的**语义**对齐起来。这个点 Tatsu 说后面会再回来（确实，在 Chameleon 部分我们会看到为追求精确生成离散 token 所付出的代价）。

## 小结：从 CLIP 中学到了什么

到目前为止的要点：

- 用 CLIP 编码的图像**捕捉了图像的语义（semantics）**，因为它与文本配对训练，而文本通常谈论的就是语义；
- 必须注意，**这些设计决策都是基于图像分类任务**的——所以它不是特别"细粒度（fine-grained）"的表示；
- 但它仍然可以作为后面一切工作的**稳健起点**；
- 一个技术上的缺点：CLIP **需要很大的 batch size**（比如 3 万）；batch size 为 1 显然不行，甚至 10 也不行；而且 **softmax 要在整个 batch 上操作**，很难分解（decomposable）成各个样本独立计算。相比之下，语言模型训练里一个 batch 的各条序列是完全并行的，只需要在最后做一次聚合。

这最后一个缺点，正是下一节的 **SigLIP** 要解决的问题。

<!-- lecture-nav -->

**← 上一节**：[引言：从语言模型到全模态模型]（01-introduction-and-omni-models.md）　**→ 下一节**：[SigLIP：用二分类损失训练图像编码器](03-siglip.md)
