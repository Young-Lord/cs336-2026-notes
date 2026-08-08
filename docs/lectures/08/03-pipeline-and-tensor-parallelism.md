---
title: "08 · 模型并行：流水线并行与张量并行"
lecture: 8
---

# 模型并行：流水线并行与张量并行

上一节结尾我们说到：想真正往下压显存、又要不消耗 batch size，需要**模型并行**——切模型本身。这一节看两种"切"法：沿深度切（流水线并行）与沿宽度切（张量并行）。先讲概念上最简单的切层方案。

## 层切（layer-wise parallel）：直觉上很自然的失败方案

我最初学并行的时候想的就是："哦，这很简单——把层切开、放到不同的 GPU 上。"非常直觉。前向时把激活往前传，反向时把部分梯度往回传。但如果你真的这么干，会得到一张**非常令人沮丧的图**。

假设有 4 块加速器，每块负责模型的四分之一。横轴是时间、每行是一块加速器：最下面的加速器 0 负责第一层，它先做一点计算、然后**停下来**，把结果交给加速器 1；加速器 1 做完又交给 2……于是**任意时刻只有一块 GPU 在干活**，大部分 GPU 大部分时间都在空转。反向也同理：一次只有一块 GPU 处理反向。这种空闲就叫**气泡（bubble）**——流水线里什么都不做的那部分时间。利用率低到可怕。

## 流水线并行：用 micro-batch 填满流水线

解决办法就是流水线化（pipelining）——本质上是**批处理（batching）**：把大 batch 拆成 **micro-batch**（微批次）。图里是 4 个 micro-batch：处理完第一个 micro-batch 立刻开始第二个，同时把第一个交给下一层——像接力一样把元素往上送、再往下送。

![](/lectures/08/slide-34.png)

### bubble 占比的完整推导

设流水线有 $n_{\text{stages}}$ 个阶段（stage，即 GPU 数）、$n_{\text{micro}}$ 个 micro-batch，每个 micro-batch 经过一个阶段耗时 1 个单位时间。前向的调度结构是：第一个 micro-batch 要走完所有 $n_{\text{stages}}$ 个阶段（这叫**填充阶段（fill）**），之后每个新 micro-batch 进入流水线只需追加 1 个单位，最后一个 micro-batch 还要再走 $n_{\text{stages}} - 1$ 步才能出流水线。于是完成全部前向的**总时间**为

$$T_{\text{total}} = n_{\text{stages}} + n_{\text{micro}} - 1$$

每块 GPU 真正在做有用计算的时间是 $n_{\text{micro}}$（它要处理这么多 micro-batch；把反向也算上，分子分母同乘 2，结论不变）。因此每块 GPU 的**气泡时间与有用计算时间之比**为

$$\frac{T_{\text{total}} - n_{\text{micro}}}{n_{\text{micro}}} = \frac{n_{\text{stages}} - 1}{n_{\text{micro}}}$$

这正是幻灯片上的公式。从这个式子读出两个结论：

- **气泡随 micro-batch 数线性衰减**——$n_{\text{micro}}$ 越大，气泡占比越低；
- 想让气泡趋近于零，需要**很大的 batch size**。这就是为什么我在上一节说"batch size 是宝贵资源"——你可以把它花在另一个地方：灌进流水线，减少空闲时间。

### 为什么流水线并行虽然"烂"但还是要用？

流水线看起来挺糟糕，那为什么我们要用它？有两条理由：

1. **省显存（对比 DDP）**：我们把层切开放在不同 GPU 上，每块 GPU 只需持有部分层；而且可以和数据并行组合，两者同时做；
2. **通信特性极好（对比 FSDP）**：它的通信量**只依赖于激活**——大小是 $b \times s \times h$（batch × 序列长 × 隐藏维），而且通信是**点对点**（point-to-point）的：一个 stage 只和它的邻居 stage 通信，不是 all-to-all。只要能安排好顺序，它是对网络利用最有效的方式之一。

于是实践中有一条准则：**流水线并行放在最慢的网络链路上**。如果你有多个数据中心、多个 pod，它们之间的连接很慢——就用流水线并行去跨它们，因为它是能拿出来的**通信效率最高**的并行方式。

### 性能高度依赖 batch size

流水线性能对 batch size 极其敏感。Nvidia 的 **Megatron 论文**里有大量漂亮的参数扫描：当你用大 batch size 配大流水线深度时，利用率可以接近"不做流水线"的水平；但如果 batch 不够大，流水线会**迅速恶化**利用率。batch size 是隐藏气泡的关键。

> **课堂问答：为什么流水线的通信特性比 FSDP 好？**
>
> （同学：你说流水线相比 FSDP 通信特性更好，具体机制是什么？）
>
> **Tatsu**：因为**通信的数据量更小**。流水线要传的是 $b \times s \times h$ 的激活，这几乎总是远小于"把一整块参数矩阵传来传去"的量。

### 更聪明的调度：交错前向/反向

人们还尝试了很多更巧妙的办法压缩气泡大小。你可以做聪明的**调度（scheduling）**：在不同 micro-batch 之间交错安排不同层的前向/反向元素——把某些前向计算插到另一些反向计算之间。这个图来自 **DeepSeek 论文**：通过精心安排"什么时候做前向、什么时候做反向"，可以进一步缩小气泡。

### Zero-bubble：把反向拆成两半

比调度更聪明的是所谓 **zero-bubble（零气泡）流水线**。它的洞察不在于调度，而在于**重新思考反向传播的结构**。反向时在每个计算图节点上其实要做两件事：

1. **传播激活梯度（propagating partials）**：把偏导数沿计算图继续往回传（记号里的 $B$，即 backprop）——这是"下一步能干活"的前提，**必须尽快做**；
2. **计算权重梯度（computing weight gradients）**（记号里的 $W$）：算当前权重的导数——这在计算图上是个"叶节点"，**什么时候做都行**。

于是做法是：**先把 $B$（传播）尽快做完**，把 $W$**（权重梯度）推迟**到有空档的时间片再做。这样几乎能把流水线的气泡**完全填满**。zero-bubble 实现起来比一般流水线复杂得多，但几乎能彻底解决流水线利用率问题（取决于具体工作负载）。

## 沿宽度切：张量并行

流水线是沿**深度**（depth）切。那能不能沿**宽度**（width）切、获得更好的利用率？答案就是**张量并行（tensor parallel，TP）**。

出发点是矩阵乘法的简单观察：一个矩阵乘法可以**分解成子矩阵乘法、再相加部分和**。设

$$Y = XW = X \begin{bmatrix} W_1 & W_2 \end{bmatrix} = \begin{bmatrix} XW_1 & XW_2 \end{bmatrix}$$

（按列切 $W$），或者

$$Y = XW = \begin{bmatrix} X_1 & X_2 \end{bmatrix} \begin{bmatrix} W_1 \\ W_2 \end{bmatrix} = X_1 W_1 + X_2 W_2$$

（按行切 $W$）。关键是**把部分和加起来**这一步——在分布式中它就是一个通信原语。

### f 和 g：TP 的前向/反向对偶

在 Megatron 风格的张量并行里，每个 transformer 块里插两个"通信点"，习惯上叫 $f$ 和 $g$：前向时 $f$ 是恒等（identity）、$g$ 是 **all-reduce**；反向时两者对调——$f$ 是 all-reduce、$g$ 是恒等。

![](/lectures/08/slide-40.png)

直觉：前向时，各 GPU 用自己手里的子矩阵算出一个部分和，必须 **all-reduce** 把它们加起来才能得到完整输出；反向时，梯度要从输出一侧传回各 GPU 的输入一侧，所以**输入侧的通信点（$f$）变成 all-reduce**。这就是前向 all-reduce 与反向 reduce-scatter（或 all-reduce）的**对偶性**——上一讲讲张量并行时 Percy 已经铺垫过这个思想。

### 行切 vs 列切

在 transformer 块里，张量并行怎么分配？

- **按列切（column-wise）**：QKV 投影、up-projection（MLP 第一个线性层）——输出维被切开；
- **按行切（row-wise）**：注意力输出投影（attention output）、down-projection（MLP 第二个线性层）——输入维被切开；
- **复制（replicated）**：LayerNorm、router（MoE 的路由器）等小而关键的部分直接复制。

这种"列切 + 行切交替、中间夹 all-reduce"的安排，就是 Megatron 的经典做法：QKV 用列切、注意力输出用行切，于是注意力块内只需要**两次** all-reduce；MLP 的 up-projection 列切、down-projection 行切，又是两次。合起来每个 transformer 块**四次** all-reduce。

### 什么时候用张量并行？

在 GPU 上，张量并行只用在**节点内**——最多 8 块 GPU（NVLink 域内）。原因后面马上算：它的通信量很大，只有 NVLink 这种高速互联才扛得住。

## 张量并行 vs 流水线并行：通信量对比

两者的优劣怎么量化？把通信量摆出来：

- **流水线并行**：每 micro-batch 是**点对点**通信，量级为 $b \times s \times h$（激活本身的大小）；
- **张量并行**：每个 transformer 块要做**四次 all-reduce**（两次注意力、两次 MLP），而一次 all-reduce 的通信量约等于 $2 \times$（数据量）（还记得"all-reduce 约等于把数据搬两遍"）。于是每层每 micro-batch 的通信量为

$$\text{TP 每层通信量} = 8 b s h \cdot \frac{n_{\text{devices}} - 1}{n_{\text{devices}}}$$

其中 $8bsh = 4 \times 2bsh$：4 次 all-reduce × 每次 $2bsh$（$bsh$ 是激活大小，all-reduce 搬两遍所以乘 2）；因子 $\frac{n_{\text{devices}} - 1}{n_{\text{devices}}}$ 是 all-reduce 的环形拓扑系数（回忆第七讲的推导，大 $n$ 时趋于 1）。

于是张量并行的优缺点一目了然：

- **优点**：没有气泡——只要网络够快，谁都不用等谁；**复杂度低**——容易"包一层 wrapper"就把模型改成 TP，不需要动基础设施；**不需要大 batch** 就能工作得很好；
- **缺点**：通信量比流水线**大得多**（$8bsh$ 对 $bsh$，差 8 倍，而且是 all-reduce 而非点对点）。

所以规则是：**只要你有低延迟、高带宽的互联，就用张量并行**；通信很贵的慢链路上，流水线是更优解。

<!-- lecture-nav -->

**← 上一节**：[数据并行与 ZeRO：把显存摊到所有 GPU](02-data-parallelism-and-zero.md)　**→ 下一节**：[激活显存、序列并行与专家并行](04-activation-sequence-expert.md)
