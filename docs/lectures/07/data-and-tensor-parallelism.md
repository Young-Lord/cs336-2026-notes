---
title: "07 · 数据并行（DDP）与张量并行"
lecture: 7
---

# 数据并行（DDP）与张量并行

现在真正开始思考**怎么训练模型**。我们只讲一种非常朴素的（bare-bones）实现——在**深度 MLP** 上训练。顺带一提，"多层的多层感知机"有点冗余，但没关系；关键点是：**MLP 正是 Transformer 里真正的计算瓶颈**（FFN 层就是 MLP），所以拿它做演示很有代表性。

三种并行策略的"切法"总览（这是张示意性质的图，别抠得太细，重点是它帮你建立"我们在切什么"的心智模型）：

- **数据并行（data parallelism）**：把**数据**切成几块，每块 GPU 负责一块，照常做完整的模型训练，然后**同步**；
- **张量并行（tensor parallelism）**：切的是**每一层**（沿宽度维）；
- **流水线并行（pipeline parallelism）**：切的是**层**（沿深度维）。

我们先讲数据并行。

## 数据并行：切 batch

策略一句话：**每个 rank 拿数据的切片**。

![](/lectures/07/data-parallelism.png)

先生成样本数据：batch size 为 128、维度 1024，也就是一个 $128 \times 1024$ 的数据矩阵。进入数据并行的主函数后：

```python
def data_parallelism_main(rank: int, world_size: int, data: tensor, num_layers: int, num_steps: int):
    setup(rank, world_size)

    # 拿到本 rank 的数据切片（实践中应该让每个 rank 只加载自己的数据）
    # 示意：把数据矩阵按行切分成 B0/B1/B2/B3 四份
    batch_size = data.size(0)        # 128
    num_dim = data.size(1)           # 1024
    local_batch_size = int_divide(batch_size, world_size)  # 128 / 4 = 32
    start_index = rank * local_batch_size
    end_index = start_index + local_batch_size
    data = data[start_index:end_index].to(cuda_if_available(rank))

    # 创建 MLP 参数 params[0], ..., params[num_layers-1]（每个 rank 都持有全部参数）
    params = [get_init_params(num_dim, num_dim, rank) for layer in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)  # 每个 rank 都有自己的优化器状态
    ...
```

我们要把数据矩阵的**行**切成 world size（这里为 4）份，每个 rank 拿到一份。我定义**本地 batch size（local batch size）**= batch size / world size：每块 GPU 只看到 32 个数据点。`data[start:end]` 取到该 rank 负责的那片数据，并放到它的 GPU 上。此时每块 GPU 持有**互不相同**的数据张量。（实践中每个 rank 应该自己加载自己的数据，而不是这样统一切分——这只是为了讲解方便。）

然后实例化模型：假设有 num_layers 层，每层一个 $1024 \times 1024$ 的随机参数矩阵，全部喂给 `AdamW` 优化器。接着是训练循环：

```python
    for step in range(num_steps):
        # 前向：在本地的数据切片上前向
        x = data
        for param in params:
            x = x @ param
            x = F.gelu(x)
        loss = x.square().mean()  # 损失 = 平均平方大小

        # 反向
        loss.backward()

        # 同步梯度（与标准训练的唯一区别！）
        for param in params:
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)

        # 更新参数
        optimizer.step()
        ...
```

前向：拿到数据（注意这里不是全部数据，比如 rank 2 只有 B2 那份），逐层做 `x @ W` 和 GeLU；然后 `loss.backward()`。到这一步，如果按普通训练，事情就该结束了——但别忘了，**每个 rank 的数据不同，梯度因此也不同**。所以这里插入让数据并行成立的**关键一步**，也是**数据并行与普通训练之间的唯一区别**：

> 对每个参数做一次 `dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG)`——把各 rank 的梯度**求平均**并同步。

all-reduce 之后，每个 rank 的梯度都**完全相同**了，然后照常 `optimizer.step()` 更新参数。你看这多优雅：它基本就是普通的训练，只是用**本地 batch** 前向/反向，然后在 backward 之后插上一行——把梯度 all-reduce 求平均。训练过程中，每个 rank 表现得**仿佛它看到了全部数据**，但实际只处理了其中一份。

### 课堂问答：关于 DDP 的实现细节

**问答：batch size 是不是必须大于 1？**

（同学：只有 batch size 大于 1 才能这么做吗？）

**Percy**：是的，你的 batch size 至少得是 world size，这件事才有意义；通常还应该比 world size 大不少。

**问答：batch size 必须是 world size 的倍数吗？**

（同学：batch size 应该是 world size 的整数倍吧？）

**Percy**：是的话最好。如果不是，你可以补零（pad）之类的处理——办法是有的，但大家的日子都更好过一些。

**问答：Transformer 里它会长什么样？**

（同学：对 Transformer 来说这又是什么样？）

**Percy**：基本一模一样。DDP 有个很棒的性质——**非常模块化**：你照常做前向、照常做反向，DDP 只是在这里把梯度求平均。它根本不关心你的前向长什么样。

## 数据并行：通信与显存分析

把 DDP 收束一下：**损失在各 rank 上是不同的**（它们各自在本地数据上算）；**梯度初始也不同**，但 all-reduce 之后变得一致；因此**参数在所有 rank 上始终保持一致**——这正是"仿佛看到了全部数据"的数学含义。

代价有两个方面。**通信**：每一步训练要做**一次梯度 all-reduce**，梯度的尺寸就是参数个数 $|\theta|$。回忆上一节的 ring 推导，每 rank 的通信量 $\approx 2|\theta|$ 字节——也就是说，**每一步都要把"两倍的模型"搬过网络**。模型越大，这一步越贵。**显存**：每个 rank 持有**完整的参数 + 梯度 + 优化器状态**（AdamW 还要额外维护动量和二阶矩），即整个集群里有 $N$ 份模型副本。放得下时这不是问题，放不下时就是问题。

**问答：DDP 一个明显的"缺陷"是每卡都要持有完整模型，怎么办？**

这正是下讲（周三）Tatsu 要讲的**更高级的数据并行：FSDP 与 ZeRO**。它们的想法我在前面已经埋了伏笔：这里我们用的是 all-reduce 这种"整体式"操作——简单，但它**要求每块 GPU 把整个模型的参数都放在内存里**。如果模型参数根本放不进显存呢？那就必须更聪明：用 **all-gather**（前向/反向前把参数分片拼回来）与 **reduce-scatter**（反向后把梯度按分片归约、分散存储），这就是 FSDP/ZeRO 的主题——本讲不展开。

## 张量并行：切每层的宽度

**张量并行**的切法换了个方向：不切数据，切**每一层**——每个 rank 拿到**每层的一部分**。一般这意味着要传输**多得多**的数据，我们稍后会讨论。

![](/lectures/07/tensor-parallelism.png)

演示时为了简单，我们让每个 rank 都持有全部数据。数据还是 $128 \times 1024$。现在定义一个**本地维度（local num dim）**：本 rank 只负责维度的一个子集：

```python
def tensor_parallelism_main(rank: int, world_size: int, data: tensor, num_layers: int):
    setup(rank, world_size)

    data = data.to(cuda_if_available(rank))  # 所有 rank 都拿到全部数据（batch_size x num_dim）
    batch_size = data.size(0)  # 128
    num_dim = data.size(1)     # 1024
    local_num_dim = int_divide(num_dim, world_size)  # 把 num_dim 切成 4 份：256

    # 创建模型：每个 rank 只拿到 1/world_size 的参数
    #      |  |  |  |
    #     W0 W1 W2 W3
    #      |  |  |  |
    params = [get_init_params(num_dim, local_num_dim, rank) for layer in range(num_layers)]

    # 前向
    x = data
    for layer in range(num_layers):
        # 计算激活（batch_size x local_num_dim）
        x = x @ params[layer]   # 注意：这里只用到了参数矩阵的一个切片
        x = F.gelu(x)

        # 为激活分配内存（world_size x batch_size x local_num_dim）
        activations = [torch.empty(batch_size, local_num_dim, device=cuda_if_available(rank))
                       for _ in range(world_size)]

        # 用 all-gather 交换激活
        dist.all_gather(tensor_list=activations, tensor=x, async_op=False)

        # 拼接成完整的 batch_size x num_dim
        x = torch.cat(activations, dim=1)
    ...
```

每个 rank 的模型仍有全部层，但每层的参数是 $1024 \times 256$（num_dim × local_num_dim）——也就是说对每层的参数矩阵，我们是**沿列方向切开**的（每卡拿一部分列）。这正是所谓的**列张量并行（column tensor parallel）**；你也可以按行切，但我们今天不谈。

前向是怎么跑的？逐层做：`x @ params[layer]`——注意 `params[layer]` 只是参数矩阵的**一个切片**：如果我站在 rank 1 上，我只拿到矩阵的这一小块，但照样能算——因为矩阵乘可以**按列拆开**，每块 GPU 各算各的列，然后拼接。GeLU 是逐元素的，照样可以算。但接下来必须**通信激活**：rank 1 有激活的一部分（对应它那几列），其他 rank 也有各自的部分，我需要把**所有激活都放到所有 rank 上**——这我们认识，正是 all-gather。

代码里的流程是：`x`（形状 $128 \times 256$）是本 rank 算出的激活分片；`activations` 是给每个 rank 分配的输出槽（一个列表，长度 = world size）；`dist.all_gather` 把每个 rank 的 `x` 分别拷进对应槽位；最后 `torch.cat(activations, dim=1)` 沿列维拼回完整的 $128 \times 1024$。**每一层都做一遍这个 all-gather + 拼接**。

注意数据并行和张量并行的一个关键差别：**数据并行不用动模型**（模型被当作一个"黑盒模块"，非常优雅）；而张量并行**必须动模型**——它强烈利用了这样一个事实：矩阵乘法可以拆成一堆更小的矩阵乘法、在不同 rank 上各算各的、再 gather 结果。这是用手搓出来的，稍显笨拙，但机制一目了然。

### 课堂问答：关于张量并行的反向传播

**问答：反向传播时怎么办？**

（同学：那反传的时候呢？）

**Percy**：反传时你手里有激活，需要 **reduce-scatter** 把梯度分到各个 rank 上。所以某种程度上，**all-gather 和 reduce-scatter 是一对对偶操作**：前向时 all-gather，反向时就 reduce-scatter。

**问答：这些 autograd 会自动做吗？**

（同学：调 `loss.backward()` 会自动处理这些吗？）

**Percy**：不会。只调 `loss.backward()` 是不行的，因为里面根本没有任何并行。PyTorch 确实有很多帮你自动做的事情，但在这个例子里我们**管理得相当显式**——所以我们没有写反传部分，真要做的话你得自己调用 reduce-scatter。这是有意为之的设计：这门课是"从零构建语言模型"，让你看见每个零件；在真实工程里你多半不用自己干这个。

## 张量并行：通信与显存分析

张量并行的代价为什么"更大"？设 batch size 为 $B$、宽度为 $D$、层数为 $L$、世界大小为 $N$。

**显存**：每 rank 只持有参数的 $1/N$——这是它的好处，模型参数被均摊了。

**通信**：**每一层**都要做一次 all-gather，交换的是**激活**：每 rank 算出的分片是 $B \times (D/N)$，all-gather 之后每 rank 要持有完整的 $B \times D$。由 ring 分析，每层每 rank 的通信量 $\approx (N-1) \cdot BD/N \approx BD$。于是 **$L$ 层的总通信量 $\approx L \cdot BD$**——它正比于**激活的大小**，而激活是很大的。反传时还要对每层做一次 reduce-scatter（数量级相同，再翻一倍）。

这正是"张量并行需要非常快的互联"的根本原因：通信量和模型本身的计算量同阶（每层都要搬一遍完整激活），只有 NVLink 这种 TB/s 级别的互联才扛得住。因此张量并行一般**只在一个节点内、一个 NVLink/NVSwitch 域内**使用，而不会跨节点、更不会跨洲际——我们下一节讲流水线并行时还会回到这个对比。

<!-- lecture-nav -->

**← 上一节**：[torch.distributed：实现 collectives 与通信带宽]（torch-distributed.md）　**→ 下一节**：[流水线并行与课程小结](pipeline-parallelism.md)
