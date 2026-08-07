---
title: "07 · 流水线并行与课程小结"
lecture: 7
---

# 流水线并行与课程小结

最后很快地过一遍**流水线并行（pipeline parallelism）**。

## 流水线并行：切层

思路：这次**沿深度方向切网络**——每个 rank 拿到**一部分层**，每层内部的维度是完整的；数据（激活）被传来传去。

![](/lectures/07/pipeline-parallelism.png)

进入主函数后的设置：

```python
def pipeline_parallelism_main(rank: int, world_size: int, data: tensor, num_layers: int, num_micro_batches: int):
    setup(rank, world_size)

    # 使用全部数据
    data = data.to(cuda_if_available(rank))
    batch_size = data.size(0)  # 128
    num_dim = data.size(1)     # 1024

    # 切分层：每 rank 负责 num_layers / world_size 层
    local_num_layers = int_divide(num_layers, world_size)  # 4 / 2 = 2

    # 每个 rank 只拿一部分层，但每层都是完整的 num_dim x num_dim
    local_params = [get_init_params(num_dim, num_dim, rank) for layer in range(local_num_layers)]

    # 前向：把 batch 拆成 micro-batch 以减小 pipeline bubble
    micro_batch_size = int_divide(batch_size, num_micro_batches)  # 128 / 4 = 32
    if rank == 0:
        # rank 0 持有数据，把它 chunk 成 num_micro_batches 份
        micro_batches = data.chunk(chunks=num_micro_batches, dim=0)
    else:
        # 其他 rank 为激活分配内存
        micro_batches = [torch.empty(micro_batch_size, num_dim, device=cuda_if_available(rank))
                         for _ in range(num_micro_batches)]

    for x in micro_batches:
        # 从上一个 rank 接收激活
        if rank - 1 >= 0:
            dist.recv(tensor=x, src=rank - 1)

        # 计算分配给本 rank 的层
        for param in local_params:
            x = x @ param
            x = F.gelu(x)

        # 发送给下一个 rank
        if rank + 1 < world_size:
            print(f"[pipeline_parallelism] Rank {rank}: sending {summarize_tensor(x)} "
                  f"to rank {rank + 1}", flush=True)
            dist.send(tensor=x, dst=rank + 1)
    ...
```

本地层数 `local_num_layers` 是本 rank 要处理的层数；`local_params` 里只有这么多层（每层是完整的 $1024 \times 1024$）。这里用到了前面没提过的 **send / recv** 这两个**点对点（point-to-point）**操作，它们的语义很直白：`dist.recv(tensor=x, src=rank-1)` 说"我要从 rank-1 接收这个张量"，`dist.send(tensor=x, dst=rank+1)` 说"我要把张量 x 发给 rank+1"。

### 为什么要 micro-batch？pipeline bubble 的推导

你可能注意到我在切完层之后又把 batch 拆成了 **micro-batch（微批次）**——为什么？因为流水线并行有一个著名的低效来源。设想不用 micro-batch：拿着一个 batch 从 rank 0 出发，算完它的层、发给 rank 1，rank 1 算完、再发给 rank 2……在这个"接力"过程中，**大部分时间里大部分 GPU 都在空等**（等别人把数据传过来），这种空闲就叫 **pipeline bubble（流水线气泡）**。batch 越大，接力越久，气泡越严重。

用 micro-batch 把它拆小：每个 micro-batch 都是"快速算完 → 立刻传给下一卡"，于是各个 rank 能更快地并行起来——**rank 0 在算 micro-batch 2 的时候，rank 1 正在算 micro-batch 1**，流水线进入稳态。气泡因此大幅缩小。

来做一个完整的定量分析。设流水线有 $N$ 个阶段（stage，即 rank 数）、$K$ 个 micro-batch，每个 micro-batch 经过一个阶段耗时 1 个单位时间（无通信重叠的朴素调度）。前向的最优排程如下：第一个 micro-batch 要走完 $N$ 个阶段（填充阶段，fill），之后每个新的 micro-batch 只需追加 1 个单位（稳态），所以**总时间**为

$$T_{\text{total}} = N + K - 1$$

而每个 GPU 真正做有用计算的时间是 $K$（它要处理 $K$ 个 micro-batch）。于是每个 GPU 的空闲（气泡）占比为

$$\text{bubble 占比} = \frac{T_{\text{total}} - K}{T_{\text{total}}} = \frac{N - 1}{N + K - 1}$$

从公式读出的结论很直观：

- 若 $K = 1$（不拆 micro-batch），气泡占比 $\frac{N-1}{N}$，**几乎全部时间都在空等**——极其糟糕；
- 若 $K \gg N$（micro-batch 很多），气泡占比 $\to 0$，流水线被充分填满；
- 反传（backward）也是同样的接力结构，气泡同理存在（作业里会再见到）。

回到代码：我们的例子是 world_size = 2、num_layers = 4、num_micro_batches = 4，于是 $N = 2$、$K = 4$，气泡占比 $\frac{2-1}{2+4-1} = \frac{1}{5} = 20\%$——micro-batch 把它从 $\frac{1}{2}$ 压到了 $1/5$。

### 没被处理的事：重叠通信与计算

这个朴素版本还有一个非常关键的缺口：**通信与计算的重叠（overlapping communication and computation）**。在流水线并行里，你在计算的同时应该已经在接收/发送数据——比如给 send/recv 加上异步（async）标志，让它变成"发起传输、立刻回来"，再配合额外的管理逻辑确保正确性。这样**计算和通信并行进行**，总耗时进一步下降。本讲没有实现这一点，但它是流水线并行真正高效的关键；下周 Tatsu 会深入讲。

## 这一讲还缺什么

收尾之前，点名几件本讲"没讲到"的事：

- **通信/计算重叠**：不只是流水线并行需要它。数据并行里我们也是先完整反向、最后才做一堆 all-reduce；如果够聪明，反向传播中**每个参数的梯度一算完就立刻开始发**——这正是你作业二（assignment 2）会探索的东西；
- **更一般的模型**：我们只讲了 MLP。我觉得 MLP 已经涵盖了理解这些基础知识所需的大部分内容；更大的模型（带注意力等）只是**需要更多的簿记（bookkeeping）**，核心机制是同一套；
- **其他并行形式**：**序列并行（sequence parallelism）**——把整条序列切碎，从而并行化注意力；**专家并行（expert parallelism）**——并行化 MoE 里的专家，这就是前面提到 all-to-all 用武之地的地方；以及**各种并行策略的组合**（这些在作业里都会出现）。

### 选择哪种并行，取决于硬件

一个值得强调的点：**选哪种并行，强烈依赖硬件**。

- **张量并行**通信量巨大（每层都要搬完整激活），所以**只在一个节点内、NVLink/NVSwitch 域内**做；
- **流水线并行**能**容忍慢得多的互联**——它只在阶段边界通信。有些去中心化训练（decentralized training）的工作甚至用流水线并行，因为参与方可能隔着半个地球；但你绝不会想在这种环境下做张量并行；
- 于是实际系统常常是**分层组合**：节点内做张量并行，节点间做数据并行（或 FSDP），必要时再上流水线并行；
- 数据并行还有自己的天花板：你可以把 batch 撑得很大，但一旦超过**临界 batch size（critical batch size）**，batch 再大也不会带来更多收益——那就只是在浪费算力，不如转用张量并行。这些考量我们在课程后面还会展开。

### PyTorch 与 JAX/TPU 的哲学对比

最后提一句之前冒出来的 **TPU**。我们是**刻意**用 PyTorch、而且刻意只用最原始（primitive）的 collective 操作的——这样你能**机械地**看见每一步到底发生了什么。另一条路线（尤其 JAX + TPU 生态）是：你只需要定义**模型**和**分片策略**，然后让编译器替你决定需要哪些通信操作——你只需说"这份数据需要在这、这、这"，编译器用"魔法"替你编排（比如 Levanter 这类框架）。这当然很吸引人，但它显然会**夺走从零搭建的乐趣**。

## 课程小结

- **并行化有很多切法**：切**数据**（数据并行）、切**张量/专家**（宽度维）、切**流水线**（深度维）、切**序列**（长度维）；
- 本讲看了**数据并行**，且只做了 DDP——下次讲 **FSDP 与 ZeRO**；
- **张量并行**需要非常快的互联（NVLink）；
- **流水线并行**对互联要求低，但要下功夫把 **pipeline bubble** 压下去；
- 在高层次上，这个模式反复出现：

> **你可以重算（re-compute），或存在内存里（store in memory），或存在另一块 GPU 上再通信（store on another GPU's memory and communicate）。**

还记得激活检查点（activation checkpointing）吗？那就是"重算 vs 存内存"的权衡；而这一讲是它的自然延伸——数据并行在某种意义上在做**冗余计算**：每个 rank 都更新并保存整套参数，代价是白做，换来的是**不用搬运优化器状态**。

最后一点：硬件确实在变快，但从某种意义上说，**我们永远会想要更大的模型**。所以这种**层级结构**——单 GPU 内用 L1/shared/HBM，单节点内用 NVLink，跨节点用 InfiniBand/Ethernet——会一直在那里。

好了，今天就到这里。下周三，Tatsu 会对并行做一次更深入的剖析（FSDP/ZeRO、更多并行策略的组合、真实大规模训练中的细节）。

<!-- lecture-nav -->

**→ 下一讲**:[08 Parallelism](../08/)
