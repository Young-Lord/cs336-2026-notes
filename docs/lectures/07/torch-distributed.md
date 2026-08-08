---
title: "07 · torch.distributed：实现 collectives 与通信带宽"
lecture: 7
---

# torch.distributed：实现 collectives 与通信带宽

讲完硬件，我们真正开始写代码，把上面这套硬件用起来。**PyTorch** 贴心地提供了 `torch.distributed` 库，它把 collective 操作包装成**干净的接口**，你不需要显式去想 NCCL。

## torch.distributed 概览

- `torch.distributed` 提供 collective 操作的干净接口（例如 `all_gather_into_tensor`、`all_reduce`、`reduce_scatter_tensor`）；
- 它支持**不同的后端**（backend）以适配不同硬件：在 GPU 上用 **nccl** 后端，在 CPU 上用 **gloo** 后端——并行编程可比 GPU 早得多，你在 CPU 上同样可以跑这些 collective；
- 它还支持更高级的模型与算法，比如 **FSDP（FullyShardedDataParallel）**——不过本课程**不用**这些高级封装，因为我们要从零开始、亲手搭建。

## 第一个分布式程序

先看一个叫做 `spawn` 的包装函数。它的作用是"把这个函数复制成 world size 份，每份在一个进程里跑"。正常写法是调用 `torch.multiprocessing.spawn`，但课堂 trace 没法穿过 multiprocessing，所以我写的这个包装在 trace 模式下会**禁用分布式函数**（替换成空操作）、单进程直接运行。直接运行本讲讲义时，走的才是真正的 multiprocessing 分支：

```python
def spawn(func: Callable, world_size: int, *args, **kwargs):
    """启动 world_size 个进程，每个进程调用 func（带 world_size、args、kwargs）。
    注意：如果在 trace 模式（edtrace）下，就直接单进程运行并禁用分布式函数。"""
    if not sys.gettrace():
        # 正常代码路径：multiprocessing
        args = (world_size,) + args + tuple(kwargs.values())
        mp.spawn(func, args=args, nprocs=world_size, join=True)
    else:
        # 被 trace 时：单进程直接运行（分布式函数被替换为空操作）
        with DisableDistributed():
            args = (0, world_size,) + args + tuple(kwargs.values())
            func(*args)
```

于是我们进入这个"本应并发运行在每个进程上"的函数。回忆一下：world size 是进程/设备的数量，rank 从 0 一直取到 world size 减 1；也就是说有 world size 份这样的函数同时在跑。现在我在 rank 0 上。第一件事是 `setup`：

```python
def setup(rank: int, world_size: int):
    """初始化分布式环境（在每个进程开头调用）。"""
    # 指定 master 的位置（rank 0），用于协调；真正的数据不走这里（走 NCCL）
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "15623"

    if torch.cuda.is_available():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
```

注意 `MASTER_ADDR` 和 `MASTER_PORT` 只是元数据与协调用的：GPU 之间的实际数据并不从这里走，而是走 NCCL，否则会慢得离谱。有 CUDA 就用 nccl 后端；我在笔记本上跑，所以用 gloo 后端。

然后是 **barrier（屏障）**——一个同步工具：`dist.barrier()` 会**等待所有进程都到达这里**再放行。因为各进程异步运行，你无法控制谁先跑完、谁会交错执行，所以当你想保证"某些代码必须在另一些代码之前执行"时，就插入同步屏障。当然，屏障放多了也有代价——你可能会白等。

### 第一个操作：all-reduce

接下来做一次 all-reduce。为了让例子更有趣，每个 rank 持有一个不同的张量：rank $i$ 持有 $[0,1,2,3] + i$。`dist.all_reduce` 把 `data` **就地**（in-place）修改——它既是输入又是输出：

```python
### All-reduce（dist = torch.distributed）
dist.barrier()  # 等所有进程到达这里（本例是为了 print 不乱序）

data = tensor([0., 1, 2, 3], device=cuda_if_available(rank)) + rank  # 输入 = 输出

print(f"Rank {rank} [before all-reduce]: {data}", flush=True)
dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)  # 就地修改 data
print(f"Rank {rank} [after all-reduce]: {data}", flush=True)
```

跑起来会看到：all-reduce 之前，rank 0 打印 `[0, 1, 2, 3]`、rank 1 打印 `[1, 2, 3, 4]`，以此类推——注意**打印顺序是硬件随机的**（各进程异步），但数据都在；all-reduce（SUM）之后，每个 rank 都得到按列求和并复制的结果，即 $[6, 10, 14, 18]$。

这里的调用方式很简单：传入张量、归约运算（SUM），以及 `async_op=False`（同步执行）。这个调用在 GPU 上会调用 gloo（或 nccl），nccl 后端会**启动 CUDA kernel 做通信**，一切替你包办，最后就地写回 `data`。如果你想要异步，就把 `async_op` 设为 `True`——不过那样会把这一堆 print 的顺序全打乱，所以我故意多放了几个屏障。

### 课堂问答：关于异步与同步

**问答：异步 all-reduce 是怎么工作的？**

（同学：`async_op=True` 的 all-reduce 内部是怎么运作的？）

**Percy**：all-reduce 是个"整体式"操作——你说"去做 all-reduce"，它就启动 kernel、做通信。CUDA 相对于进程本来就是异步的，现在进程之间又异步。`async_op=True` 的意思是：这个调用**立即返回**，你可以先去做别的事（典型的场景是：启动通信之后，顺便去加载下一批数据——这个数据与通信相互独立）；等你想确认通信真的完成了，再调用 `wait()` 或者同步。这就是"**重叠计算与通信**"（overlapping computation and communication）的思路——本讲不展开讲，但它是作业里会探索的东西。

### 第二个操作：reduce-scatter

这次不是就地修改，而是显式的"输入张量 + 输出张量"。输入是 `torch.arange(world_size) + rank`，输出先分配一块空内存；`dist.reduce_scatter_tensor` 执行完之后，**输入不被触碰**，输出里装的是"每个分量在所有 rank 上求和、并落到对应 rank"的结果：

```python
### Reduce-scatter
dist.barrier()

input = torch.arange(world_size, dtype=torch.float32, device=cuda_if_available(rank)) + rank  # 输入
output = torch.empty(1, device=cuda_if_available(rank))  # 分配输出

print(f"Rank {rank} [before reduce-scatter]: input = {input}, output = {output}", flush=True)
dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
print(f"Rank {rank} [after reduce-scatter]: input = {input}, output = {output}", flush=True)
```

也就是：rank 0 的输入是 $[0,1,2,3]$、rank 1 是 $[1,2,3,4]$……reduce-scatter 之后，rank 0 拿到第 0 维的和 $6$，rank 1 拿到第 1 维的和 $10$，rank 2 拿到 $14$，rank 3 拿到 $18$。

### 第三个操作：all-gather

最后做 all-gather。我们把 reduce-scatter 的**输出**当作 all-gather 的**输入**，再分配一个长度为 world size 的输出，调用 `dist.all_gather_into_tensor`：

```python
### All-gather
dist.barrier()

input = output  # 输入 = reduce-scatter 的输出
output = torch.empty(world_size, device=cuda_if_available(rank))  # 分配输出

print(f"Rank {rank} [before all-gather]: input = {input}, output = {output}", flush=True)
dist.all_gather_into_tensor(output_tensor=output, input_tensor=input, async_op=False)
print(f"Rank {rank} [after all-gather]: input = {input}, output = {output}", flush=True)
```

all-gather 之后，每个 rank 上的 output 都是 $[6, 10, 14, 18]$——于是你亲眼看到了一个"**用例子证明**"：

> **all-reduce = reduce-scatter + all-gather。**

收尾时调用 `cleanup()`（`torch.distributed.destroy_process_group()`）释放资源——好习惯。这就是你的第一个 `torch.distributed` 程序。

### 课堂问答：关于同步的细节

**问答：为什么要对 CUDA kernel 做 synchronize？**

（同学：为什么要 `torch.cuda.synchronize()`？）

**Percy**：说到底我们还是在做 CUDA 操作——只是现在是多个进程、每个进程一块 GPU。而 CUDA 操作默认是**异步**的：当你执行到 Python 的下一行时，那个 CUDA 操作可能还没跑完。所以我们总要用 `synchronize` 确保它真的做完了。

**问答：barrier 和 synchronize 的顺序有讲究吗？**

（同学：是不是先做 barrier 再做 synchronize 就行？）

**Percy**：不是的。如果先 barrier：CUDA kernel 可能还在跑，你就已经到 barrier 了；而 barrier 只同步进程，每个进程各自同步自己的 CUDA kernel——于是你们并没有真正同步。如果那些操作都只是"返回了"（没真正完成），barrier 实际上没起到什么作用。

## 基准测试：通信到底有多快

这段会比较快，因为我想早点进入第二部分。**通信到底有多快**？我们来测 all-reduce 和 reduce-scatter 的有效带宽（这跟上一讲算 MFU 的思路是同一个：算出"应该搬了多少字节"，除以耗时）。

### 测量 all-reduce

```python
def all_reduce(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)

    # 创建张量
    data = torch.randn(num_elements, device=cuda_if_available(rank))

    # 热身（warmup）
    dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()  # 等 CUDA kernel 跑完
    dist.barrier()            # 等所有进程到这里

    # 正式测量
    start_time = time.time()
    dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()
    dist.barrier()
    end_time = time.time()

    duration = end_time - start_time
    print(f"[all_reduce] Rank {rank}: all_reduce(world_size={world_size}, "
          f"num_elements={num_elements}) took {render_duration(duration)}", flush=True)

    # 计算有效带宽
    dist.barrier()
    size_bytes = data.element_size() * data.numel()
    sent_bytes = size_bytes * 2 * (world_size - 1)  # 2x：发送 + 接收；world_size-1：all-reduce 的步数
    total_duration = world_size * duration
    bandwidth = sent_bytes / total_duration
    print(f"[all_reduce] Rank {rank}: all_reduce measured bandwidth = "
          f"{round(bandwidth / 1024**3)} GB/s", flush=True)
    cleanup()
```

注意这里有两个"异步"来源要同时处理：**CUDA kernel 的异步**（同一个进程里，操作提交后不等待完成）和**进程之间的异步**（不同进程进度不同）。所以计时前要 `torch.cuda.synchronize()` 保证本进程的 kernel 跑完，再用 `dist.barrier()` 保证所有进程都到齐，然后才开始计时；操作结束后同样地同步 + 屏障，再停表。这个函数在**每一个 rank** 上各自执行，所以四个 rank 会各报一个测量值（可能略有不同，因为它们本来就是不同的进程）；要汇报一个数的话，可以取平均。

测出来大约 **1.6 ms**（100 百万元素）。这个数是好是坏？于是我们算**有效带宽（effective bandwidth）**——"这次通信总共应该搬了多少字节"除以总时间。

先看代码里的记账方式。`size_bytes` 是数据张量的字节数 $M$。**发送了多少字节**需要拆开看：对 all-reduce 来说，打个比方，最简单地你想做"rank 0 + rank 1 + rank 2 + rank 3"，需要迭代 **world size 减 1 步**（因为只有 world size 减 1 次加法）；这里有个因子 **2**，因为既要发送又要接收；再乘以载荷大小——这就是 `sent_bytes = 2(N-1)M`。**总时长**则把测得的墙钟时间乘以 world size（`total_duration = N·T`），因为所有 rank 都在同时做这件事。于是：

$$\text{有效带宽} = \frac{\text{sent\_bytes}}{\text{total\_duration}} = \frac{2(N-1)M}{N \cdot T}$$

测出来的结果在 **400 GB/s** 上下。有三个要点值得记下：

1. **有效带宽 $\approx 2 \times \text{size\_bytes} / \text{duration}$**——因为 $\frac{N-1}{N} \to 1$（$N$ 增大时）；
2. **与世界大小无关**——GPU 加多了，这个带宽数字不会变；
3. **与拓扑无关**——无论是 ring 还是 tree，NCCL 都会自己决定怎么把消息传来传去。

### ring all-reduce 的通信量：完整的推导

为什么会有上面这些性质？这要从 **ring all-reduce** 的算法讲起。设世界大小为 $N$，每个 rank 上待归约的张量有 $M$ 字节（总数据）。把 $M$ 切成 $N$ 块、每块 $M/N$，再把 $N$ 个 rank 排成一个环（rank $i$ 与 rank $i+1 \bmod N$ 相邻）。

**阶段一：reduce-scatter。** 一共 $N-1$ 步。每一步，每个 rank 把当前持有的某一块（$M/N$ 字节）发给环上的下一个邻居，同时从上一个邻居收到一块并**累加**（归约）。$N-1$ 步之后，每个 rank 恰好持有**归约好的 $1/N$**。每 rank 共发送 $(N-1) \times (M/N)$ 字节，于是这一步耗时：

$$T_{\text{reduce-scatter}} = (N-1) \cdot \frac{M/N}{B} = \frac{(N-1)M}{NB}$$

其中 $B$ 是链路带宽。

**阶段二：all-gather。** 同样的 $N-1$ 步，但这次只是**转发**（不归约）：每一步每 rank 把一块 $M/N$ 传给下一个邻居，直到每一块都传遍所有 rank。耗时与 reduce-scatter 相同：

$$T_{\text{all-gather}} = \frac{(N-1)M}{NB}$$

**总耗时**（环上的 all-reduce）：

$$T_{\text{all-reduce}} = T_{\text{reduce-scatter}} + T_{\text{all-gather}} = \frac{2(N-1)M}{NB}$$

**每 rank 的通信量**（发送量，不含接收）：

$$V_{\text{all-reduce}} = 2 \cdot \frac{(N-1)M}{N} = 2\frac{N-1}{N}M$$

当 $N$ 很大时 $\frac{N-1}{N} \to 1$，于是 $T_{\text{all-reduce}} \to \frac{2M}{B}$、通信量 $\to 2M$——这就是"all-reduce 大约等于把数据搬两遍"的由来，也是代码里"有效带宽 $\approx 2 \cdot \text{size\_bytes}/\text{duration}$"的原因。

如果把每次传输的**延迟（latency）** $\alpha$ 也算进来（每次传输除了带宽还要付固定开销，每步是 $\alpha + \frac{M/N}{\beta}$，其中 $\beta$ 是带宽）：

$$T_{\text{all-reduce}} = 2(N-1)\alpha + \frac{2(N-1)M}{N\beta}$$

这条公式解释了 NCCL 为何要在 ring 与 tree 之间选择：**消息越大**，带宽项 $\frac{(N-1)M}{N\beta}$ 越占主导，ring 用 $N-1$ 步充分压榨带宽；**消息越小**，延迟项 $2(N-1)\alpha$ 越要紧，tree 拓扑把步数压到 $\log N$ 级更划算。

现在把推导代回代码的记账：$N$ 很大时，

$$\frac{2(N-1)M}{N \cdot T} \to \frac{2M}{T} \to \frac{2M}{2M/B} = B$$

也就是说代码算出的有效带宽**收敛到链路带宽 $B$**，这正是"与世界大小无关、与拓扑无关"两条结论的出处——只要算法能把链路喂饱，加多少 GPU、换什么拓扑，数字都不变。

### 测量 reduce-scatter

reduce-scatter 的测量套路几乎一样，只是输入是每 rank 一个 $(N, \text{num\_elements})$ 矩阵，输出是每个 rank 一个 $\text{num\_elements}$ 向量：

```python
def reduce_scatter(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)

    # 输入：每 rank 一个 (world_size, num_elements) 矩阵；输出：每 rank 一个 num_elements 向量
    input = torch.randn(world_size, num_elements, device=cuda_if_available(rank))
    output = torch.empty(num_elements, device=cuda_if_available(rank))

    # 热身
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()
    dist.barrier()

    # 正式测量
    start_time = time.time()
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()
    dist.barrier()
    end_time = time.time()

    duration = end_time - start_time
    print(f"[reduce_scatter] Rank {rank}: reduce_scatter(world_size={world_size}, "
          f"num_elements={num_elements}) took {render_duration(duration)}", flush=True)

    # 计算有效带宽
    dist.barrier()
    data_bytes = input.element_size() * input.numel()  # 输入里有多少数据
    sent_bytes = data_bytes * (world_size - 1)         # 需要发送多少（这里没有 2x）
    total_duration = world_size * duration
    bandwidth = sent_bytes / total_duration
    print(f"[reduce_scatter] Rank {rank}: reduce_scatter measured bandwidth = "
          f"{round(bandwidth / 1024**3)} GB/s", flush=True)
    cleanup()
```

记账的区别在于：reduce-scatter **没有那个 2 倍的因子**（它只是把归约后的数据散布开，没有"复制到所有 rank"那一半），所以 `sent_bytes = (N-1) \times data\_bytes`。测出的带宽和 all-reduce 差不多，都在 400 GB/s 量级（有时有随机波动）。

两条要点：

- **all-reduce = reduce-scatter + all-gather**。所以 all-reduce 自然要搬**两倍**的数据——reduce-scatter 有一份代价、all-gather 有一份代价，all-reduce 两者都做；
- 但它也花**两倍**的时间——两个代价抵消，于是算出来的**有效带宽一样**。

好了，第一部分到此结束。进入第二部分之前，还有问题吗？

<!-- lecture-nav -->

**← 上一节**：[硬件：GPU 如何连接](hardware.md)　**→ 下一节**：[数据并行（DDP）与张量并行](data-and-tensor-parallelism.md)
