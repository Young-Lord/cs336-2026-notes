---
title: "06 · 基准测试与性能剖析"
lecture: 6
---

# 基准测试与性能剖析

硬件讲完了。接下来讲的是贯穿整个 kernel 编写过程的方法论。Percy 强调，这一部分的**哲学**比内容更重要：

> **成功的配方（recipe for success）**：先 **benchmark 和 profile** 你的代码 → **做出改动** → 再 **benchmark 和 profile** 你的代码。

之所以把 benchmark 与 profile 放在教你怎么手写 kernel **之前**，是因为你**永远应该先测量**：在动手写任何 kernel 之前，先搞清楚代码里到底发生了什么、瓶颈在哪里。

## Benchmarking：代码要花多久？

**Benchmarking**（基准测试）测量的是执行某个操作的**墙钟时间（wall-clock time）**——它只给你一个端到端的时间，**不告诉你时间花在了哪里**（那是 profiling 的事）。

尽管如此，它依然非常有用：

- **对比不同实现**——哪个更快？
- **理解性能如何扩展（scaling）**——比如随着维度增大，时间怎么变？

PyTorch 有现成的 `torch.utils.benchmark` 工具，但本课“从零开始”的精神，Percy 选择**自己实现一个**，顺便指出计时的几个坑。先看怎么准备一次运算：

```python
def run_operation2(dim: int, operation: Callable) -> Callable:
    # 创建两个随机的 dim x dim 矩阵
    x = torch.randn(dim, dim, device=cuda_if_available())
    y = torch.randn(dim, dim, device=cuda_if_available())
    # 返回一个执行该运算的函数
    return lambda: operation(x, y)
```

`run_operation2` 生成两个随机方阵，返回一个闭包；调用它即执行 `operation`（比如 `a @ b`）。然后看正确的计时方法：

```python
def benchmark(run: Callable, num_warmups: int = 1, num_trials: int = 3) -> float:
    """对 `run` 做 `num_trials` 次基准测试，返回平均耗时。"""
    # Warmup：第一次运行可能因为编译等原因更慢。
    # 既然 kernel 会反复执行，真正重要的就是稳态（steady state）耗时。
    for _ in range(num_warmups):
        run()
    torch.cuda.synchronize()  # 等待 CUDA 线程结束（非常重要！）

    # 现在正式计时
    times: list[float] = []
    for _ in range(num_trials):  # 多次运行以捕捉方差
        # 用 CUDA events 计时，得到准确的 GPU 时间（避免计入 CPU 开销）
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()  # 开始计时
        run()  # 真正执行计算
        end_event.record()  # 结束计时

        torch.cuda.synchronize()  # 等待 CUDA 线程完成

        times.append(start_event.elapsed_time(end_event))

    return mean(times)  # 这里简单取平均
```

几个关键点：

- **一定要 warmup**。有些东西是惰性编译（lazily compiled）的，第一次运行的时间不该计入——因为你通常关心的是“反复运行时的速度”，初始条件无关紧要；
- **要多次计时**，因为存在方差。更讲究的话可以看整个分布（比如 P95），这里简单取平均；
- **用 CUDA events 计时**：`start_event.record()` 与 `end_event.record()` 记录的是 GPU 时间，不会把 CPU 端的调度开销算进去——直接 `time.time()` 包住调用会把 CPU 开销混进来；
- **记得 `torch.cuda.synchronize()`**：GPU 上的一切都是异步发生的，不显式同步，计时就会在 kernel 还没跑完时结束。

### 看看扩展性：matmul 的时间曲线

对矩阵乘法做基准测试（矩阵维度取 1024，操作是 `a @ b`），然后看时间随维度如何变化——对 $[256, 512, 1024, 2048, 4096, 8192]$ 各测一次：

```python
results = {}
for dim in [256, 512, 1024, 2048, 4096, 8192]:
    results[dim] = benchmark(run_operation2(dim=dim, operation=lambda a, b: a @ b))
```

观察结果：**矩阵乘法的时间如预期地以三次方（cubic）增长**；但注意存在一个“地板”——在维度达到约 2000 之前，时间几乎是个常数。原因在于前面讨论过的硬件特性：GPU 是为**足够大的矩阵乘**而设计的，如果矩阵太小（比如 2×2），它是非常低效的。

## Profiling：时间到底花在哪里？

**Benchmarking 只看端到端时间，profiling（性能剖析）则告诉你时间花在哪里。** 另外，profiling 有一个独立于性能的价值：**帮助你理解底层到底发生了什么**。尤其在使用高层语言时，你写了一段代码、它跑出一个结果——有时候了解一下“实际执行了什么”是很有益的。

PyTorch 内置了 **profiler**；本课程作业里你会用到 **Nsight** 获得更多细节，这里出于时间考虑略过。看 profiling 的实现：

```python
def profile(run: Callable, num_warmups: int = 1):
    # Warmup
    for _ in range(num_warmups):
        run()
    torch.cuda.synchronize()

    # 在 profiler 上下文中运行代码
    with torch.profiler.profile(activities=[ProfilerActivity.CUDA]) as prof:
        run()
        torch.cuda.synchronize()

    # 打印按 CUDA 时间排序的表格
    table = prof.key_averages().table(sort_by="cuda_time_total",
                                      max_name_column_width=100,
                                      row_limit=10)
    return table
```

### 例一：`a + b`

对两个 2048×2048 的矩阵做加法，profiler 显示：时间几乎 100% 花在**一个名字很长的 CUDA kernel** 上（以 `...add` 结尾，由 CUDA 的 functor 生成）。如果你不深入 profiler，大概不会想到“两个 tensor 相加”在底层竟然是一个独立的 kernel。这个例子本身时间分布很无聊（就一个 kernel），但它清楚地告诉你：**在 PyTorch 里你写一个 `+`，底层有一个叫 `add` 的东西在干活**。

### 例二：`a @ b` 与 kernel 名字泄露的信息

对 2048×2048 的矩阵乘法做 profiling，同样出现一个长名字的 kernel。这个例子比 `add` 有意思得多——**kernel 的名字直接泄露了实现细节**。一个典型的名字长这样：

```
cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x64x16_1x1x1_3_nnn_align1_...
```

逐段解读：

- **cutlass**：NVIDIA 的 CUDA 线性代数库（CUTLASS）；
- **sm100**：对应 NVIDIA **Blackwell** 架构（即 B200）——这是一个专门为 Blackwell 设计的 kernel；
- **f32**：float32；
- **64x64x16**：**tile 的形状**（这正是后面讲 tiling 时要详细讨论的东西）。

再注意一个细节：把维度改成 **128×128**，你会得到**另一个不同的 kernel**——它的 tile 变成 **32x32x16**。也就是说：

> **在 PyTorch 里，同一个运算（matmul），维度不同，底层调用的 CUDA kernel 就不同。** 底层有各种针对特定尺寸调优的实现，profiler 里那些长名字就是它们的身份证。

由此得到两条观察：①你可以看到实际被调用的 CUDA kernel（那些长名字的家伙）；②**不同维度会调用不同的 CUDA kernel**，而且名字能告诉你实现的很多信息（用什么库、为哪代架构、用什么 tile 形状）。

## 案例研究：GeLU 的三个实现

把 benchmark 与 profile 用在一个具体例子上——**GeLU 激活函数**。回忆 GeLU 的定义，它常被近似成计算上更友好的 **tanh 近似**：

$$\mathrm{GeLU}(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\!\left(\sqrt{\frac{2}{\pi}} \cdot (x + 0.044715 \cdot x^3)\right)\right)$$

其中 $\sqrt{2/\pi} \approx 0.79788456$。现在我们有三个“参赛选手”：

**选手一：naive——从零手写。** 把上面的公式直接翻译成 PyTorch：

```python
def naive_gelu(x: torch.Tensor):
    # GeLU 的 tanh 近似
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))
```

**选手二：builtin——PyTorch 内置。** PyTorch 也提供了现成的版本：

```python
def builtin_gelu(x: torch.Tensor):
    # PyTorch 内置的 tanh 近似版 GeLU
    return torch.nn.functional.gelu(x, approximate="tanh")
```

可以先用随机输入核对两个版本结果一致。

**选手三：compiled——让编译器优化 naive 版本。** 这是很多人未必接触过、但非常重要的一点：**任何 PyTorch 函数都可以交给 `torch.compile`**，它会生成另一个行为相同的函数：

```python
compiled_gelu = torch.compile(naive_gelu)  # 编译不应改变语义
```

同样可以用随机输入验证它和 naive 版本结果一致。

### 基准测试结果

对 $16384 \times 16384$ 的矩阵分别跑三个实现：

```python
naive_time = benchmark(run_operation1(dim=16384, operation=naive_gelu))
builtin_time = benchmark(run_operation1(dim=16384, operation=builtin_gelu))
compiled_time = benchmark(run_operation1(dim=16384, operation=compiled_gelu))
```

结果：naive 版本明显最慢（约 3.75 的量级），builtin 快得多，compiled 也快得多、但略逊于 builtin。三个实现算出的答案完全相同，性能却天差地别——**为什么？** 用 profiler 一看便知。

### Profiler 揭示的秘密：算子融合

对三个实现分别做 profiling：

- **naive_gelu**：profiler 里出现**一堆不同的 kernel**——`binary`（二元运算）、`unary`（一元运算）、`add`、`tanh`……每一个都是一次独立的 kernel 启动。原因在于：PyTorch 里写一个表达式，计算图里的**每个基本操作都会被实现成一个 kernel**。每启动一个 kernel，都要从 HBM 读数据、搬到 SM、算完、写回；下一个 kernel 再从 HBM 取走结果……数据在 HBM 与 SM 之间**来回往返**。这就是**没有算子融合**（no fusion）的代价；
- **builtin_gelu**：profiler 里只有一个 **GeLU 的 CUDA kernel**。为什么？因为大家都用 GeLU，所以有人为它专门写了一个 kernel 放进了标准库——没什么神奇的，就是“被优化的热门操作”；
- **compiled_gelu**：最有趣——profiler 显示它**只有一个 kernel**，而且**是一个 Triton kernel**。`torch.compile` 做的事，本质上是**看懂你的计算图，然后把它编译成（Triton）kernel**。

所以结论很清楚：

- naive：多个 kernel，需要多次读写 HBM，**没有融合**，慢；
- builtin 与 compiled：都是单 kernel，GeLU 的所有运算被**融合**进一个 kernel——每个元素**读一次 HBM、写一次 HBM**；
- compiled 的 kernel 是一个 **Triton kernel**——这正好成为进入 Triton 的引子。

### 课堂问答

**问答：builtin 的 kernel 也是用 CUDA 写的吗？**

（同学：builtin 那个 kernel 是不是也是直接用 CUDA 写的、绕开 PyTorch？）

**Percy**：profiler 里它显示为 “CUDA kernel”，所以大概确实有人用 CUDA 手写了它。

**问答：为什么 Triton kernel 比 builtin 还慢？**

（同学：为什么 compiled 的 Triton kernel 比 builtin 的 CUDA kernel 慢？）

**Percy**：Triton kernel 在这个例子里**并不**比 builtin 快——compiled 版本是单 kernel，但比 builtin 略慢。去年讲这门课的时候两者还差不多，这些东西变化很快、且高度依赖硬件；而且这里的实现都没做极端调优，只是给你一个整体概念。

## 小结

最后再强调一遍：**benchmark 和 profile 你的代码！** 作业里也会要求你这样做，所以你没有借口不做。测量的价值再怎么强调也不为过——它是连接“编程模型”与“硬件现实”的桥梁。

<!-- lecture-nav -->

**→ 下一节**：[Triton 入门：GeLU kernel 与 PTX](triton-gelu.md)
