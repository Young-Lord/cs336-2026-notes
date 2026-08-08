---
title: "06 · Triton 入门：GeLU kernel 与 PTX"
lecture: 6
---

# Triton 入门：GeLU kernel 与 PTX

现在开始写真正的 kernel。先回顾编程模型：网格（grid）里是一堆线程块（thread block），线程块里是一组线程。那么写 kernel 时该以什么粒度思考？

## CUDA 与 Triton：指定“线程做什么”还是“块做什么”

**CUDA** 由 NVIDIA 开发，多年来是写 kernel 的标准方式。CUDA 的心智模型是：**指定每个线程做什么**。你写一小段代码，代码里通过某个 ID 区分“我是哪个线程”，然后执行运算。好处是它**非常贴近底层实际发生的事**，给你**细粒度的控制**；坏处是你需要**管理更多东西**：还记得线程块里的线程需要通信吗？那就得自己处理同步（synchronization）、自己安排“所有人一起从 HBM 读入 → 同步 → 计算 → 同步 → 写回”这类簿记（bookkeeping）。如果只做逐元素运算，CUDA 完全够用、甚至更简单；但一旦运算变复杂，这些簿记就变得很烦人。

**Triton** 由 OpenAI 开发，如今已是相当主流的方案。Triton 的心智模型是：**指定每个线程块做什么**。对大多数情况（尤其是本课这种入门场景），这个粒度**足够强大**；如果你想榨干最新硬件的每一个新特性，它也许给不了全部灵活性——但本讲不纠结这个。

Triton 的**概念框架**非常简单：

> **把数据 load 进 shared memory → 在上面运算 → 把结果写回全局内存（global memory）。**

在这个框架里，块处于“逐个元素思考”与“整体大操作思考”之间的中间点。在 PyTorch 里，你定义巨大的矩阵、说“把它们乘起来”，这是**原子级的大操作**，你的心思主要花在“怎么把问题归约成若干个大矩阵乘”上；而 Triton 某种程度上是“大操作”与“逐元素”的**混合体**——下面会看到。

## 第一个 Triton kernel：GeLU

先定义一个 8192 维的向量，然后写 Triton。第一步是普通的 PyTorch 准备工作：

```python
x = torch.randn(8192, device=cuda_if_available())
y = triton_gelu(x)
```

这里没有返回值的概念了——**Triton 不再是函数式的，你得显式地读和写**。所以先分配一个输出 tensor，让 kernel 往里写：

```python
def triton_gelu(x: torch.Tensor):
    # 检查输入
    assert x.is_cuda
    assert x.is_contiguous()

    # 分配输出 tensor
    y = torch.empty_like(x)

    # 确定网格（把元素划分成块）
    # | T T T T T T T T | T T T T T T T T | ... | T T T T T T T T |
    # |    Block 0      |    Block 1      |     ...    |  Block 7  |
    num_elements = x.numel()          # 8192
    BLOCK_SIZE = 1024                 # 每块 1024 个线程
    num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)  # 8

    # 启动 kernel
    kernel = triton_gelu_kernel[(num_blocks,)](
        x, y, num_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return y
```

这个 tensor 可以任意大，一般放不进一个 SM，所以要把数据**切块**：`num_elements = 8192`，块大小（block size）暂定为 1024，得到 **8 个块**。`kernel[(num_blocks,)]` 是 Triton 特殊的启动语法——方括号里的内容描述**网格的形状**：这里网格有 8 个块，每个块都调用一次 `triton_gelu_kernel`，传入 `x`、`y`、元素总数与块大小。

kernel 本身是“你能想象到的最简单的 kernel”：

```python
@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    # 输入从 x_ptr 开始，输出从 y_ptr 开始

    pid = tl.program_id(axis=0)      # 标识这个块（我是谁？）
    start = pid * BLOCK_SIZE         # 本块的起始下标

    # 本块要操作的索引
    offsets = start + tl.arange(0, BLOCK_SIZE)

    # 不要读写到 tensor 末尾之外
    mask = offsets < num_elements

    # 读入
    x = tl.load(x_ptr + offsets, mask=mask)

    # GeLU 近似：0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # tl.tanh 不存在，用 tanh(a) = (exp(2a) - 1) / (exp(2a) + 1) 计算
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp = tl.exp(2 * a)
    tanh = (exp - 1) / (exp + 1)
    y = 0.5 * x * (1 + tanh)

    # 写回
    tl.store(y_ptr + offsets, y, mask=mask)
```

注意：之前那个函数里的 `x`、`y` 到这里变成了**指针（pointer）**——你可以把它们想象成整数（内存地址），需要适应这一点。`num_elements` 与 `BLOCK_SIZE` 则从启动处传进来。逐行看它的执行过程：

1. **醒来，问“我是谁”**：`tl.program_id(axis=0)` 给出 **program ID（pid）**，标识当前是哪个块——块 0、1、2、3……各得各的编号；
2. **确定操作哪段数据**：`start = pid * BLOCK_SIZE` 是 `x` 指针上的偏移。`pid = 0` 时 `start = 0`，`pid = 1` 时 `start = BLOCK_SIZE`，以此类推；
3. **确定操作的跨度**：`offsets = start + tl.arange(0, BLOCK_SIZE)`。`tl.arange(0, BLOCK_SIZE)` 概念上给出整数 $0, 1, \dots, \text{BLOCK\_SIZE} - 1$，于是对于块 1，`offsets` 是 `BLOCK_SIZE` 到 `2 * BLOCK_SIZE - 1`；
4. **mask 防越界**：本例中 8192 恰好被 1024 整除，但一般情况不是。所以 Triton 代码里常有 `mask = offsets < num_elements`——tensor 只到这里为止，mask 在有效范围内为真、之后为假；对非末尾的块它全为真；
5. **读**：`tl.load(x_ptr + offsets, mask=mask)` 就是**指针算术**——把 `offsets` 加到 `x_ptr` 上，得到前 `BLOCK_SIZE` 个元素的地址，按 mask 读入（被 mask 掉的就不读）；
6. **计算**：现在可以把 `x` 当作一个**向量**来写正常的数学运算了。因为 `tl.tanh` 不存在，用恒等式 $\tanh(a) = \dfrac{e^{2a} - 1}{e^{2a} + 1}$ 实现；注意 GeLU 近似里的常数 $\sqrt{2/\pi} \approx 0.79788456$；
7. **写回**：`tl.store(y_ptr + offsets, y, mask=mask)`，把结果写回 HBM。

于是整个 kernel 的形态就是：**load 从 HBM 读入 → 做点计算 → store 写回 HBM**。这也是之后所有 kernel 的通用骨架——输入输出、醒来定位、读取、计算、写回。

### 课堂问答

**问答：这和 CUDA 有什么不同？**

（同学：对逐元素运算来说，这个和 CUDA 看起来几乎一样？）

**Percy**：确实，逐元素时两者差不多，甚至 CUDA 更简单——它真是逐元素的，你醒来、确定线程号、直接操作那一个元素。Triton 这里是**向量化**（vectorized）的版本：一个块操作一段数据。等我们后面做比逐元素更复杂的东西（比如需要线程间通信的运算）时，你就会发现 CUDA 会麻烦得多。

**问答：这和 tensor core 有什么关系？**

（同学：如果想用张量单元（tensor units）怎么办？）

**Percy**：后面会展示这段代码实际编译成什么，到时再说。简短回答是：**你不直接控制这一点**——硬件自己决定数据放哪里、用哪个单元算。

**问答：从 HBM 到 shared memory 再到寄存器，具体怎么走的？**

（同学：能不能逐层讲一下执行时数据到底怎么流动？）

**Percy**：先说机制层面：GPU 上并不会真的“调用 Triton 库”来执行这段代码——它是给我们看的、用来描述计算的。编译器（后面会看到）把它编译成 **PTX**，PTX 才是真正干活的东西。从**概念**层面理解：`x_ptr` 是 HBM 里的一个内存位置，它描述了一段连续的地址范围；`load` 取回这些地址对应的数据，放进局部变量 `x` 里——实践中这通常是**寄存器或 shared memory**，具体放哪由 Triton 决定。你并不需要预先精确安排“这块进 shared、那块进寄存器”——编译器替你做。

**问答：`load` 什么时候执行？会不会阻塞？**

（同学：看这段 Triton 代码，`load` 会阻塞若干个周期，然后……是不是有点像 CPU 的 trap？等内存读回来之前，我被换出去，别的 warp 先跑？）

**Percy**：正是如此。这个 `load` 在某线程的某 warp 上运行、而 SM 同时跑多个 warp——当这个 warp 卡在等数据时，调度器就找另一个 warp 来跑；`load` 完成后，warp 调度器再切回来继续。就机制而言，这些 `load` 语句确实是“阻塞若干周期”的，硬件正是靠 warp 切换来填充这些等待。

## 看 PTX：编译出的汇编

Triton 会把 kernel 编译成 **PTX（Parallel Thread eXecution）**——GPU 的**汇编语言**。Percy 当然不会逐行讲汇编，但值得扫一眼 PTX 长什么样、观察几个要点（可以把 Triton 生成的 PTX 存到文件里查看）。几个观察：

- **`ld.global.*` 与 `st.global.*`**：从全局内存（HBM）**读**与**写**的指令；
- **`%ctaid.x` 是块索引（block index）**，**`%tid.x` 是线程索引（thread index）**——这正好对应编程模型里的“块”与“线程”两个概念；
- **`%f*` 是浮点寄存器，`%r*` 是整数寄存器**——`move` 把常数放进寄存器，`mul` 之类做乘法，PTX 就是这个粒度的指令序列；
- **一个线程同时处理 8 个元素（线程粗化）**——看 PTX 会发现一段段重复的指令块：编译器认为这个线程太“轻”，于是让它一次处理 8 个元素，也就是前面讲过的**线程粗化**。编译后每个线程执行的都是这份（处理 8 个元素的）代码。

还有一个重要事实：**这份代码只编译一次，所有线程共享同一份**。线程如何区分彼此？靠 ID——`%ctaid.x` 告诉这块代码“我在哪个块”，`%tid.x` 告诉它“我在块内哪个线程”。

最后，PTX 里仍然有很多未指定的东西：运行在哪些 SM 上、warp 如何组织等等——那些大多由硬件控制，你在 PTX 里根本看不到。

### 课堂问答

**问答：PTX 是编译出来的，不是人写的吧？**

（同学：PTX 看起来像汇编，它是由编译器生成的吗？）

**Percy**：是的，PTX 由编译器生成，正常情况下你不该去手写。确实有人会手写 PTX——如果你觉得自己比编译器更懂的话。NVIDIA 的编译器总体已经很成熟，但一些没那么成熟的其他加速器，有时候你确实得“伸手进去”多照看一些。一般不需要这样做。

**问答：一个 SM 上为什么有 4 个 warp 调度器？**

（同学：同一个 kernel 在每个 SM 上为什么是 4 个 warp 在跑？为什么是 4 个 warp 调度器？）

**Percy**：具体为什么是 4 个 warp 调度器，我不太清楚背后的确切原因——这是硬件设计的选择，你通常不需要操心它。

到这里，第一个 Triton kernel 就讲完了：**load 从 HBM 读、计算、写回 HBM**，再看一眼 PTX 了解底层发生了什么。GeLU 虽然计算上“有点乱”，但它只是逐元素操作，概念上最简单。接下来增加一点难度：softmax（行内归约）。

<!-- lecture-nav -->

**→ 下一节**：[归约与“婴儿 tiling”：softmax 与 row sum](softmax-row-sum.md)
