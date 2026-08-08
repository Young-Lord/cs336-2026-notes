---
title: 02 · FLOPs 计数与 MFU
lecture: 2
---

# FLOPs 计数与 MFU

## 插曲：用 einops 思考 tensor 操作

在进入 FLOPs 核算之前，Percy 先插入了一小段关于 **einops** 的介绍——这是他们会在课程里使用的一个库。Percy 现场问了一下有多少人用过 einsum，大约三分之二的人举了手。

### 动机：维度转置让人头疼

传统 PyTorch 代码很容易写错维度。比如 `z = x @ y.transpose(-2, -1)`，你要盯着 `-2`、`-1` 猜半天它到底转置的是什么。

**einops** 是一个**按命名维度**操作 tensor 的库，灵感来自**爱因斯坦求和记号（Einstein summation notation，1916 年）**。与其用数字下标，不如给维度起名字。einops 可以看作“**带良好记账的广义矩阵乘法**”。

### einsum

最简单的例子：

```python
x = torch.ones(3, 4)  # seq1 hidden
y = torch.ones(4, 3)  # hidden seq2

# 旧写法
z = x @ y             # seq1 seq2

# einops 写法
z = einsum(x, y, "seq1 hidden, hidden seq2 -> seq1 seq2")
```

**没有出现在输出里的维度会被求和掉**——这里是 `hidden`。Einops 的做法是：枚举所有变量（seq1、hidden、seq2）的所有取值，从 x 里取值、从 y 里取值，相乘，累加到结果 $Z_{\text{seq1},\text{seq2}}$。

更复杂的例子，两个都是 2×3×4（batch seq hidden）的 tensor：

```python
x = torch.ones(2, 3, 4)  # batch seq1 hidden
y = torch.ones(2, 3, 4)  # batch seq2 hidden

# 旧写法
z = x @ y.transpose(-2, -1)  # batch seq1 seq2

# einops 写法
z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")
```

注意 einops 里**没有 transpose**——命名本身就完成了转置。如果你写的是 "hidden seq2" 而不是 "seq2 hidden"，那就是不转置的版本。Percy 说，他总被 transpose 搞晕，而 einops 让他不用再想转置，这让他很开心。

还可以用 **`...`** 代替任意数量的 batch 维度：

```python
z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")
```

这在语言建模里很实用：当你有 batch、sequence、head 等多个维度时，不用把它们一一枚举；而且能写出**模块化**的代码——不管进来的 tensor 形状如何，都能用。

### reduce

reduce 是 sum / mean / max / min 的推广。例如对最后一维求和：

```python
x = torch.ones(2, 3, 4)  # batch seq hidden

# 旧写法
y = x.sum(dim=-1)

# einops 写法
y = reduce(x, "... hidden -> ...", "sum")
```

输出里没出现的 `hidden` 会被约掉，聚合操作由最后一个参数指定（sum/mean/max/min）。

**学生问**：reduce 这类操作有没有加速？

**Percy 答**：没有，它最终归结为同样的基本原语操作——可以把它当作**语法糖**，速度是一样的。

### rearrange

有时候一个维度其实代表两个维度，而你想只对其中一个做操作（比如一个矩阵被展平了，你想重新拆开再操作）。例如：3×8 的矩阵，其中 8 这个维度实际代表 2×4（heads × hidden1）：

```python
x = torch.ones(3, 8)  # seq total_hidden
w = torch.ones(4, 4)  # hidden1 hidden2

# 把 total_hidden 拆成两个维度
x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)
# 用 w 做变换
x = einsum(x, w, "... hidden1, hidden1 hidden2 -> ... hidden2")
# 再把 heads 和 hidden2 合并回去
x = rearrange(x, "... heads hidden2 -> ... (heads hidden2)")
```

**学生问**：把一个二维的东西压成一维时，按什么顺序？是不是有 row-major 和 column-major 两种？

**Percy 答**：顺序由**括号里维度的顺序**指定——你写 `(heads hidden1)` 时，展平的顺序就是先 heads 后 hidden1（或反过来，取决于你怎么写）。刚开始需要一点时间适应，但非常值得：一旦用了 einops，你就会换一种方式思考，所有 transpose 和 reduction 都变得顺畅。

## FLOPs：计算量的度量

回到资源核算。有了 tensor、知道了内存，现在问：**运算要多少计算量？**

我们用来度量计算成本的单位是 **flop（floating-point operation，浮点运算）**，基本单元是**加法或乘法**。GPU 还能做别的，但 matmul 这类运算是“面包和黄油”，吃掉大部分时间，其它先忽略。

### 两个发音相同、含义不同的缩写

Percy 强调这是他的一根“雷区”：

- **FLOPs（小写 s）**：floating-point **operations**，衡量**完成的计算量**。比如 GPT-3 用了约 3.14e23 FLOPs；
- **FLOP/s（有时大写写成 FLOPS）**：floating-point operations **per second**，衡量**硬件速度**。Percy 说他总会写成 /s 来区分。

所以“H100 有 989 teraFLOP/s”是速度；而“GPT-3 花了 3e25 FLOPs”是总量。

### 量级直觉

- 训练 GPT-3（2020）约 **3.14e23 FLOPs**；
- 训练 GPT-4（2023）据推测约 **2e25 FLOPs**；
- H100 的峰值是 **1979 teraFLOP/s（带稀疏性）**；
- 8 块 H100 跑两周，大约能得到 $8 \times 2 \text{ 周} \times \text{秒数} \times (1979\text{e12}/2)$ 的 FLOPs，量级在 1e19 左右。

这些都是让你对“硬件给多少、模型要多少”有感觉的粗略算术。

## 线性层的 FLOPs

很多 FLOPs 计数的核心其实都是线性层的 matmul，所以先把这个搞清楚（这不失一般性）：有 $B$ 个数据点，每个点 $D$ 维，把它们映射到 $K$ 维输出。令 $x$ 是 $B \times D$ 的数据矩阵，$W$ 是 $D \times K$ 的权重矩阵：

$$y = xW,\quad x \in \mathbb{R}^{B\times D},\ W \in \mathbb{R}^{D\times K}$$

**FLOPs ≈ 2BDK**。推导：每个输出元素要对 $D$ 个三元组 $(i,j,k)$ 各做**一次乘法 + 一次加法**，严格说是 $D-1$ 次加法，但**忽略那个减 1**。于是总数是 $2 \times B \times D \times K$。

### 另一种看法：2 × （#数据点） × （#参数）

因为 $W$ 是 $D\times K$，参数数是 $DK$，所以：

$$\text{FLOPs} = 2 \times (\text{# 数据点}) \times (\text{# 参数})$$

这个形式对 transformer 也成立——你已经能看到 $6ND$ 公式的影子了。

### 其它操作的 FLOPs

**element-wise 操作（比如加法）**的 FLOPs 就是**元素个数**（NM）。一般来说，**没有别的操作比大矩阵乘法更贵**（对足够大的矩阵），所以我们主要盯着 matmul 看——但讲到内存时要小心，这个 caveat 后面会回来。

**学生问**：有没有其它做矩阵乘法的算法，比如亚三次方（sub-cubic）的复杂度？

**Percy 答**：现实中，人们探索的矩阵乘法优化，绝大多数是**如何与系统协同设计（co-design with the systems）**，而不是这些渐进复杂度（asymptotic）算法。

**学生问**：能不能把加法和乘法区别对待？加法能不能比乘法更高效？

**Percy 答**：在硬件实现里，两者基本是一样的——直觉上加法好像应该更快，但硬件就是按同等代价造的。

## 计时：测出实际 FLOP/s

前面数 FLOPs 是与硬件无关的“算账”。现在问：**在硬件上实际要花多久？**

用**计时**来测。Percy 给了几个要点（几讲之后会正式讲 benchmarking，这是预览）：

- **必须调用 `CUDA synchronize`**：GPU 是异步执行的，不加同步，计时会“虚假地快”——因为 non-blocking 调用直接返回了；
- **操作之后也要同步**：在操作前后各设一道同步屏障；
- **多次运行取平均**。

```python
def benchmark(func, num_trials: int = 5) -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()      # 等之前的 CUDA 线程做完
    def run():
        func()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 等本次操作做完
    total = timeit.timeit(run, number=num_trials)
    return total / num_trials
```

于是**实际 FLOP/s = FLOPs ÷ 时间**。

## 规格表：承诺的峰值

每块 GPU 都有规格表，给出**峰值性能**，而且**强烈依赖数据类型**：

- 查 H100 的规格，bf16 那行写的是 **1979 teraFLOP/s**——你一 benchmark，发现根本达不到；
- 再读小字脚注：1979 是**带稀疏性（sparse）**的数字，**稠密（dense）要除以 2**；
- 所以 H100 bf16 稠密峰值 = **1979e12 / 2 ≈ 989e12 FLOP/s**。

这就是为什么幻灯片里总出现除以 2。

## MFU：模型 FLOPs 利用率

实际值总是低于承诺值。衡量这个差距的指标叫 **MFU（Model FLOPs Utilization，模型 FLOPs 利用率）**：

$$\text{MFU} = \frac{\text{实际 FLOP/s}}{\text{承诺 FLOP/s}}$$

这里忽略了通信与其它开销。要点：

- 现代模型 **MFU ≈ 0.5 就很不错了**，应该对自己满意；
- 如果只是**纯 matmul**，可以到 **0.8** 左右；
- 如果只有 **0.1**，说明哪里出了问题，该去查了；
- MFU 依赖**硬件**和**数据精度**：现在 fp32 非常慢（硬件不为它优化），而 bf16/fp8 快得多。

写模型的时候，你就能算 MFU：数模型需要的逻辑 FLOPs，再看墙钟时间，相除即可。

### 学生互动

**学生问**：规格表里那个“承诺的 FLOP/s”是不是已经是 1979 除以 2 之后的值？

**Percy 答**：是的，规格表里的数字**除以 2 之后**才是承诺值；而在此基础上，你一般只能拿到它的 0.5（视具体计算而定）。

**学生问**：为什么只能拿到 50%？

**Percy 答**：这是个好问题——**留到讲内存瓶颈（memory bottlenecks）时再回答**。

## 小结

- einops 让 tensor 操作更清晰：einsum（命名维度的 matmul）、reduce（sum/mean/max/min）、rearrange（拆/合维度）；
- FLOPs 是计算量，FLOP/s 是速度，两者发音相同但含义不同；
- 线性层 $y = xW$ 的 FLOPs ≈ $2BDK$，等价于 2 × （#数据点） × （#参数）；
- 计时必须 CUDA synchronize 并取平均；
- H100 bf16 稠密峰值 = 1979e12/2；
- MFU = 实际 FLOP/s ÷ 承诺 FLOP/s，0.5 很好、0.8（纯 matmul）很好、0.1 说明有问题；
- 为什么到不了 1？下一节讲算术强度与内存瓶颈。
