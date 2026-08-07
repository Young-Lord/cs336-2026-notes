---
title: "11 · Muon 与 μP 深入推导"
lecture: 11
---

# Muon 与 μP 深入推导

## 故事开场：nanoGPT speedrun 里的 Muon

本节的故事从一个很小、但很能说明问题的基准开始：**nanoGPT speedrun**（我们的作业一就是受它启发）。在这个基准上，你用很小的模型、在较短的时间里尽快把损失压到某个目标（讲者记不清具体数字，大概是 3.x）。很多人在这上面不断“爬山”。其中最引人注目的进展之一，是 **Muon**（图里紫色）——它对 **Adam**（蓝色，就在绿色线下面）取得了**显著的收益**，而且速度并不比 Adam 慢多少。

看到紫色和绿色之间的那条差距，你会说“哇，光是把优化器换成这个叫 Muon 的东西，就拿到了这么大的收益”。但紧接着问题来了：**这东西在大规模上真的好吗？** 跑一些缩放研究，结果可能没那么好。这是做研究里非常棘手、也非常重要的一环：我们有一个小规模实验，有一个肯定尺度依赖的东西（优化器），小规模上看到了大收益——那大规模上怎么办？

讲者并不想给出一个坚实的“处方”，而更想讲一个“故事”，让你理解：**我们能做什么、哪些是我们无法事先知道的。**

## Muon：针对“矩阵值参数”的优化器

Muon 是为**矩阵形状的参数**设计的优化器。它的关键操作是：对动量矩阵做（近似）**正交化**。

假设我们有矩阵 $B_t = U S V^{\top}$（SVD 分解）。Muon 想要的是**丢掉奇异值、只保留正交因子**：

$$B_t = U S V^{\top} \;\longrightarrow\; U V^{\top}$$

这个“把矩阵变成正交矩阵”的操作，由 **Newton–Schulz 迭代**近似完成。

### Newton–Schulz 迭代：如何“丢掉奇异值”

在 SVD 的意义下，正交化 $B_t = USV^{\top} \to UV^{\top}$ 就是把每个奇异值 $\sigma_i$ 都变成 1。**Newton–Schulz 迭代**正是实现这一点、且**可微、免 SVD** 的近似。它的形式非常简单：

$$X_{k+1} = \frac{1}{2}\,X_k\left(3I - X_k^2\right)$$

在 $B_t = USV^{\top}$ 的奇异值分解下看，这个迭代对每个奇异值逐项作用同一个标量映射

$$g(\sigma) = \frac{3\sigma - \sigma^3}{2}$$

收敛性分析：$g$ 的不动点是 $g(\sigma) = \sigma$ 的解，即 $\sigma \in \{0, \pm 1\}$：

- $\sigma = 1$：$g'(1) = 3(1 - 1^2)/2 = 0$，**二次收敛**的吸引不动点；
- $\sigma = 0$：$g'(0) = 3/2 > 1$，排斥不动点（奇异值会被“清理”掉）；
- 对 $\sigma \in (0, \sqrt{3})$，迭代把 $\sigma$ 单调推向 $1$。

所以迭代收敛到 $B_t$ 的**零次幂（zeroth power）** $B_t^0 = UV^{\top}$——这正是我们要的正交化。实践中不需要做 SVD，直接在原矩阵上反复做矩阵乘即可，而且每一步都是矩阵乘法、可以稳定地在 bfloat16 下运行。

slide 上以及实际实现里用的是系数经过调优的**五次迭代**（quintic iteration）：一次迭代里用多项式 $p(y) = a + by + cy^2$（$a = 3.4445$、$b = -4.7750$、$c = 2.0315$，按“最大化在零点的斜率”选取）作用在 $X_k X_k^{\top}$ 上，从而用更少的迭代步完成正交化。值得注意的是，它并不严格收敛到 $UV^{\top}$，而是收敛到 $US'V^{\top}$，其中 $S'$ 的对角元大致落在 $0.5$ 到 $1.5$ 之间——经验上这对模型性能几乎没有影响。下面是官方实现的简化版：

```python
def zeropower_via_newtonschulz(G, steps=5):
    """把 G 近似正交化：把奇异值推向 1（即“零次幂”UV^T）。"""
    a, b, c = 3.4445, -4.7750, 2.0315            # 五次迭代的多项式系数
    X = G.float()
    if X.size(-2) > X.size(-1):                  # 转成“高”矩阵，使 X @ X^T 是小的 Gram 矩阵
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)   # 把谱范数压到 ≤ 1
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X                        # 一步：奇异值 σ → σ(a + bσ² + cσ⁴)
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95):
    momentum.lerp_(grad, 1 - beta)               # 标准的 SGD 动量
    O = zeropower_via_newtonschulz(momentum)     # 把动量正交化 ≈ UV^T
    O = O * max(1, O.size(-2) / O.size(-1)) ** 0.5   # 非方矩阵的修正因子
    return O
```

参数更新就是 $W \leftarrow W - \eta\, O$（学习率 $\eta$ 的单位是“每次更新的谱范数”）。**要点是：Newton–Schulz 提供了把矩阵投影到正交群上的可微、免 SVD 的近似**——这正是 Muon 的核心。

### Muon 与缩放：证据链

slide 上并排展示了三张图：最左是 **nanoGPT speedrun**（极小规模）——Muon 明显更好；中间是**缩放研究**——增益随规模增大而缩水；最右是 **Kimi K2**——一个在大规模训练里实际用上了 Muon 的模型。

![Muon 的三个证据：nanoGPT speedrun（小）、缩放研究（中）、Kimi K2（大规模实际采用）](/lectures/11/slide-42.png)

讲者的总结是：**缩放收益很难干净地测量**，但 Muon“在大规模上显然是有效的”。他给了一个很有意思的“入选标准”：本课讨论新研究主题时，常问“**它进过大型训练运行吗？**”而 Muon 现在已经达标了。

## μP 深入：为什么我们要关心“参数化”

回想一下 μP（最大更新参数化，maximal update parametrization）当初的**承诺**：如果它成立，那么**超参调优可以在不同规模之间迁移**——这当然非常诱人。但问题是：**它到底怎么工作？在实践中真的有用吗？**

### 先看证据：CerebrasGPT

**CerebrasGPT** 用 Chinchilla 配方训练了 0.1B 到 13B 的一系列模型，核心发现是：**使用 μP 参数化让缩放更稳定**。在小规模上把超参调好、外推到大模型，μP 让整个过程不那么容易“翻车”。

### 什么是 μP：两个断言

有一个“给婴儿看的 μP”（muP for babies）式的表述，把 μP 总结成**关于网络宽度 $n_l$ 的两个断言**：

> **A1**：初始化时，激活（activation）应当保持 $\Theta(1)$；
> **A2**：一步梯度更新之后，激活的变化也应当是 $\Theta(1)$。

注意一个换算：如果单个激活是 $\Theta(1)$，那么**整个激活向量的范数是 $\Theta(\sqrt{n_l})$**。

## 完整推导一：条件 A1（初始化缩放）

考虑一个简单的深度线性网络

$$h_l = W_l h_{l-1}, \qquad W_l \in \mathbb{R}^{n_l \times n_{l-1}}, \qquad W_l \sim \mathcal{N}\left(0, \sigma^2 I_{n_l \times n_{l-1}}\right)$$

**第一步：矩阵集中（matrix concentration）。** 对高斯随机矩阵，谱范数几乎必然收敛到

$$\|W_l\|_2 \;\to\; \sigma\left(\sqrt{n_{l-1}} + \sqrt{n_l}\right)$$

**第二步：范数的传递。** 由次可乘性，激活范数满足

$$\|h_l\|_2 \;\approx\; \|W_l\|_2\,\|h_{l-1}\|_2$$

**第三步：归纳。** 归纳假设 $\|h_{l-1}\|_2 = \Theta(\sqrt{n_{l-1}})$。现在选择

$$\sigma \;=\; \frac{\sqrt{n_l / n_{l-1}}}{\sqrt{n_{l-1}} + \sqrt{n_l}} \;=\; \Theta\left(\frac{1}{\sqrt{n_{l-1}}}\,\min\left(1,\; \sqrt{\frac{n_l}{n_{l-1}}}\right)\right)$$

这个选择让谱范数变成 $\|W_l\|_2 \to \sigma(\sqrt{n_{l-1}} + \sqrt{n_l}) = \sqrt{n_l / n_{l-1}}$。于是归纳步：

$$\|h_l\|_2 \;\approx\; \sqrt{\frac{n_l}{n_{l-1}}} \cdot \sqrt{n_{l-1}} \;=\; \sqrt{n_l}, \qquad \text{即}\qquad \|h_l\|_2^2 = n_l + o(n_l)$$

![μP 条件 A1 的推导：初始化标准差的选择](/lectures/11/slide-47.png)

讲者补了一句很有分寸的注释：这是一个“**最坏情况**”的推导——$\approx$ 实际上是一个上界（用到了 $\|AB\| \le \|A\|\,\|B\|$）。这不是严格的数学证明，而是“物理学家式的量级记账”。

**结论（初始化规则）**：初始化标准差取

$$\sigma \;=\; \Theta\left(\frac{1}{\sqrt{n_{l-1}}}\,\min\left(1,\; \sqrt{\frac{n_l}{n_{l-1}}}\right)\right)$$

当 $n_l \ge n_{l-1}$（fan-out ≥ fan-in）时，这就是熟悉的 $\Theta(1/\sqrt{\text{fan-in}})$；当 fan-out < fan-in 时，还要额外乘上 $\sqrt{n_l/n_{l-1}}$，把初始化压得更小。

## 完整推导二：条件 A2（更新与学习率）

现在处理“更新”。对线性层，一步 **SGD** 更新是**损失对激活的梯度与前一激活的秩一外积**：

$$\Delta W_l \;=\; -\,\eta_l\, \nabla_{h_l}\ell \; h_{l-1}^{\top}$$

一步之后激活的变化，来自两个来源——前一层激活的变化，以及自己权重的变化：

$$\Delta h_l \;=\; W_l\,\Delta h_{l-1} \;+\; \Delta W_l\left(h_{l-1} + \Delta h_{l-1}\right)$$

展开成三项。我们希望**每一项都是 $\Theta(\sqrt{n_l})$**——因为 A2 要求单个激活变化 $\Theta(1)$，于是范数变化就是 $\Theta(\sqrt{n_l})$（假设没有相消）。逐项看：

1. **$W_l\,\Delta h_{l-1} = \Theta(\sqrt{n_l})$**：由归纳假设（$\Delta h_{l-1}$ 的范数量级正确）+ 条件 A1 的谱范数结论直接得到；
2. **$\Delta W_l\, h_{l-1}$**：$\|\Delta W_l\, h_{l-1}\| = \|\Delta W_l\|_*\,\|h_{l-1}\| = \|\Delta W_l\|_*\,\sqrt{n_{l-1}}$。要它等于 $\Theta(\sqrt{n_l})$，就要求

$$\|\Delta W_l\|_* \;=\; \Theta\left(\sqrt{\frac{n_l}{n_{l-1}}}\right)$$

3. **$\Delta W_l\, \Delta h_{l-1} = O(\|\Delta W_l\|_*\,\sqrt{n_{l-1}})$**：高阶小量，不会主导。

所以目标落到了第 2 项：**选择一个学习率，使得 $\|\Delta W_l\|_* \sqrt{n_{l-1}} = \Theta(\sqrt{n_l})$**。

### 第 2 部分：从“损失更新 $O(1)$”解出学习率

这需要最后一个假设——讲者认为这是整段推导里**最不可口**的一个：

> **假设：每步的损失更新 $\Delta \ell = O(1)$。** 也就是说，模型随规模增大仍然做出相当的、非平凡的学习进度，而且这个进度大致与规模无关。

由 Taylor 展开，损失变化近似为

$$\Delta \ell \;\approx\; \Theta\left(\langle \Delta W_l,\; \nabla_{W_l}\ell\rangle\right) \;=\; \Theta\left(\|\Delta W_l\|_F\,\|\nabla_{W_l}\ell\|_F\right) \;=\; \Theta\left(\|\Delta W_l\|_*\,\|\nabla_{W_l}\ell\|_*\right)$$

（最后一步用到 von Neumann 迹不等式/柯西–施瓦茨，把内积用两个范数乘积控制住。）利用 SGD 的 $\Delta W_l = -\eta_l \nabla_{W_l}\ell$，代入 $\Delta \ell = O(1)$ 与 $\|\Delta W_l\|_* = \Theta(\sqrt{n_l/n_{l-1}})$，得到梯度的谱范数：

$$\|\nabla_{W_l}\ell\|_* \;=\; \Theta\left(\sqrt{\frac{n_{l-1}}{n_l}}\right)$$

最后，由 $\Delta W_l = -\eta_l\,\nabla_{h_l}\ell\, h_{l-1}^{\top}$ 以及谱范数的关系，学习率必须取两者的比值：

$$\eta_l \;=\; \frac{\|\Delta W_l\|_*}{\|\nabla_{W_l}\ell\|_*} \;=\; \Theta\left(\frac{n_l}{n_{l-1}}\right)$$

**这就是 μP 的逐层学习率规则**：fan-out 越大、fan-in 越小，学习率越大。

> **Adam 的情形**：Adam 把每个坐标的更新逐坐标归一化，更新量的大小直接由学习率决定，推导的细节与 SGD 不同（slide 上注明了“with Adam, $\|\Delta W_l\|_*\,\sqrt{n_{l-1}} = \Theta(\eta_l)$”这条关系）。结果上，Adam 的逐层学习率不再带 fan-out，而是**与 fan-in 成反比**：

$$\eta_l^{\text{Adam}} \;=\; \Theta\left(\frac{1}{n_{l-1}}\right)$$

也就是说，**fan-in 大的层用更小的学习率**。这正是“逐层自适应学习率（layer-adaptive learning rate）”的由来——在实现里（例如 mup 包），它写作“全局学习率 × (base fan-in / 当前 fan-in)”。

## mini recap：μP 到底改了什么

把上面的推导浓缩成一张表（这就是 slide 上的“mini recap”）：

|  | 初始化标准差 | 学习率（SGD） | 学习率（Adam） |
|------|------|------|------|
| **标准参数化（SP）** | $\Theta\left(\frac{1}{\sqrt{n_{l-1}}}\right)$ | $\Theta(1)$ | $\Theta(1)$ |
| **μP** | $\Theta\left(\frac{1}{\sqrt{n_{l-1}}}\,\min\left(1, \sqrt{\frac{n_l}{n_{l-1}}}\right)\right)$ | $\Theta\left(\frac{n_l}{n_{l-1}}\right)$ | $\Theta\left(\frac{1}{n_{l-1}}\right)$ |

![μP mini recap：初始化与学习率的完整规则](/lectures/11/slide-50.png)

对比标准参数化，μP 的差异有两点：**Adam 的学习率变了**（从 $\Theta(1)$ 变成与 fan-in 成反比）；**当 fan-out < fan-in 时初始化也变了**（否则两者初始化相同）。

### 更进一步：不同组件，不同缩放

μP 是一个“**把超参写成宽度的函数**”的缩放程序。不同类别的参数在 μP 下各自有不同的缩放：embedding、注意力参数（attention params）、输入/输出 MLP 的矩阵乘法（input/output MLP MM）、softmax 线性层（softmax linear）——在 μP 与标准参数化下，它们的缩放规则都不一样。这正是为什么 MiniCPM 那张清单里既要有“缩放 embedding 输出”，也要有“逐张量学习率”和“缩放 LM head”。

### 方法论：这种“推导”本身值得注意

讲者强调，μP 有意思的地方不止于它给出的实用规则，**推导它的方式本身就很有趣**：这是一种非常不同的数学风格——你先对网络取一个缩放极限（scaling limit），然后**断言网络的一些不变量**（A1、A2），再补充额外假设（如损失更新 $O(1)$），最后**解出超参受到的约束**。而这些约束恰恰给出了超参自身的缩放极限。这是一个**发明算法/超参缩放的通用原则**，可能和你习惯的 CS/ML 训练里的思维方式很不一样。

## 复现与鲁棒性：什么会破坏 μP

有一位独立研究者写了一篇论文，系统地“压力测试”了 μP 在各种设定下的表现。复现的结果令人鼓舞：**只要你把 μP 弄对、并且用受控的方式只缩放宽度，你就能得到学习率最优性的精确不变性**——这正是 MiniCPM 等论文的头条结果；无论是基线 μP，还是追踪注意力投影偏置（projection biases for attention）的变体，迁移都很漂亮。

但真实世界的现代语言模型有很多**偏离 μP 理论**的地方：SwiGLU 与 squared ReLU 激活、大小 batch、零注意力之类的初始化变体、RMSNorm 的 gain、花哨的优化器（Lion）、正则化……那么哪些真的会破坏 μP？

- **大部分其实不会**：上述绝大多数与 μP 兼容；
- **RMSNorm 的可学习 gain 会破坏 μP**：在我们的架构里，RMSNorm 带可学习的 gain，结果它打破了 μP 的性质。不过，在很多情况下把这些 gain 去掉损失很小——所以也许不是坏事；
- **基于梯度符号的花哨优化器会破坏 μP**：比如 Lion，它的更新只依赖梯度的符号。这类“符号梯度”优化器与 Muon 等想法在精神上相似，可能还会有其它有趣的东西与 μP 冲突；
- **强解耦 weight decay（0.1）是 μP 最大的失败**：这是讲者看到的唯一显著失败——大 weight decay 下 μP 的性质会失效。

## 那么 μP 有用吗？

总体结论是：**μP 大体上有用**——至少相对于标准参数化（SP）而言，SP 明显更不稳定。现有证据表明，μP 的参数化/初始化**更容易调优**：大量实验显示，标准参数化下，宽度增大时最优学习率会以可预测但强烈的方式漂移；而 μP 下它不漂移。

讲者的收尾措辞很谨慎：μP 只是工具箱里控制“超参随尺度漂移”的**众多工具之一**，它并不是“唯一正确做法”，这仍是一个有趣的开放研究领域；而“直接拟合缩放律”同样很有前途。

## 收尾：scaling in the wild

最后一页把整讲串起来：**“野外”的缩放在实践中非常棘手。** 挑战有三：

1. 设定模型**架构超参**（宽度等）；
2. 设定**优化器超参**（学习率、batch）；
3. 拟合大规模 Chinchilla 扫描所需的**计算量**。

对应的三类对策：

1. **假设稳定性**（或者直接用 μP）；
2. **在小规模上搜索最优学习率/batch**，要么保持固定、要么预测其缩放；
3. 使用**替代学习率调度（WSD 之类）**，让扫描变便宜。

回到开场那句话：缩放定律的初次呈现，让它听起来像一门**科学**——“画一条线，按这个流程走，你就能确定大规模会发生什么”。但现实中它**更混乱、更未知**：人们确实用缩放定律来选架构、选优化器、选超参，但那里面有很大成分是**艺术**——你并不真正知道外推能否永远成立，你只能做一些“合理的事”来最大化成功概率。可以用的办法很多：μP、搜索最优学习率、各种控制超参漂移的手段——但**目前还没有银弹**。也许明年就会有一个“我们解决了”的模块，但还不是现在。

<!-- lecture-nav -->

**→ 下一讲**:[12 Evaluation](../12/)
