---
title: 02 · 训练的记忆与计算核算
lecture: 2
---

# 训练的记忆与计算核算

前面我们把 tensor、精度、FLOPs、算术强度都过了一遍。现在把它们组合起来，核算**训练一个网络**到底需要多少内存、多少计算。

## 深网络：参数有多少？

考虑一个有 $L$ 层、每层输入/激活/输出都是 $D$ 维的网络，每一层是**线性变换 + ReLU**：

```python
class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim) / math.sqrt(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.weight  # 线性
        x = F.relu(x)        # 激活
        return x

class DeepNetwork(nn.Module):
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([Block(dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
```

每层是一个 $D\times D$ 的权重矩阵，所以总参数数：

$$\text{# 参数} = D^2 L$$

![](/lectures/02/deep-network.png)

## 梯度的形状：一个简单例子

先看一个最简单的线性模型 $y = 0.5(x \cdot w - 5)^2$：

```python
x = torch.tensor([1., 2, 3])
w = torch.tensor([1., 1, 1], requires_grad=True)
pred_y = x @ w
loss = 0.5 * (pred_y - 5).pow(2)
loss.backward()
assert torch.equal(w.grad, torch.tensor([1, 2, 3]))  # w.grad = [1, 2, 3]
```

前向算 loss，反向算梯度，得到 $w.grad = [1, 2, 3]$。这就是“前向 + 反向”的骨架，下面我们来数反向的 FLOPs。

## 单层 backward：恰好是 forward 的两倍

考虑一个简化的两层线性网络（$h_1 = xW_1$，$h_2 = h_1 W_2$），聚焦**第二层** $h_2 = h_1 W_2$：

- **forward FLOPs**：$2BDD$（就是 $2BDK$，$K=D$）；
- **backward 需要算两个梯度**：
  - 对输入的梯度 $h_1.grad = \partial \text{loss}/\partial h_1$，用 einsum 写就是：

```python
h1_grad = einsum(h2.grad, w2, "batch out, in out -> batch in")   # 2BDD
```

  - 对参数的梯度 $w_2.grad = \partial \text{loss}/\partial w_2$：

```python
w2_grad = einsum(h2.grad, h1, "batch out, batch in -> in out")   # 2BDD
```

- 两个梯度各要 $2BDD$ FLOPs，合计 backward $= 4BDD$。

所以结论：**单看这一层，backward 恰好是 forward 的两倍**：

$$\text{backward} = 2 \times \text{forward}$$

## 全部层加起来：6ND

对每一层都这么算，把全部 $L$ 层加起来（对 MLP 精确成立）：

- **forward** = $2 \times (\text{# 数据点}) \times (\text{# 参数})$；
- **backward** = $4 \times (\text{# 数据点}) \times (\text{# 参数})$；
- **总计** = $6 \times (\text{# 数据点}) \times (\text{# 参数})$。

$$6 N D$$

这就是 **6ND 公式**的来源！每个训练步的 FLOPs 大致是 $6 \times (\text{数据点数})\times(\text{参数数})$。对这个公式，现在我们已经“祛魅”了：它来自“每个参数在每份数据上，前向一次、反向两次”。

### 对 transformer 的适用性

这个公式对多层感知机（MLP）是精确的；对 **transformer，只要上下文（context）不太长，也是一个很好的近似**。上下文太长时会出现**与序列长度平方相关**的项（注意力），这是后面要处理的部分。

## 优化器：从 SGD 到 Adam

梯度算完了，下一步用优化器更新参数。优化器家族的谱系：

- **momentum** = SGD + 梯度的指数平均；
- **AdaGrad（2011 年）** = SGD + 按梯度平方和缩放——介于 SGD 与 Adam 之间，存储的是**梯度平方的累加和** $g^2$；
- **RMSProp** = AdaGrad，但用梯度平方的**指数平均**；
- **Adam** = RMSProp + 动量：同时存**一阶矩**和**二阶矩**。

AdaGrad 的实现（幻灯片里的代码）：

```python
class AdaGrad(torch.optim.Optimizer):
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                state = self.state[p]
                grad = p.grad.data
                # 二阶矩:g2 = sum_{i<t} g_i^2
                g2 = state.get("g2", torch.zeros_like(grad))
                g2 += torch.square(grad)
                state["g2"] = g2
                # 更新参数
                p.data -= lr * grad / torch.sqrt(g2 + 1e-5)
```

## 优化器状态内存

优化器除了参数本身，还要额外存**状态**：

- **AdaGrad**：每参数 **4 字节**(fp32)，存二阶矩 $g^2$；
- **Adam**：每参数 **8 字节**(fp32)，存一阶矩 + 二阶矩。

为什么优化器状态用 **fp32**？这是**稳定性**的习惯做法：你要对很多步的平方做**累加/平均**，低精度下误差会累积，所以状态保持 fp32。

## 训练一个 deep network 的内存分解

对 $D^2L$ 个参数，训练时的内存大致分四块（以 Adam 为例）：

| 项目 | 每参数字节数 | 说明 |
|------|------|------|
| 参数 | 2 | bf16 |
| 梯度 | 2 | bf16 |
| 优化器状态 | 4(AdaGrad)/ 8(Adam) | fp32，为稳定性 |
| 激活 | $2 \cdot B \cdot D \cdot L$ 字节 | bf16，取决于 batch 与层数 |

激活这一项不是按参数算的，而是按数据流算：每层要保留进出的激活，$2BDL$。

## 每个训练步骤的计算

$$\text{每步 FLOPs} = 6 \times B \times (\text{# 参数})$$

其中 $B$ 是 batch size。注意这里数据点就是 batch size。

## 内存既决定“装不装得下”，也决定速度

前面热身问题里“53B 上界”用的就是这套账。还要强调一遍：内存大小不仅决定模型**能不能放进显存**，也决定**跑得快不快**——因为数据要从内存**搬到计算核**，内存越大，搬运越久。

## transformer 的核算更复杂

本节的账是给 deep network（线性 + ReLU）算的。**transformer 的核算更复杂，但思路相同**——**作业 1 会让你亲自做这件事**。幻灯片也推荐了两篇博客：讲 transformer 训练内存的（transformer-memory）和讲 transformer FLOPs 的（transformer-flops）。

## 小结

- 深网络参数 = $D^2L$；
- 单层 backward 是两个 $2BDD$，恰好是 forward 的两倍；
- 全部层加起来：**forward = 2×（数据点）×（参数），backward = 4×（数据点）×（参数），总计 6ND**；
- AdaGrad（2011）存 $g^2$，Adam = RMSProp + 动量，存一阶、二阶矩；
- 优化器状态内存：AdaGrad 4 字节/参数、Adam 8 字节/参数（fp32）；
- 训练内存分解：参数 2(bf16)+ 梯度 2(bf16)+ 优化器状态 4/8(fp32)+ 激活 $2BDL$；
- 每步 FLOPs = $6 \times B \times (\text{# 参数})$；
- transformer 的核算更复杂，作业 1 会做。
