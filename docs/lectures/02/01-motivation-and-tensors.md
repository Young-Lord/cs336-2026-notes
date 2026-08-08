---
title: 02 · 资源核算动机与 tensor 基础
lecture: 2
---

# 资源核算的动机与 tensor 基础

## 课程的核心目标：在有限资源下训练最好的模型

回到这门课的根本问题：**给定一组有限的资源，我们能训练出最好的模型是什么？** 这里的资源包括：

- **计算（compute）**；
- **内存（memory）**；
- 有时还有**数据（data）**——但对本课程来说，数据通常不是限制因素。

换句话说，我们的目标是**最大化（计算）效率**。而想要优化计算效率，前提是先理解**一次给定计算的效率**——这需要我们对**计算量（compute）**和**内存（memory）**都有清晰的账本。这正是本讲的主题：资源核算。

## 两个热身问题

Percy 先给了两个问题，作为“学完本讲你应该能回答的问题”的预览。它们都是非常粗略的“信封背面（napkin math）”估算——本讲的目的不是精确算出每个数字，而是**对资源的量级建立感觉**。

### 问题一：70B 参数、15T token，在 1024 块 H100 上要训练多久？

回答这个问题需要四步：

1. 用公式算出总 FLOPs：**6 × 参数数 × token 数**（$6ND$）——这个公式从哪来，后面会详细推导；
2. 查 H100 的规格表，得到它的 FLOP/s；
3. 引入 **MFU（模型 FLOPs 利用率）**，取 0.5——MFU 衡量实际利用率，后面会讲；
4. 估算这台机器**每天能提供多少 FLOPs**，拿总 FLOPs 一除，得到天数。

算出来的答案是：**143 天**。

### 问题二：用 8 块 H100（每块 80GB HBM）跑 AdamW，最大能训多大的模型？

关键是把“每块 H100 有 80GB 内存”换算成“能放多少个参数”：

- H100 每块 80GB HBM，8 块共 640GB；
- 每个参数需要 **2 + 2 + 4 + 4 = 12 字节**：2 字节存参数（bf16）+ 2 字节存梯度（bf16）+ 4 + 4 字节存优化器状态（fp32，Adam 的一阶矩与二阶矩）；
- 640GB ÷ 12 字节/参数 ≈ **530 亿参数（约 53B）**。

注意 caveat：这里**没有算激活（activations）**，而激活取决于 batch size 与序列长度。所以 53B 是一个**上界**。

这两个问题展示了 napkin math 的威力：不需要精确计算，几步就能对“硬件能跑多大规模的模型”有感觉。

## 三种知识：Mechanics、Mindset、Intuitions

上一讲我们谈到从这门课带走的三类知识，今天分别对应：

- **Mechanics（机制）**：东西是怎么工作的。今天的 mechanics 很简单直接——PyTorch 的 tensor 语义。没有魔法；
- **Mindset（心态）**：资源核算极其重要。Percy 希望大家养成一个习惯：**每写一行代码，都想想它的性能特征**；
- **Intuitions（直觉）**：感受资源是如何被花掉的。今天没有 ML 魔法，那部分留给 Tatsu 下一讲。

## Tensor 基础：一切皆为 tensor

我们从最底层开始往上搭：tensor 是**存储一切的基本构件**。参数（parameters）、梯度（gradients）、优化器状态（optimizer state）、数据（data）、激活（activations）——本质上**一切都是 tensor**。

举个例子：去看 **DeepSeek v3.2** 模型，你会发现模型本身就是一**大堆不同的 tensor**，每个 tensor 有自己的**形状（shape）**与**精度（precision）**。精度我们下一节专门讲，这里先看形状。

### rank：维度数

tensor 的 **rank** 就是它的维度数：

```python
x = torch.zeros(4)      # rank 1,向量
x = torch.zeros(4, 8)   # rank 2,矩阵
x = torch.zeros(4, 8, 2) # rank 3
```

在 transformer 里，我们会看到 **rank-4** 的 tensor：

```python
B = 32   # 批大小(batch size)
S = 16   # 序列长度(sequence length)
H = 16   # 头数(number of heads)
D = 64   # 每头维度(hidden dimension per head)
x = torch.zeros(B, S, H, D)
```

即 $B \times S \times H \times D$ 的张量。tensor 是向量、矩阵的推广，可以推广到任意维度。

## tensor 存哪里：CPU 还是 GPU？

默认情况下，你在 PyTorch 里创建的 tensor 存储在 **CPU 内存**里：

```python
x = torch.zeros(32, 32)
assert x.device == torch.device("cpu")
```

但 GPU 的并行能力远大于 CPU。想利用 GPU 的**大规模并行**，就必须把 tensor 移到 **GPU 内存**里：

```python
x = x.to(device)
```

或者直接在 GPU 设备上下文里创建 tensor：

```python
with torch.device(device):
    x = torch.zeros(32, 32)
```

![](/lectures/02/cpu-gpu.png)

Percy 还透露了一个小插曲：他讲的这套幻灯片是在**笔记本（没有 GPU）上执行的**，所以有些 GPU 相关代码他只展示、不现场运行。移动 tensor 到 GPU 本身很简单——**但一定要记得做，否则你根本得不到任何加速**。

## 小结

- 资源核算的目标：给定计算/内存资源，训练最好的模型，最大化计算效率；
- 两个热身问题给出了本讲的路线图：6ND FLOPs、规格表、MFU、每参数字节数；
- tensor 是存储一切的基本构件，rank 是维度数，transformer 常见 rank-4 张量 $B \times S \times H \times D$；
- tensor 默认在 CPU 内存，要利用 GPU 并行必须显式移动。

下一节我们看看 tensor 到底占多少内存——这取决于数值精度。

---

<!-- lecture-nav -->

**← 上一讲**:[01 Overview and Tokenization](../01/)
