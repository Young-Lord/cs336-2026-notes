---
title: 02 · 开场与动机
lecture: 2
---

# 第 2 讲：资源核算（Resource Accounting）

**讲师**：Percy(Liang)· **主题**：资源核算（系统方向）

## 开场：希望大家没被雨淋到

Percy 的开场第一句话是："I hope everyone is staying dry. I'm not.“——希望大家都没被雨淋到，我自己可没有。这是个关于天气的玩笑，也一下子把课堂气氛带回了”很实在“的基调。紧接着他就分享了一个相当令人振奋的进展。

## Marin 项目：1e23 FLOPs 的训练跑完了，而且预测被验证

上一讲我们提到，课程在跑一个叫 **Marin** 的项目。Percy 今天宣布：**那个一直在跑的 1e23 FLOPs 训练已经跑完了，而且结果与预测完全匹配**。

- 还记得我们当时在跑的那些曲线吗？其中**每一条曲线本质上都是一条 ISOFLOP 曲线（等算力曲线）**：在固定的算力预算下，跑一批小模型，找到计算最优的点，然后拟合出一条**缩放定律（scaling law）**；
- 基于拟合出的缩放定律，我们**预测**了某个点的 loss；现在把这个模型真正训练出来，实测的 loss 与预测**只差了 0.05**；
- 如果把这个预测**外推到 GPT-5 级别**的性能，就能得到你看到的这条 loss 曲线。

Percy 觉得这真的很酷。当然他也提醒了一句：你的实际结果可能因缩放定律的具体拟合方式而异（your mileage might vary），但至少这次，理论预测与实际训练对上了。

![](/lectures/02/marin-run.jpg)

## 课程公告

- 加入 CS336 的 **Slack**；
- 用 **Stanford 邮箱**在 **Modal** 上注册（课程用到的云算力平台）；
- 阅读课程**AI 政策指南**（AI policy guide）；
- 阅读**集群指南**（cluster guide）。

## 上一讲回顾

上一讲我们给了整个课程一个 **overview（总览）**，并深入讲了 **tokenization（分词）**——它会出现在第一次作业里。Percy 提到，今天的内容会更偏向**系统**（systems）那一侧。

## 今天的主题：资源核算

今天要讲的是 **resource accounting（资源核算）**。用一句话概括主线：给定一组有限资源，我们要**训练出最好的模型**，并且**最大化（计算）效率**。而在优化计算效率之前，首先得理解**一次给定的计算到底有多高效**——为此我们需要同时理解**计算**（compute）和**内存（memory）**。

这节课没有 ML 魔法，只有踏踏实实的”算账“：tensor 占多少内存、一次运算要多少 FLOPs、硬件承诺了多少、我们实际用到了多少。

## 本讲导航

| 页面 | 内容 |
|------|------|
| [02 · 资源核算动机与 tensor 基础](01-motivation-and-tensors.md) | 课程目标、两个热身问题（143 天、53B）、三种知识、tensor 与 rank、CPU/GPU 内存 |
| [02 · 数值精度详解](02-precision-formats.md) | fp32/fp16/bf16/混合精度/fp8/fp4、下溢问题、AMP、块缩放与 FP4 |
| [02 · FLOPs 计数与 MFU](03-flops-and-mfu.md) | einops 简介、线性层 2BDK、FLOPs 与 FLOP/s、计时与同步、MFU |
| [02 · 算术强度与 roofline](04-arithmetic-intensity.md) | 通信/计算时间、memory-bound 与 compute-bound、ReLU/GELU/点积/matmul 强度、roofline |
| [02 · 训练的记忆与计算核算](05-memory-and-compute.md) | 深网络参数、backward 是 forward 两倍、6ND 公式、优化器状态、内存分解 |
| [02 · 降低内存的技巧与总结](06-memory-optimizations.md) | 梯度累积、激活检查点（重物化）、检查点间隔的取舍、本讲总结 |

## 本讲要点

- 本讲的目标是**资源核算**：给定计算与内存资源，训练最好的模型，并最大化计算效率；优化之前，先理解给定计算的效率；
- 两个”信封背面（napkin math）“热身问题：70B 模型、15T token、1024 块 H100，约需 **143 天**；8 块 H100（每块 80GB）用 AdamW 最多约 **530 亿参数**（不含激活的上界）；
- 三种知识：今天 Mechanics 很简单（PyTorch/tensor 语义），重点是培养 **Mindset（随手做资源核算）** 与 **Intuitions（感受资源去向）**；
- 一切皆是 tensor：数据、参数、梯度、优化器状态、激活；DeepSeek v3.2 模型就是一大堆 tensor；
- 数值精度从 fp32 一路走到 fp4，核心权衡是**动态范围 vs 分辨率 vs 内存**；混合精度训练（bf16 + fp32 优化器状态）是现代实践；
- FLOPs 计数的基石：线性层 $2BDK$，或者说 $2 \times (\text{# 数据点}) \times (\text{# 参数})$；
- **MFU = 实际 FLOP/s ÷ 承诺 FLOP/s**，现代模型 0.5 就很不错，纯 matmul 可到 0.8；
- 为什么到不了 1？因为**内存搬运**是瓶颈——这就是算术强度与 roofline 分析要回答的问题；
- 每个训练步的 FLOPs 是 **6 × （# 数据点） × （# 参数）**，即 6ND；backward 恰好是 forward 的两倍；
- matmul 是 compute-bound，基本其它一切（element-wise 激活、点积、矩阵向量乘）都是 memory-bound；
- **梯度累积**与**激活检查点**（重物化）用更多计算换更少内存，从而能上更大的 batch。

## 课程导航

- [上一讲：01 Overview & Tokenization](../01/)
- [下一讲：03 Architecture](../03/)
