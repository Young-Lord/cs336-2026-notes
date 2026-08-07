---
title: "2025 → 2026 第 1/2 讲差异"
lecture: "1-2"
---

# 2025 → 2026 第 1/2 讲差异

> 你已经完整学过了 **2025 版第 1、2 讲**（Overview & Tokenization;PyTorch, Resource Accounting）。本文档只覆盖 **2026 版第 1、2 讲**中**新的 / 改变的**内容，并显式标注「相同，可跳过」；已掌握的部分不做完整展开，新增内容的讲解则完整深入。

## 总览：2026 第 1、2 讲相对 2025 的整体变化

2026 版是 CS336 的**第三版**（2025 是第二版）。第 1、2 讲的**总体结构没有变**：第 1 讲仍是「课程总览 + 分词（Tokenization）」，第 2 讲仍是「PyTorch 原语 + 资源核算（Resource Accounting）」。两讲的开头叙事、BPE 算法、einops 三件套、$6\times\text{参数}\times\text{token}$ 推导等核心内容与 2025 几乎逐字相同。变化集中在四个方面：

1. **年份刷新**：所有数字、模型、硬件都更新到 2025–2026 年——分词器演示从 GPT-2 换成 **GPT-5(o200k_base)**，前沿模型清单换成 **Kimi K2 / GLM-5 / Minimax M2 / 小米 MIMO V2 / Qwen 3.5 / OLMo 3** 等，硬件语境从 H100 推进到 **B200**，课程计算资源从 Together AI 集群换成 **Modal(B200)**。
2. **课程定位微调**：第三版明确提出「更注重**单位时间的价值密度**(high value-per-time)，别只见树木不见森林」，并新增对 **MoE、long-context、agents** 的覆盖；AI 政策从「风险自负」升级为**强制使用 AGENTS.md**。
3. **第 2 讲内容前移**:2026 把**算术强度（arithmetic intensity）与 roofline 分析**从 GPU 讲座**前置到第 2 讲**，并新增 **FP4/NVFP4 与 NeMo-3 Super**、**梯度累积（gradient accumulation）**、**激活检查点（activation checkpointing）**的完整展开。
4. **现身说法**：第 2 讲开场用 **Marine 项目**「缩放定律预测 loss 与实际训练 loss 只差 0.05」的实验结果作为动机，把"预测要能命中"从口号变成第一方证据。

## 各讲差异

| 页面 | 覆盖范围 | 一句话摘要 |
|------|----------|-----------|
| [第 1 讲差异：Overview, Tokenization](lecture-01-diff.md) | 课程第三版组织变化、前沿模型清单与开放权重分类、**GPT-2 → GPT-5(o200k_base) 分词器演示**、B200/Modal/AGENTS.md 等年份刷新 | 结构零改动，内容整体「年份刷新」；唯一需要动手看的新东西是 GPT-5 分词器及其压缩率 |
| [第 2 讲差异：PyTorch](lecture-02-diff.md) | Marine 0.05 案例、**算术强度/roofline 前置**（完整推导）、**FP4/NVFP4 与 NeMo-3 Super**、einops 讲授方式变化、硬件数字更新、**梯度累积 + 激活检查点**、混合精度代码 | 本讲新增一个完整的「算术强度 + roofline」公式板块与若干显存优化技巧，其余为年份刷新 |

## 快速结论

- **可以直接跳到第 3 讲**。第 1、2 讲是"结构不变、年份刷新"的两讲，你已掌握的 2025 知识（三类知识、苦涩的教训、BPE 全流程、$6\times\text{参数}\times\text{token}$、fp32/fp16/bf16、einops 例子、MFU 定义）在 2026 全部仍然成立。
- **需要重点补看的点**（按优先级):
  1. **GPT-5 分词器演示**:`tiktoken.get_encoding("o200k_base")`，同一字符串压缩率从 GPT-2 的 1.6 提到 **2.5**；词表约 20 万（多语言分词器普遍 100k–200k），理解"压缩率 ↔ 词表大小 ↔ 稀疏性"的权衡。
  2. **前沿模型清单与开放权重分类**:2026 把开放模型重排为「早期 GPT-3 复刻尝试 / 可信开放权重 / 开源（weights+paper+code+data）」三档，并点名 Kimi K2、GLM-5、Minimax M2、Qwen 3.5、OLMo 3 与 Marin 项目。
  3. **第 2 讲的算术强度 / roofline 板块**(2026 新增)：加速器强度 ≈ 295、ReLU 0.25 / 内积 0.5 / matvec ≈ 1 / matmul ≈ n/3，以及「推理 decode 是 memory-bound、训练 matmul 是 compute-bound」的结论与 MFU 的关系。
  4. **FP4/NVFP4 与 NeMo-3 Super**:4 位数值 + 按块缩放，NeMo-3 Super 用 FP4 训练。
  5. **梯度累积与激活检查点**：两个显存优化技巧，第 2 讲首次完整展开。
  6. **Marine 0.05 案例**：缩放定律预测命中的第一方证据。
- **明确可跳过**:BPE 算法与 "the cat in the hat" 推导、字符/字节/词级分词缺陷、einops 的 einsum/reduce/rearrange 例子、$6\times\text{参数}\times\text{token}$ 与反向 2× 推导、fp32/fp16/bf16 位布局、AdaGrad 实现与训练循环——这些两版完全相同。
