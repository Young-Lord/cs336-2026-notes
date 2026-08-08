---
title: "03 Architectures"
lecture: 3
---

# Lecture 3: Everything You Didn't Want to Know About LM Architecture and Hyperparameters

**讲师**: Tatsu Hashimoto · **主题**：语言模型架构与超参数（LM Architecture and Hyperparameters）

## 本讲内容

这一讲的标题是 *Everything You Didn't Want to Know About LM Architecture and Hyperparameters*（关于语言模型架构与超参数，那些你本不想知道的一切）。它的方法论很特别：最好的学习方式是**自己动手训练模型**，但课程既没有算力也没有时间把整个设计空间都遍历一遍，所以退而求其次——**从别人的经验中学习**。Tatsu 把从 2017 年原始 transformer 到 2026 年初的数十个公开语言模型放在一起做"演化分析"，回答三个问题：**这些模型有哪些共同点？哪些部分各有取舍？我们能学到什么？** 随后依次深入三大板块：**常见架构变体**（归一化、激活与 FFN、位置编码）、**真正重要的超参数**(FFN 维度比例、注意力头维度、纵横比、词表大小、正则化)，以及过去一年最受关注的**训练稳定性技巧**(z-loss、QK norm、logit 软封顶)，最后以**注意力头变体**（GQA/MQA、滑动窗口注意力、长短程注意力交错）收尾。

| 页面 | 内容 |
|------|------|
| [03 · 开场与 transformer 回顾](overview.md) | 课程主题"从别人的经验中学习"、原始 transformer 与作业实现的现代变体、近年模型发布的演化分析、本讲路线图 |
| [03 · 归一化：pre-norm、LayerNorm 与 RMSNorm](normalization.md) | pre-norm/post-norm 之争与唯一共识、double norm、LayerNorm 与 RMSNorm、FLOPs ≠ 运行时间、去掉 bias 项 |
| [03 · 激活函数与 FFN：从 ReLU 到 SwiGLU](activations-ffn.md) | ReLU/GeLU 动物园、门控线性单元 GLU/ReGLU/GeGLU/SwiGLU、Shazeer 与 Narang 的证据、serial 与 parallel 层 |
| [03 · 位置编码：RoPE 详解](position-embeddings.md) | sine/absolute/relative 位置编码、RoPE 的动机（只依赖相对位置）、2D 旋转直觉、高维块对角旋转与实现 |
| [03 · 超参数：该选什么、为什么](hyperparameters.md) | FFN 维度比例（4× / 8/3× / 64×）、注意力头维度、纵横比（约 100）、词表大小、dropout 与 weight decay 的真相 |
| [03 · 稳定性技巧与注意力头变体](stability-and-attention.md) | softmax 的两个"雷区"、z-loss、QK norm、logit 软封顶、GQA/MQA 的算术强度分析、稀疏/滑动窗口注意力 |

## 本讲要点

- 本讲主题是**从别人的经验中学习**：通过分析 2017 年以来数十个模型的架构选择，找到"大家都做对的事"和"各有取舍的事";
- 架构要同时满足三件事：**从数据中学习**（表达力）、**在 GPU 上高效训练**（系统效率）、**训练过程中不炸掉**（稳定性)——所有乱七八糟的设计都是为了平衡这三个目标；
- 归一化存在一条**铁律**:**把 LayerNorm 放到残差流之外**（pre-norm）几乎人人遵守——残差流保持"干净"、梯度传播漂亮、训练更稳定；而 **RMSNorm** 与**去掉 bias 项**是共识性的简化（更少参数搬运、更少操作）；
- 一个贯穿始终的系统教训：**FLOPs 不等于运行时间（runtime）**。归一化只占约 0.17% 的 FLOPs，却可能占 25% 的运行时间——因为它的瓶颈是**数据搬运（data movement）**;
- 激活函数方面，**门控线性单元（GLU）**（SwiGLU/GeGLU）自 2023 年以来几乎成为标配，证据指向一致的小幅收益——但它**并非必需**(GPT-3 就不用，**Nemotron 340B** 甚至用了平方 ReLU);
- **RoPE（旋转位置编码)**是位置编码之争的赢家：把"相对位置"编码进向量的**旋转角度**，利用内积的旋转不变性，使注意力分数只依赖相对位置差；它作用在**注意力层**（旋转 Q、K）而不是嵌入底部；
- 超参数存在一些**令人惊讶的共识**:$d_{ff} = 4\,d_{model}$（GLU 变体为 $\frac{8}{3}\,d_{model}$ 左右）、head 维度 × 头数 ≈ 模型维度、纵横比 $d_{model}/n_{layer} \approx 100$;
- **weight decay 不是为了防过拟合**：单遍 SGD 训练下过拟合几乎不发生；它通过与**学习率衰减（余弦调度）**的交互，在训练尾段带来隐式加速，从而获得**更好的训练 loss**;
- 过去一年最受关注的新发展是**训练稳定性技巧**:z-loss（约束输出 softmax 的归一化常数）、QK norm（对 Q、K 做归一化）、logit 软封顶——它们的共同点是**把矛头指向 softmax**（指数运算 + 除法);
- 推理成本的核心瓶颈是**算术强度（arithmetic intensity）**：自回归解码时 KV cache 带来的内存访问让算术强度恶化，$n/d$ 这一项几乎无解，于是有了 **MQA/GQA**（共享 KV 头）与**稀疏/滑动窗口注意力**、以及**全注意力与局部注意力交错**的混合结构；
- 模型间真正主要的**分歧**只剩三处：**位置编码、激活函数、分词（tokenization）**。

## 课程导航

- [上一讲：02 Resource Accounting](../02/)
- [下一讲：04 Attention Alternatives](../04/)
