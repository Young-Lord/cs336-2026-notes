---
title: 02 · 降低内存的技巧与总结
lecture: 2
---

# 降低内存的两种技巧与总结

上一节的内存分解里，激活这一项（$2BDL$）会随 batch size 线性增长。这一节讲两个“用计算换内存”的经典技巧：**梯度累积**（gradient accumulation）与**激活检查点（activation checkpointing）**，最后总结整讲。

## 为什么想上更大的 batch

一般来说，我们想让 batch size **足够大**，因为大 batch 能**改善训练稳定性**——存在一个**临界 batch size(critical batch size)**，超过它之后再增大收益递减（Tatsu 后面会详细讲）。

但问题在于：**激活内存随 batch size 增大**：

$$\text{激活内存} = 2 B D L \ \text{字节}$$

batch 一大，显存就可能爆掉（OOM）。怎么既用大 batch，又不爆内存？

## 技巧一：梯度累积（Gradient Accumulation）

思路非常朴素：与其一次前向/反向一个大 batch，不如把它拆成几个 **micro-batch（微批）**：

- 在 **micro-batch** 上计算梯度；
- **累积**这些梯度，**不清零**；
- 每 **batch_size / micro_batch_size** 步，才**更新一次参数并清零梯度**。

伪代码大概是：

```python
for step in range(num_train_steps):
    x, y = get_batch()          # micro-batch
    loss = model(x).mean()      # 前向
    loss.backward()             # 反向,梯度累积到 .grad,不清零
    if (step + 1) % accum_steps == 0:
        optimizer.step()        # 更新参数
        optimizer.zero_grad()   # 清零梯度
```

这只是一个**非常简单的代码改动**，却能省下大量激活内存——因为每次前向/反向只在 micro-batch 上做，激活只按 micro-batch 大小分配。幻灯片里的账：batch 64 时激活 $2 \times 64 \times D \times L$；拆成 4 个 micro-batch 后，激活降到 $2 \times 16 \times D \times L$，省了 4 倍。

## 技巧二：激活检查点（Activation Checkpointing）

### 训练要存所有层的激活，推理不用

回顾一下：

- **训练**需要所有层的激活来算梯度，内存是 $2BDL$；
- **推理**不需要梯度，只需要**当前层**的激活。

问题来了：训练时能不能少存点激活？

### 核心思想：重物化（rematerialization）

**激活检查点**（也叫 **gradient checkpointing（梯度检查点）** 或 **rematerialization（重物化）**）的关键想法是：

- **前向**时，只在**部分层（检查点）保留激活**；
- **反向**时，从**最近的一个检查点重新计算**缺失的激活。

所以叫“检查点（checkpointing）”。这是一个很通用的系统技巧：**想省内存？那就重新计算**——用**内存换计算**。

![](/lectures/02/deep-network.png)

### 实现：一行代码

在 PyTorch 里用 `torch.utils.checkpoint` 包一层就行：

```python
class DeepNetworkCheckpointed(nn.Module):
    def forward(self, x):
        for layer in self.layers:
            # KEY:只存检查点处的激活,其余重算
            x = torch.utils.checkpoint.checkpoint(layer, x)
        return x
```

意思是：“做这个计算，但**不要存中间激活**，只存必要的东西。”在我们的 deep network 里，每个 block 有 ReLU 之前的量（pre-ReLU）和之后的量（post-ReLU）——如果不做检查点，两者都要存；做了检查点，pre-ReLU 不存，反向时从检查点轻松重算，大约能**省一半**内存。

### 检查点间隔怎么选？

对深网络（大 $L$），检查点到底该多密？三种极端：

- **每一层都存**：激活内存 $O(L)$，无需重算；
- **一层都不存**：内存 $O(1)$，但计算变成 $O(L^2)$——每一层都要从最开头重算；
- **每隔 $\sqrt{L}$ 层存一个**：激活内存 $O(\sqrt{L})$，重算开销也 $O(\sqrt{L})$。

**每隔 $\sqrt{L}$ 层存一个检查点，是甜点（sweet spot）**：内存和计算开销都控制在 $O(\sqrt{L})$。幻灯片里的直观示意：全部存（每个 h 都留）、全不存（空）、每隔几个存（h3、h6、h9……）。

## 总结本讲

最后，Percy 把整讲浓缩成几条：

- **一切都是对 tensor 的操作**：参数、梯度、激活、优化器状态、数据——本质上全是 tensor；
- **einops** 提供了一种更好的思考 tensor 操作的方式（命名维度，einsum/reduce/rearrange）；
- 每个训练步的 FLOPs ≈ **6 × （#数据点） × （#参数）**（如果这是训练步，这里的“数据点”就是 batch size）——现在我们已经完全理解了它从哪来；
- **算术强度 / roofline 分析**让我们能诊断一次计算是 **compute-bound 还是 memory-bound**；
- **矩阵乘法是 compute-bound，基本上其它一切都是 memory-bound**；
- **梯度累积**与**激活检查点**是用更多计算换更少内存，从而能**上更大的 batch**。

今天的课就到这里。下一讲 Tatsu 会继续讲架构与超参数。

---

<!-- lecture-nav -->

**→ 下一讲**:[03 Architectures](../03/)
