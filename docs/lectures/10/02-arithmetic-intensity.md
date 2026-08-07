---
title: "10 · 算术强度与推理核算：为什么推理是访存受限的"
lecture: 10
---

# 算术强度与推理核算：为什么推理是访存受限的

这一节是本讲的数学核心。我们要做三件事：先复习**算术强度（arithmetic intensity）**的概念并算一个矩阵乘法的账；然后把它应用到 Transformer 的 MLP 与注意力层，区分 prefill 与 generation 两个阶段；最后用这些结论核算一个真实模型（Llama 2 13B）在 H100 上的吞吐与延迟。

## 复习：矩阵乘法的算术强度

算术强度在第二讲就讲过了，这里快速复习一下。假设我们要把两个矩阵相乘：一个 $B \times D$ 的矩阵 $X$ 和一个 $D \times F$ 的矩阵 $W$。直觉上：$B$ 是 batch 维度，$D$ 是隐藏（模型）维度，$F$ 是 MLP 的 up 投影维度。我们同时数两本账：**FLOPs**（计算量）和**搬动的内存量**。

假设一切以 BF16 存储（每个数 2 字节），整个过程如下：

1. 从 HBM 读 $X$（$B \times D$）：$2BD$ 字节；
2. 从 HBM 读 $W$（$D \times F$）：$2DF$ 字节；
3. 计算 $Y = XW$：$2BDF$ FLOPs（矩阵乘法每输出元素一次乘一次加，共 $BDF$ 次，乘以 2）；
4. 把 $Y$（$B \times F$）写回 HBM：$2BF$ 字节。

```python
import sympy as sp

B, D, F = sp.symbols("B D F", positive=True)
c = sp.symbols("c", positive=True)   # 用于取极限的常数

flops = 2 * B * D * F
bytes_transferred = 2 * B * D + 2 * D * F + 2 * B * F

# 算术强度 = 计算量 / 搬移字节数
intensity = sp.simplify(flops / bytes_transferred)
```

于是

$$\text{FLOPs} = 2BDF, \qquad \text{bytes} = 2BD + 2DF + 2BF$$

**算术强度**定义为每搬移一个字节做了多少计算：

$$\text{intensity} = \frac{\text{FLOPs}}{\text{bytes}} = \frac{2BDF}{2BD + 2DF + 2BF}$$

我们希望它尽量高。注意分子的量级是"三次方"，而分母是"二次方"——矩阵乘法之所以能获得高算术强度，靠的就是这个立方 vs 平方的结构。如果 batch 维度 $B$ 远小于 $D$ 和 $F$，上式可以简化。用极限语言：令 $D = cB$、$F = cB$，让 $c \to \infty$（即 $D,F \gg B$）：

```python
# 令 D = cB, F = cB, 取 c → ∞
intensity_limit = sp.limit(intensity.subs(D, c * B).subs(F, c * B), c, sp.oo)
# => B
```

极限下分母中 $2DF = 2c^2B^2$ 这一项占主导，得到

$$\text{intensity} \longrightarrow B$$

也就是说，**这个矩阵乘法的算术强度（约）等于 batch 大小 $B$**。这与第二讲里"全连接层算术强度约 $N/3$"的结论是一回事——那里是方阵情形，这里是长方形矩阵的推广。

### 与硬件比较：Roofline

接下来把计算强度与硬件的"加速器强度（accelerator intensity）"比较。加速器强度就是把芯片的峰值算力除以内存带宽：

```python
flops_per_second = 989e12      # H100 峰值 FLOPs/秒
memory_bandwidth = 3.35e12     # H100 内存带宽 字节/秒
accelerator_intensity = flops_per_second / memory_bandwidth
# ≈ 295
```

H100 的加速器强度约为 $989 \times 10^{12} / 3.35 \times 10^{12} \approx 295$。于是 Roofline 判断：

- 若**计算强度 > 加速器强度**，则**计算受限（compute-bound）**——好事，芯片的计算单元是瓶颈；
- 若**计算强度 < 加速器强度**，则**访存受限（memory-bound）**——坏事，芯片在等着数据从 HBM 搬进来。

对本例，计算受限当且仅当

$$B > 295$$

### 极端情形：$B = 1$

考虑一个极端情形：只有一个样本，即 $B = 1$，这对应于**矩阵-向量乘积**。此时算术强度约为 $1$——你把整个 $D \times F$ 的矩阵读进来，却只做了 $2DF$ 次 FLOPs，每个字节平均只做约一次计算，远远达不到 H100 需要的 295。这必然是访存受限。

> **"这就是推理时的情形。"** 推理时你拿不到那些"厚"的矩阵，只能拿到很"薄"的张量。

记住这一点，下面分析推理时我们会反复遇到它。

## 推理的算术强度：prefill 与 generation

### 朴素推理为什么是 $O(T^3)$

先看最朴素的做法。假设我们有一个黑盒 Transformer，输入一个序列、输出下一个 token 的分布。要生成响应，就反复应用它：把 prompt "Never gonna give you" 喂进去，得到 logits、采样一个 token；把 token 拼回 prompt，再喂一次，得到下一个 token……

这个朴素方案能工作，但效率极差：每次生成一个 token，都要把**当前整个历史**重新过一遍 Transformer，成本是 $O(T^2)$（注意力部分就是 $T^2$ 量级）；生成 $T$ 个 token 总成本为

$$\sum_{t=1}^{T} O(t^2) = O(T^3)$$

这非常糟糕。但关键观察是：**很多工作其实可以在前缀之间共享**。例如无论你在生成 "going to" 还是 "give you"，前缀 "Never gonna give you" 对应的 key、value 都是一样的，因为这些 token 不会改变。这是由**因果（causal）注意力**保证的：如果是双向注意力，追加一个 token 会改变一切；但对因果 Transformer，前面 token 的激活不会因为你追加了内容而改变。

于是第一个显然的改进是：把算好的 key-value 对存进**KV cache**，这样两次相邻的生成之间可以直接复用它们。生成 "going to" 时，你完全不需要重算前面那些 token 的 key 与激活。

### 两阶段：prefill 与 generation

有了 KV cache，推理分为两个阶段：

1. **prefill（预填充）**：给定 prompt，把它编码成向量、填满 KV cache。这一步和训练一样**可以沿序列并行**——因为你能看到整个 prompt，可以一次算出全部 KV cache；
2. **generation（生成）**：逐 token 顺序地生成新的响应 token。至少你不必为已经看过的 token 重新付 KV cache 的代价。

形式上，KV cache 是：对每个序列（共 $B$ 个）、每个 token（共 $S$ 个）、每一层（共 $L$ 层）、每个 KV 头（共 $K$ 个），存一个 $H$ 维向量。

![推理的两个阶段：prefill 并行编码 prompt，decode 逐 token 生成](/lectures/10/prefill-decode.png)

下面我们用符号系统计算 MLP 层和注意力层的 FLOPs 与访存量。约定：$S$ 是条件 token 数（已经生成的、存进 KV cache 的 token），$T$ 是要生成 logits 的 token 数。后面我们会分别特化为 prefill（$T = S$）与 generation（$T = 1$）。

### MLP 层的账

MLP 层只算矩阵乘法部分（其余操作 FLOPs 很少，而且可以融合进 matmul）。流程与刚才的 matmul 记账完全同构：

1. 从 HBM 读 $X$（$B \times T \times D$）：$2BTD$ 字节；
2. 从 HBM 读 $W_{\text{up}}$（$D \times F$）、$W_{\text{gate}}$（$D \times F$）、$W_{\text{down}}$（$F \times D$）：$3 \times 2DF$ 字节；
3. 计算 $U = XW_{\text{up}}$：$2BTDF$ FLOPs，写回 $U$（$B \times T \times F$）：$2BTF$ 字节；
4. 计算 $G = XW_{\text{gate}}$：$2BTDF$ FLOPs，写回 $G$（$B \times T \times F$）：$2BTF$ 字节；
5. 计算 $Y = \text{GeLU}(G) \cdot U \cdot W_{\text{down}}$：$2BTDF$ FLOPs，写回 $Y$（$B \times T \times D$）：$2BTD$ 字节。

```python
flops = 6 * B * T * D * F
bytes_transferred = 4 * B * T * D + 4 * B * T * F + 6 * D * F

intensity = sp.simplify(flops / bytes_transferred)
# 令 D = cBT, F = cBT, 取 c → ∞
intensity = sp.limit(intensity.subs(D, c * B * T).subs(F, c * B * T), c, sp.oo)
# => B*T
```

与刚才一样，令 $D, F \gg BT$，得到

$$\text{intensity}_{\text{MLP}} = \frac{6BTDF}{4BTD + 4BTF + 6DF} \longrightarrow BT$$

结论和单个 matmul 的情形完全同构：**MLP 的算术强度约等于 $B \cdot T$**。这很自然——MLP 本质上就是一大块 matmul，而且 batch 维与序列维互不干扰、独立贡献。

对两个阶段：

- **prefill**：$B \cdot T$ 很容易做大（大 batch、长序列），prefill 阶段容易做成计算受限——这是好事；
- **generation**：有两个问题。一是 $T = 1$（一次只生成一个 token），算术强度退化为 $B$；二是 $B$ 是**并发请求数**，对交互式应用（比如聊天机器人）来说就是并发用户数，可高可低、随时间波动、难以预测。这一点我们讲到连续批处理（continuous batching）时再处理。总的来说 generation 的 MLP 还好：只要有足够大的 batch，序列长度帮不上忙，但大 batch 就能让你满足。

### 注意力层的账

注意力层是另一个故事。同样记 $S$ 为已生成（已缓存）的 token 数、$T$ 为要生成 logits 的 token 数，以 FlashAttention 的视角只看矩阵乘法：

1. 从 HBM 读 $Q$（$B \times T \times D$）、$K$（$B \times S \times D$）、$V$（$B \times S \times D$）：$2BTD + 2BSD + 2BSD$ 字节；
2. 计算 $A = QK^{\top}$：$2BSTD$ FLOPs；
3. 计算 $Y = \text{softmax}(A) V$：$2BSTD$ FLOPs；
4. 把 $Y$（$B \times T \times D$）写回 HBM：$2BTD$ 字节。

（softmax 本身 FLOPs 很少，忽略不计。）

```python
flops = 4 * B * S * T * D
bytes_transferred = 4 * B * S * D + 4 * B * T * D

intensity = sp.simplify(flops / bytes_transferred)
# => S*T / (S + T)
```

于是

$$\text{intensity}_{\text{attention}} = \frac{4BSTD}{4BSD + 4BTD} = \frac{ST}{S + T}$$

因为注意力里的一切都是 matmul，所以 FLOPs 总是比访存量高一阶多项式——唯一的悬念是系数因子长什么样。这里得到的因子正是 $ST/(S+T)$。

### prefill 与 generation 的注意力强度

特化到两个阶段：

- **prefill（$T = S$）**：

$$\text{intensity}_{\text{prefill}} = \frac{S \cdot S}{S + S} = \frac{S}{2}$$

很好！只要序列足够长，prefill 的注意力也能维持高算术强度。注意这里**没有出现 batch 维 $B$**——为什么，一会儿解释。

- **generation（$T = 1$）**：

$$\text{intensity}_{\text{gen}} = \frac{S \cdot 1}{S + 1} = \frac{S}{S+1} < 1$$

坏消息：算术强度小于 1。而我们希望的是像 295 那样去喂饱 H100。**这才是推理真正的瓶颈。**

### 为什么注意力不受益于 batching

我们做了 MLP 和注意力、prefill 和 generation 的全部核算，发现注意力 generation 是瓶颈。为什么？对比一下：

- **MLP 层**：每个序列命中**相同的 MLP 权重**（$W_{\text{up}}, W_{\text{gate}}, W_{\text{down}}$ 不依赖 $B$）。$B$ 大是好事——把权重读进内存一次，就能给所有序列复用，算术强度因此提高：load 一次、处理整个 batch；
- **注意力层**：每个序列有**自己的 KV cache 向量**（$Q, K, V$ 全都依赖 $B$）。增大 $B$ 只是并行了更多互相独立的小 matmul，本质上是做了一堆点积——还记得开头那个 $BD \times BD \to B$ 的例子吗？那就是**按坐标批量化的点积**，算术强度极差。注意里那个蓝色的 $B$ 正是"注意力算术强度不随 $B$ 缩放"的原因，也是它成为瓶颈的原因。

总结下来：

| 阶段 | MLP 强度 | 注意力强度 |
|------|------|------|
| prefill | $B \cdot S$（很好） | $S/2$（不错，可接受） |
| generation | $B$（需要并发请求） | $S/(S+1) < 1$（**根本性瓶颈**） |

> **prefill 是计算受限的，generation 是访存受限的。** 只要你还用 Transformer，generation 注意力的算术强度就几乎无法改善——这是架构决定的。所以当人们说"推理是访存受限的"，你现在知道为什么了。

## 吞吐与延迟：Llama 2 13B 的核算

既然推理是访存受限的，很多事情就简化了：**做一件事要花多长时间，主要看要搬动多少内存**——只要通信与计算能完美重叠（overlap），瓶颈就是"必须处理的字节数"。这既是好消息（计算更简单）也是坏消息（你的加速器常常闲在那里）。

下面把上面的公式实例化到 **Llama 2 13B 跑在 H100** 上。Llama 2 13B 的形状参数为：序列长度 $S = 1024$、模型维度 $D = 5120$、前馈维度 $F = 13824$、query 头数 $N = 40$、KV 头数 $K = 40$（这里没有 GQA，$N = K$）、头维度 $H = 128$、层数 $L = 40$、词表大小 $V = 32000$，内存带宽取 H100 的 $3.35 \times 10^{12}$ 字节/秒。

先看需要哪些统计量：参数量、内存占用（参数 + KV cache）、延迟、吞吐。逐项推导：

**参数量**。一个 Transformer 的参数分三块：embedding（词嵌入与输出层，$2VD$）、每层的 MLP（三个矩阵，$3DF$）、每层的注意力投影（Q 投影 $D \times NH$，K、V 投影各 $D \times KH$，共 $(2DNH + 2DKH)$）：

```python
num_params = 2 * V * D + 3 * D * F * L + (2 * D * N * H + 2 * D * K * H) * L

parameter_size = 2 * num_params   # BF16，每个参数 2 字节
```

代入配置，$2 \cdot 32000 \cdot 5120 = 3.28 \times 10^8$（embedding）、$3 \cdot 5120 \cdot 13824 \cdot 40 \approx 8.49 \times 10^9$（MLP）、$(2 \cdot 5120 \cdot 40 \cdot 128 + 2 \cdot 5120 \cdot 40 \cdot 128) \cdot 40 \approx 4.19 \times 10^9$（注意力投影），加起来约 **$1.30 \times 10^{10}$（130 亿）个参数**——和"13B"的宣传对得上。BF16 下参数占内存约 **26 GB**。

**KV cache 大小**。对每个序列：$S$ 个 token、每个 token 有 $K$ 个 KV 头、每个头是 $H$ 维向量、共 $L$ 层，key 和 value 各一份，BF16 每数 2 字节：

```python
kv_cache_size_per_seq = S * (K * H) * L * 2 * 2   # ×2 是 K+V, ×2 是 BF16
```

代入：$1024 \cdot 5120 \cdot 40 \cdot 4 \approx 8.39 \times 10^8$ 字节 ≈ **每个序列约 839 MB**。注意这只是一个序列！整个内存占用是

```python
memory = B * kv_cache_size_per_seq + parameter_size
```

即 $B$ 个序列的 KV cache 加上参数。**延迟**由访存决定，**吞吐**是 $B$ 个 token 并行产生：

```python
latency = memory / memory_bandwidth       # 秒/token
throughput = B / latency                  # token/秒
```

**延迟是 $B$ 的线性函数**（斜率是每个序列的 KV cache 大小，截距是参数量）；**吞吐**则是 $B / \text{latency}$，随 $B$ 增大而增长、但趋向渐近线——吞吐不可能无限增大。

### 数值结果

把 $B$ 代进几个典型值：

| batch $B$ | 显存占用 | 延迟 | 吞吐 |
|------|------|------|------|
| 1 | 26.9 GB | 8.0 ms/token | 124.7 token/s |
| 64 | 79.7 GB | 23.8 ms/token | 2,689 token/s |
| 256 | 240.8 GB（**超过 H100 的 80 GB**） | 71.9 ms/token | 3,561 token/s |

几个观察：

- **增大 batch，延迟变差、吞吐变好**。延迟变差是因为 KV cache 随 $B$ 线性增长，每生成一个 token 都要把它读进读出；吞吐变好是因为参数对所有序列共享——把参数读进内存一次，就能处理一大批序列。
- 但吞吐的收益是**递减**的（$B/\text{latency}$ 有渐近线），而且你会先撞上**显存墙**：$B = 256$ 时需要约 241 GB，放不进 80 GB 的 H100；就算换成显存更大的 B200，batch 可以再往上加，但终究有极限，你永远到不了那个渐近线。
- 所以 batch 大小是一个真正的**权衡旋钮**：想要低延迟就用小 batch（单个用户的等待短），想要高吞吐就用大 batch（批量处理任务的完成时间短）。"快"在这里有互相矛盾的两个含义——这正是课上说的"延迟与吞吐之间存在张力"。

课上用了一个**公交车**的比喻帮助理解这个权衡：增加 batch 就像让每个人去等一趟大巴。batch 大了，每个用户（个人查询）得等全车人上齐才发车，所以延迟变高；但大巴一趟能运走所有人，吞吐很好。小 batch 就像私家车——随叫随走、延迟低，但一趟只能拉一两个人、吞吐差。

> **一句话总结**：小 batch → 更好的延迟、更差的吞吐；大 batch → 更好的吞吐、更差的延迟。之所以有这种张力，是因为 KV cache 随 $B$ 线性增长，而参数随 batch 被摊薄。

### 并行化与 TTFT

还有一个维度是并行化：如果在 $M$ 台设备上各放一份完整的模型副本，延迟不变、吞吐变为 $M$ 倍（简单情形）；把模型与 KV cache 本身切分到多设备上则更复杂，感兴趣的可以去读 Scaling Book 的推理章节。

最后，另一个度量 **首 token 时间（TTFT）** 本质上就是 **prefill 的时间**——prefill 一结束就可以开始生成。所以：想要更快的 TTFT，prefill 用**小 batch**；想要更高的吞吐，generation 用**大 batch**。

到这里，我们有了分析推理效率的概念框架（算术强度、吞吐、延迟）。接下来进入正题：**怎么让推理更快**。

<!-- lecture-nav -->

**← 上一节**：[推理工作负载概览：重要性、度量与 Transformer 记法](01-inference-overview.md)　**→ 下一节**：[减小 KV 缓存：GQA、MLA、CLA 与局部注意力](03-reducing-kv-cache.md)
