---
title: "07 · 集合通信原语：collectives 的编程模型"
lecture: 7
---

# 集合通信原语：collectives 的编程模型

好，我们切入正题。第一件要讲的东西叫**集合通信（collective operation）**——分布式编程里最基本的原语。

## 什么是 collective

**collective** 是分布式编程里的原语（primitive），其历史可以追溯到 **1980 年代**。并行编程的想法非常古老，它并不是为训练语言模型发明的；但到今天，我们训练大模型用的依然还是这些原语。

"collective"（集合的）这个词的意思是：你指定的是**一种跨多个设备的通用通信模式（或模板）**，而不是手动管理"这块 GPU 怎么和那块 GPU 点对点（point-to-point）通信"。好处显而易见：写起来**容易得多**，而且**系统能替你干很多活**。这是一套久经考验（tried and true）的并行编程接口。

## 基本设定：rank 与世界大小

先交代术语。这套术语我本人觉得有点怪，但它是并行编程里的标准说法。基本设定如下：

![](/lectures/07/ranks.png)

- **rank**：一个特定的设备——对我们来说是 GPU（也可以是 TPU）。图里有四个 rank，编号 0、1、2、3；
- **世界大小（world size）**：设备的总数。这里的 world size 就是 **4**。

接下来我们要过一遍几种操作。broadcast、scatter、gather、reduce 在我看来只是**热身**——它们能帮你建立对 collective 的直觉，但并不是训练的主力；**all-gather、reduce-scatter、all-reduce** 才是大模型分布式训练里反复出现的工作马（workhorse）；最后提一下 **all-to-all**，它对 **MoE（混合专家）**很重要，但本讲不打算花太多时间。

## Broadcast：从 rank 0 复制给所有 rank

最简单的是 **broadcast（广播）**。假设 rank 0（任意一个 rank 都行，这里就选它）持有一个张量 $[0, 1, 2, 3]$，广播给所有 rank。操作结束时，**每个 rank 上都有同样的张量**：

```python
# 输入：只有 rank 0 持有数据
rank0 = tensor([0., 1, 2, 3])

# 输出：所有 rank 都持有同一份数据
rank0 = tensor([0., 1, 2, 3])
rank1 = tensor([0., 1, 2, 3])
rank2 = tensor([0., 1, 2, 3])
rank3 = tensor([0., 1, 2, 3])
```

broadcast 一般不出现于训练的主路径上，它主要用于**初始化**：比如 rank 0 加载好初始 checkpoint，然后把参数广播给所有 rank——这种事做一次就够了。

## Scatter：把一个张量散开

**scatter（散开）**说的是：rank 0 持有一个张量，把它按 world size **切成几块**，然后分发到各个 rank 上。于是 rank 0 拿到第 0 个分量，rank 1 拿到第 1 个分量，rank 2、rank 3 依次拿到各自的分量：

```python
# 输入：rank 0 持有完整张量
rank0 = tensor([0., 1, 2, 3])

# 输出：每个 rank 拿到一个分量
rank0 = tensor([0.])
rank1 = tensor([1.])
rank2 = tensor([2.])
rank3 = tensor([3.])
```

scatter 本身也不直接用于训练，但它是理解 **reduce-scatter** 的垫脚石。正如名字所示，scatter 就是把一个"大张量"放在一个地方、然后摊到多个地方；你大概能想象它为什么有用——你想让所有 GPU 各自在**不同的数据**上做局部计算。

## Gather：scatter 的逆操作

**gather（汇聚）**是 scatter 的逆，应该很好理解。输入是一堆分片，每个分片住在各自的 rank 上；对某个目标 rank（比如 rank 0）做 gather，就是把这些分片**拼接**起来：

```python
# 输入：每个 rank 持有自己的分片
rank0 = tensor([0.])
rank1 = tensor([1.])
rank2 = tensor([2.])
rank3 = tensor([3.])

# 输出：rank 0 拿到拼接后的完整张量
rank0 = tensor([0., 1, 2, 3])
```

同样地，gather 不直接用于训练，但它是理解 **all-gather** 的垫脚石。

## Reduce：把各处的数据归约到 rank 0

做过函数式编程的同学应该对 **reduce（归约）**很熟，这里的语义完全一样。它的输入起点和 gather 一样——每个 rank 上有一份数据；然后你把某种**归约运算**应用在所有数据上，把结果放到 rank 0。比如用求和（sum）归约 $[0, 1, 2, 3]$ 这四个数，就得到 $6$：

```python
# 输入：每个 rank 持有自己的分量
rank0 = tensor([0.])
rank1 = tensor([1.])
rank2 = tensor([2.])
rank3 = tensor([3.])

# 输出：rank 0 持有归约结果（0 + 1 + 2 + 3 = 6）
rank0 = tensor([6.])
```

顺带一提，你可以把 **gather 看作一种 reduce**——只不过它的"归约运算"是**拼接（concatenation）**。当然，reduce 是理解 all-reduce 的钥匙。

先在这儿停一下——前面这四个（scatter、gather、reduce）只是热身，方便大家理解 collective 是什么。

### 课堂问答

**问答：这与 NumPy 里的广播（broadcasting）有关系吗？**

（同学：collective 的 broadcast 和 NumPy 的 broadcasting 是同一个概念吗？）

**Percy**：我觉得它们在概念上是相通的——都是"一个东西扩散到许多东西"。NumPy 里一个标量会被广播成一个张量。但具体实现（instantiation）不一样：这里的 broadcast 是**集合通信**意义上的，所以还是有点区别。

## All-gather：对每个 rank 都做一次 gather

**all-gather** 就是**对所有的 rank 执行 gather**，而不只是 rank 0。回顾一下 gather 做什么：它把所有分片汇聚到**一个** rank 上；all-gather 则对**每一个** rank 都做这件事。"all"这个前缀的意思就是：**把输出送到所有设备**。

```python
# 输入：每个 rank 持有自己的分片
rank0 = tensor([0.])
rank1 = tensor([1.])
rank2 = tensor([2.])
rank3 = tensor([3.])

# 输出：每个 rank 都持有拼接后的完整张量
rank0 = tensor([0., 1, 2, 3])
rank1 = tensor([0., 1, 2, 3])
rank2 = tensor([0., 1, 2, 3])
rank3 = tensor([0., 1, 2, 3])
```

all-gather 后面会频繁出现。现在不需要把这个说法抠得很精确，但预告一下：**每个 rank 只持有参数的一部分（分片），做完整前向（forward）前需要 all-gather 把参数拼成完整的一份**。一般地说，训练过程中我们会看到大量这种"gather 做点事、scatter、再 gather、再 scatter"的来回。

## Reduce-scatter：对每个维度归约，再把结果散开

**reduce-scatter** 是：对张量的**每个分量**做归约，然后把结果**散开**到不同的 rank 上。举个例子：假设有四个设备，每个设备持有一个向量。之前做 reduce 时，$[0,1,2,3]$ 被归约成 $6$；而 reduce-scatter 会说：**对第一个分量**，把这些数加起来得 $6$；**对第二个分量**，加起来得 $10$；对第三、第四个分量同样处理——然后把不同的结果放到不同的 rank 上：

```python
# 输入：每个 rank 持有一个向量
rank0 = tensor([0., 1, 2, 3])
rank1 = tensor([1., 2, 3, 4])
rank2 = tensor([2., 3, 4, 5])
rank3 = tensor([3., 4, 5, 6])

# 输出：第 i 个分量在所有 rank 上求和，结果放到 rank i
rank0 = tensor([6.])   # 第 0 维之和：0 + 1 + 2 + 3
rank1 = tensor([10.])  # 第 1 维之和：1 + 2 + 3 + 4
rank2 = tensor([14.])  # 第 2 维之和：2 + 3 + 4 + 5
rank3 = tensor([18.])  # 第 3 维之和：3 + 4 + 5 + 6
```

先剧透一下它的用武之地：**反向传播（backward pass）之后**，每块 GPU 处理的是不同的数据，各自算出的**梯度**不同；你需要把所有分片上的梯度**求和**，然后把这个归约结果**分散存储**（而不是像 all-reduce 那样复制给每块卡）——这正是 reduce-scatter 干的事。

## All-reduce：reduce-scatter + all-gather

把上面两件事拼起来就得到 **all-reduce**：**每个分量都做归约，而且归约结果复制给所有 rank**。还是刚才那个例子，all-reduce 的输入输出是：

```python
# 输入：每个 rank 持有一个向量
rank0 = tensor([0., 1, 2, 3])
rank1 = tensor([1., 2, 3, 4])
rank2 = tensor([2., 3, 4, 5])
rank3 = tensor([3., 4, 5, 6])

# 输出：每个分量在所有 rank 上求和，且结果复制到所有 rank
rank0 = tensor([6., 10, 14, 18])
rank1 = tensor([6., 10, 14, 18])
rank2 = tensor([6., 10, 14, 18])
rank3 = tensor([6., 10, 14, 18])
```

也就是说，**all-reduce = reduce-scatter + all-gather**。它的用武之地是：**反向传播之后，把所有数据分片上算出的梯度求和，并让每个 rank 都持有完整的梯度**——这正是数据并行里同步梯度的方式。而**把 all-reduce 拆成 reduce-scatter 与 all-gather 两部分**是有讲究的：拆分之后就有了灵活性，比如 ZeRO/FSDP 可以只做 reduce-scatter（梯度按分片存放），从而省掉每卡复制完整梯度那份显存。

## All-to-all：最一般的通信模式

最后是 **all-to-all**：**每个 rank 给其他每个 rank 都发送某个张量**，这是最一般（most general）的通信模式。示意图如下：

```python
# 输入：rank i 的第 j 个元素发给 rank j
rank0 = tensor([0., 1, 2, 3])      # 把 0 给 rank0，1 给 rank1，2 给 rank2，3 给 rank3
rank1 = tensor([4., 5, 6, 7])      # 把 4 给 rank0，5 给 rank1，6 给 rank2，7 给 rank3
rank2 = tensor([8., 9, 10, 11])    # 把 8 给 rank0，9 给 rank1，10 给 rank2，11 给 rank3
rank3 = tensor([12., 13, 14, 15])  # 把 12 给 rank0，13 给 rank1，14 给 rank2，15 给 rank3

# 输出：rank j 收到所有 rank 发来的第 j 个分量
rank0 = tensor([0, 4, 8, 12])
rank1 = tensor([1, 5, 9, 13])
rank2 = tensor([2, 6, 10, 14])
rank3 = tensor([3, 7, 11, 15])
```

关于 all-to-all 有几点值得一提：

- **它服务于 MoE**：在 MoE 里，每个 rank 持有一部分数据、一部分专家；根据路由结果，数据要被送到持有对应专家的 rank 上去算——这种"把数据路由给专家"的操作正是 all-to-all；
- 如果分割是**均衡（balanced）**的，把每行看作矩阵的一行，那么 all-to-all 做的事本质上就是**矩阵转置（transpose）**——把输入矩阵转置一下，每个 rank 拿到的恰好是转置后的某一列；
- 它也支持**不均衡（unbalanced）**的分割——你可以配置给任意 rank 发任意数量的字节；不过一般希望分割**尽量均衡**。还记得讲 MoE 那讲里我们做**负载均衡（load balancing）**、让事情尽量平衡吗？道德上，理想的 all-to-all 就应该长上面这个样子。

## 术语记忆法

过完这么多操作，给点帮助记忆的小贴士：

- **reduce**：做某种**可结合、可交换**的运算——sum、max、min 都行；
- **scatter 是 gather 的逆**：scatter 分发（distribute），gather 汇聚（centralize）；
- **all**：意味着**目的地是所有设备**——这就解释了 all-reduce 和 all-gather 为什么带 "all"。

### 课堂问答：关于目标 rank 与代码实现

**问答：gather/reduce 的目标 rank 固定是 rank 0 吗？**

（同学：做 gather、reduce 这类"把小块汇聚到 rank 0"的操作时，rank 0 是特别指定的吗，还是可以换？）

**Percy**：写输出写到哪个 rank 由你在调用时指定。现在我说的是 rank 0，只是举例子；到代码部分你会看到，本质上就是指定那个 GPU 的编号（rank）。它不需要提前很久就定死，但必须在**执行这次调用时**确定下来。

**问答：这些只是概念性的积木，还是真的能跑的东西？**

（同学：这些 collective 只是概念积木，还是实际存在的代码？）

**Percy**：现在我只是把它们当作概念积木来展示，但很快我们就会看到它们在代码里怎么实现。

<!-- lecture-nav -->

**← 上一节**：[开场与动机：为什么需要多 GPU 并行]（overview.md）　**→ 下一节**：[硬件：GPU 如何连接](hardware.md)
