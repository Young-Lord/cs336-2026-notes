---
title: 01 · Byte Pair Encoding(BPE)
lecture: 1
---

# Byte Pair Encoding(BPE)

## 从数据压缩到语言模型

tokenizer 把文本变成 token 序列，而 transformer 就在这些 token 上建模，这正是下面这张原始 transformer 架构图所展示的整体框架：输入先经过 tokenization 变成索引序列，再进入嵌入层与多头注意力等模块。

![](/lectures/01/transformer-architecture.png)

**Byte Pair Encoding(BPE)** 是当前最主流的 tokenization 算法。它的历史坐标很清晰：**Philip Gage 在 1994 年**就为了数据压缩提出了 BPE——远早于语言模型登上舞台；2016 年，它被引入自然语言处理领域，用于**神经机器翻译**（在此之前，论文用的都是词级 tokenization）；而 **GPT-2 是第一个把 BPE 用于语言模型的**。这三件事合起来，就是 BPE 从数据压缩走进语言模型的完整脉络。

基本思想是：**在原始文本上"训练"tokenizer，构造一个贴合数据的词表**。直觉可以概括为：**把输入切分为频繁出现的块**——**常见的字节序列用一个 token 表示，罕见的序列则拆成多个 token**。而且，罕见内容会被逐级拆开成更小的单元，而不是变成 UNK token。这就是 BPE 的核心思想。

## 训练 tokenizer：反复合并最频繁的相邻对

算法在概念上很简单。假设你的语料是一长段文本（不妨假设它是一个长字节序列）：**每个字节最初都是一个 token**；然后我们**反复合并"相邻 token 对"中出现次数最多的那一对**。每合并一次，就产生一个新 token，词表也随之扩大。

Percy 用 `"the cat in the hat"` 这个字符串带着大家把算法走了一遍：

1. 先把字符串转成字节序列；
2. **统计相邻 token 对的出现次数**——比如 `(116, 104)`（即 `t` 和 `h` 组成的 `"th"`）出现了两次，是最频繁的一对（有几个并列，取第一个）；
3. **合并这一对**：创建一个新 token **256** 来代表 `"th"`，加入词表；然后把序列中所有 `(116, 104)` 都替换成 256；
4. **迭代**：下一次，把 256 与 101（`e`）合并成 **257**，再合并一次得到 **258**。随着迭代，序列不断变短，词表不断变大。

值得注意的是，每完成一次合并，序列里出现该 pair 的地方就少一处，整个序列在逐步缩短，而词表在逐步变大——正是这个趋势带来了 BPE 的压缩收益。

对应的代码结构大致是（训练函数 `train_bpe`，输入可以是任意字符串，`num_merges` 控制合并轮数，每轮找出当前最频繁的相邻对并合并）：

```python
indices = list(map(int, string.encode("utf-8")))   # 从字符串的字节序列开始
merges: dict[tuple[int, int], int] = {}            # (index1, index2) => 合并后的新 index
vocab:  dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index => bytes

for i in range(num_merges):
    counts = count_adjacent_pairs(indices)   # 统计每个相邻 token 对的出现次数
    pair = max(counts, key=counts.get)       # 找出最频繁的一对
    new_index = 256 + i                      # 新 token 编号从 256 开始
    merges[pair] = new_index
    vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]
    indices = merge(indices, pair, new_index)
```

其中 `count_adjacent_pairs` 用 `zip(indices, indices[1:])` 把相邻对配对计数，`merge` 则扫描序列、把所有命中 `pair` 的位置替换成新索引。这个玩具例子的最终压缩率是 **1.5**。

## 使用 tokenizer：encode 与 decode

训练好 tokenizer 之后，怎么处理新文本？把新字符串 encode 一下——概念上，就是把训练时学到的**一系列 merge 规则**按顺序依次应用到新字符串上，得到一串索引。比如 `"the quick brown fox"` 会被编码成某个索引序列；再 decode 时，就是把每个 token 对应的字节片段重新拼接起来，你**得到的是同一个字符串**，完美往返。

Percy 承认这段讲得比较快，但他强调：**这个实现是能工作的，是一份完整的 BPE 实现——只是慢得离谱**。因此作业一会要求你把它做快：

- 目前 `encode` 会遍历**所有** merge 规则——而 merge 的数量大约等于"词表大小减 256"，所以你必须**只遍历真正有关的 merge**，并为此建立一些索引；
- 要**检测并保留特殊 token**（比如 `<|endoftext|>`），概念上不深，但对构建现代 tokenizer 来说很重要；
- 要使用**预分词（pre-tokenization）**，比如 GPT-2 tokenizer 用的正则；
- 尽量把实现做到最快。另一个技巧是：**把文本切成块，再对每个块分别应用 tokenizer**，而不是一次性处理整个字符串，这样会快很多。如果有一天你发现 Python 实在不够快，也可以换成你喜欢的 Rust 或 C 来写。

## 无 tokenizer 的梦想

如课程大纲里提到的，Percy 每年都希望不用再教 tokenization。**tokenizer-free 的梦想**是让模型直接操作字节：这方面已有一批工作，比如 **ByT5、MegaByte、BLT，以及最近看起来颇有希望的 H-Net**。它们很 promising，但**至今还没有被规模化到前沿模型**；既然前沿模型仍在用 tokenizer，教 BPE 依然是明智之选。

不过 Percy 提醒，即便将来真的摆脱了 tokenizer，**任何替代方案都必须满足两个性质**：

1. **模型（比如 transformer）必须操作序列的某种"块/抽象"**。这一点在文本之外（比如视频、DNA 序列）尤其明显：单个字节或基本单元的信噪比很低，你必须做一些抽象把它提升到可以建模的层次；
2. **块应当是可变长的**。你需要**自适应计算**——不能把每个字节一视同仁，否则必然是次优的。

## 总结与预告

一句话总结：tokenizer 在**字符串与 token（索引）之间**做转换；字符级、字节级、词级的 tokenization 各有各的严重缺陷；**BPE 是一个数据驱动的、相当有效的启发式算法**。tokenization 目前仍然是独立的一步——也许有一天能直接从字节端到端地完成，但那是后话。

到这里，本讲内容就结束了。Percy 预告：**周三将开始"资源核算"单元**，那算是"婴儿版系统（baby systems）"；之后会回到架构部分，从那里继续往下讲。

---

<!-- lecture-nav -->

**→ 下一讲**:[02 Resource Accounting](../02/)
