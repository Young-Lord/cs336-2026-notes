---
title: "07 · 硬件：GPU 如何连接"
lecture: 7
---

# 硬件：GPU 如何连接

在写代码之前，先讲讲**硬件**——特别是 GPU 是怎么连在一起的。我们已经知道 GPU 内部长什么样了（上一讲的 SM、内存层级、tensor core……），现在关心的是它们之间怎么通信。

## 经典拓扑（你在家里）

先看一幅非常经典、看得出年代感的图：这就是"老式"计算机的一般工作方式。你有一台服务器，上面挂着一堆 CPU，通过 **PCIe** 总线接外设（过去鼠标、键盘都走 PCIe）以及若干块 GPU，旁边是内存条；然后这台计算机通过 **Ethernet** 连到另一台计算机，如此往复。

在这个设定里：

- **同一节点内的 GPU** 通过 **PCI(e) 总线**通信（PCIe 7.0、16 通道能到约 **242 GB/s**）；
- **不同节点上的 GPU** 要通信，就得一路穿过 **Ethernet**（大约 **200 MB/s** 的量级）。

这就是"你买了一块游戏 GPU、跟朋友接上线一起训练大模型"时的情形——听起来就很痛苦。但如果你是真的认真做训练，场景会变成下面这样。

## 现代数据中心拓扑

数据中心的景象，就是开场那张图（每块 GPU 有 NVLink、NVSwitch、InfiniBand 这些连接）：

![](/lectures/07/gpu-node-overview.png)

典型的层次结构如下（注意：8 这个数字是典型的，256 是我编的示意）：

1. **节点（node）内部**：每节点通常 **8 块 GPU**，用 NVIDIA 的 **NVLink** 连到一个 **NVSwitch**。做个数量级校准：**NVLink 5.0**（B200 那代）的总带宽约 **1.8 TB/s**，而 B200 的 HBM 是 **8 TB/s**——所以 NVLink 大概比 HBM 慢 **4 倍**。跨设备能到这个数字仍然算很快，但显然不如 HBM，更不如 shared memory 和 L1。重要的是，从编程角度看，**NVSwitch 让"任何 GPU 到任何 GPU"成为可能**：你把数据交给硬件，它帮你传到交换机、再由交换机路由，你不需要操心具体链路；
2. **Pod**：当 GPU 数量继续增长、装不下一个 NVSwitch 域时，节点就要组成 **pod**，pod 内部用 **InfiniBand** 互联。InfiniBand 的路由方式和前面不同：GPU 不能直接连 GPU，要经过 **PCIe → HCA（InfiniBand 网卡）→ InfiniBand 光缆**，速度比 NVLink 低一两个数量级（约 **0.05 TB/s** 的量级）；
3. **集群 / 数据中心**：InfiniBand 也用完了、要接更大规模时，就用 **Ethernet** 连各个 pod。走 Ethernet 要经过 **PCIe → CPU**，后面会看到，这比 InfiniBand 还慢。

这跟内存层级是同构的：**节点越多，越慢**。你不可能让一个 NVSwitch 去处理十万块 GPU。

## 绕过 CPU：RDMA

上一节反复提到"经过 CPU"——这是硬件层面一个非常关键的点。传统走 **Ethernet** 时，GPU 要把数据发给另一台机器的 GPU，必须经过 CPU：先把数据拷贝到 CPU 的 **内核 socket 缓冲区**（这里的"内核"是操作系统意义上的，不是 GPU kernel），然后**构建网络包（TCP 包）**，再拷贝到**网卡的环形缓冲区**，最后才发出去——这中间会引入大量延迟和拷贝。

解决办法叫 **远程直接内存访问（Remote Direct Memory Access，RDMA）**：它允许一块 GPU **直接读写另一块 GPU 的内存，完全不经过 CPU**。显然，在 NVLink/NVSwitch 的世界里你天然就有 RDMA；**InfiniBand 也支持 RDMA**——只要走 InfiniBand，GPU 就能不经 CPU 直接互联；但**标准的 Ethernet 不支持**。

有两个值得提的进展：

- **NVL72**：NVIDIA 一直在把 pod 做得更大。对 B200/B300 这一代，他们推出了 **GB200/GB300 NVL72**——每个**托盘**（tray）上 8 块 GPU，9 个托盘叠成一个**机架（rack）**，于是**72 块 GPU 全都在一个 NVLink 域里**、全部经 NVSwitch 互联。还记得吗，NVLink 非常快。普通人只能享受"8 块 GPU 高速互联，再往外就慢下来"；有钱的话，可以买到把超快互联延伸到 72 块 GPU 的硬件；
- **RoCE（RDMA over Converged Ethernet）**：前面说标准 Ethernet 不支持 RDMA，但以太网这边也有进展。**RoCE** 让 Ethernet **绕过 CPU**，相当于对 InfiniBand 的一个回应。InfiniBand（以及许多 NVIDIA 产品）非常贵，而 RoCE 能用便宜不少的东西换到不错的性能；**Meta** 有论文在探索这条路线——所以 **Llama 可能（也可能没有）是在 converged Ethernet 上训练的**。

于是硬件全貌就是：GPU 经 NVLink 连到 NVSwitch（一个域里 8 块，或者 72 块），再往外走 InfiniBand，再往外走 Ethernet。

### 课堂问答：关于硬件细节

**问答：机架（rack）和托盘（tray）到底长什么样？**

（同学：能具体描述一下 NVL72 的 rack 和 tray 是什么吗？）

**Percy**：我并不是硬件专家，但 rack 就是你见过的数据中心里那种机架，每个 tray 是机架里的一层。NVL72 里每个 tray 有 8 块 GPU：**G 代表 Grace**——每个 tray 上两颗 Grace CPU，每颗 CPU 连 4 块 GPU，所以每 tray 一共 8 块 GPU，叠起来，全部接到这个 NVSwitch 上。

**问答：RDMA 和这些硬件（NVLink、InfiniBand）的区别是什么？**

（同学：RDMA 和刚才说的 NVLink、NVSwitch、InfiniBand 是什么关系？）

**Percy**：可以把 RDMA 想得更"操作层面"一点——它描述的是"一块 GPU 可以直接读写另一块 GPU 的内存"这件事，而实现 RDMA 有好多途径：走 NVLink 和 NVSwitch、走 InfiniBand……这些是**硬件**（什么线、什么交换机），而 RDMA 是**通信时实际发生的行为**。比如前面提到的 RoCE 也是实现 RDMA 的另一种途径。

**问答：NCCL 有没有针对多节点集群优化？**

（同学：NCCL 是不是专门为多节点集群优化的？）

**Percy**：我不清楚它具体优化了哪些细节。我只能说，NVIDIA 的主要客户就是那些大规模语言模型的提供方，所以整个软件栈都在为大规模模型的训练和推理做优化——如果他们没想过为这类负载优化，我会很惊讶。

**问答：如果有 9 块 GPU 呢？**

（同学：如果我有 9 块 GPU，怎么分配工作负载？）

**Percy**：这取决于第 9 块落在哪。通常每节点 8 块 GPU，第 9 块就会在另一个节点上——如果它们之间没有 NVLink 连起来，那就很糟糕：那个节点贡献不了多少算力，通信还贵。但如果所有 GPU 都在一个 NVSwitch 域里，那就合理多了。

**问答：这和 TPU 有什么不同？**

（同学：这套硬件结构和 TPU 的区别是什么？）

**Percy**：TPU 整体上是**简单得多的对象**。这些组件各自对应 TPU 里的什么，我不太了解细节——也许我们可以课下聊。

## NCCL：NVIDIA 集合通信库

那么怎么编程使用这套硬件？最底层有一个东西叫 **NVIDIA 集合通信库（NVIDIA Collective Communication Library，NCCL）**。它的职责是把 collective 操作（all-reduce、reduce、broadcast……）翻译成真正在 GPU 之间发送的**底层包（low-level packets）**。

你用 NCCL 的时候，本质上就是喊一声"**我要 all-reduce**"，然后：

1. NCCL **探测硬件拓扑**——有几台节点、几个交换机、NVLink 还是 PCIe，等等；
2. 据此**优化 GPU 之间的通信路径**（比如决定走 ring 还是 tree）；
3. **启动 GPU kernel** 来收发数据——因为说到底，GPU 上跑的一切都是 kernel，**通信 kernel** 也是 kernel，它们真正执行和其他 GPU 的通信。

我们不会在 NCCL 上停留太久，你只需要知道它存在。接下来要真正上手的是 **PyTorch**。

<!-- lecture-nav -->

**← 上一节**：[集合通信原语：collectives 的编程模型]（collectives.md）　**→ 下一节**：[torch.distributed：实现 collectives 与通信带宽](torch-distributed.md)
