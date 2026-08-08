---
title: "17 Alignment - Multimodality"
lecture: 17
---

# Lecture 17: Alignment - Multimodality（多模态）

**讲师**：Tatsu Hashimoto · **主题**：多模态（CLIP 与 SigLIP 图像编码器、VLM、LLaVA、Qwen-VL 系列、Chameleon、全模态模型）

## 本讲内容

这一讲是后训练/对齐系列的收尾。Tatsu 原本计划这一讲继续讲强化学习，但思来想去，既然只有这一讲的份额，完全跳过多模态会让整门课显得不完整——毕竟环顾所有主流模型，多模态能力几乎无处不在。于是他把这一讲做成了一次**多模态模型的概述**。到目前为止，这门课只讲过语言模型（text ⇒ text）；而现实世界是多模态的（文本、图像、音频、视频），人们心中的"北极星"是**全模态模型（omni model）**——输入任意模态的任意组合、输出任意模态的任意组合。Transformer 很擅长、却也只会"说 token"，所以核心问题永远是：**如何把图像等非文本模态转换成 Transformer 能消化的（离散或连续）token**。本讲的两个问题（如何输入、如何输出非文本数据）里，**几乎只回答第一个**。

主体分三块。**第一块是图像编码器**。CLIP（Contrastive Language-Image Pre-training，2021）把网上海量（图像，caption）配对变成对比学习的目标——对每个 batch 里的图像与文本分别编码，用点积构造 $N\times N$ 相似度矩阵，让对角线（匹配对）得分远高于非对角线，这正是 InfoNCE 的结构；视觉编码器用 ViT（分块 + 位置嵌入 + attention pooling），文本编码器用 GPT-2 风格 Transformer，最终零样本在 ImageNet 上超过了在 120 万张标注图上训练的 ResNet-50。SigLIP 把 CLIP 的"每行每列各一次 $N$ 分类"改成"每个（图像，文本）对一次二分类"，从而解耦了 batch size 与损失、也让并行化变得可行（10 天 256×TPUv3 → 5 天 32×TPUv4）。**第二块是把图像编码注入语言模型**——也就是 VLM。LLaVA（2023）用 CLIP + Vicuna + 一层线性投影 $W$，在 GPT-4 合成的 158K 条指令数据上两阶段训练；LLaVA OneVision（2024）升级到 SigLIP + Qwen2 + 两层 MLP，用 AnyRes 解决 OCR 需要的高分辨率、并统一处理单图/多图/视频。Qwen-VL 系列走的是另一条略有不同的路线：Qwen-VL 用一层交叉注意力适配器与特殊 token，Qwen2-VL 引入动态分辨率与三维的 MRoPE 位置编码，Qwen3-VL 则用交错式 MRoPE、显式视频时间戳、平方根归一化逐 token 损失与 DeepStack 跨层融合，把基准刷到与 Gemini、GPT-5、Opus 4.1 直接竞争。**第三块是眺望全模态模型**——Meta 的 Chameleon（2024）把所有模态（包括图像）都离散成 token，用 VQ-VAE 把 512×512 图像压成 1,024 个 token，然后就是普通语言模型训练；它优雅却不如连续编码方案能打，还面临文本低熵/图像高熵带来的训练不稳定。最后以全讲总结收尾：**连续编码器 + Transformer + diffusion** 是当下生成侧的最佳猜测组合，而"语义 vs 细粒度细节"决定了理解与生成可能需要不同的表示。

| 页面 | 内容 |
|------|------|
| [17 · 引言：从语言模型到全模态模型](01-introduction-and-omni-models.md) | 讲次动机（多模态无处不在）、**语言模型 text→text**、多模态世界与**全模态模型（omni model）**北极星、Transformer 只会说 token（离散或连续）、token 应代表语义单元、文本有 BPE 而图像没有等价物、**两个核心问题（如何输入非文本、如何生成非文本）**、本讲聚焦输入 |
| [17 · CLIP：对比语言-图像预训练](02-clip.md) | 历史背景（ImageNet/ResNet 时代 vs GPT-3 的语言"基础模型时代"）、动机（利用网上海量图文对）、**CLIP 方法与 InfoNCE 目标的完整形式化**、训练代码（归一化 → 温度缩放点积 → 双向交叉熵）、数据（500K 查询 / 4 亿图文对、OpenCLIP 与 LAION-5B 的自举）、数据处理（resize + 中心裁剪）、**ViT 视觉编码器（分块、位置嵌入、attention pooling 公式）**、ViT-L/14@336px、文本编码器（GPT-2、[BOS]…[EOS]）、**零样本 ImageNet 超越 ResNet-50**、课堂问答（弱监督噪声、位置嵌入）、消融（预测文本 vs 排序目标） |
| [17 · SigLIP：用二分类损失训练图像编码器](03-siglip.md) | CLIP 的两个技术缺点（需要超大 batch、softmax 覆盖全 batch 不可分解）、**SigLIP 逐对二分类目标（公式 + 代码）**、课堂问答（负样本采样）、数据（WebLI：OCR、过滤、100 语言）、**效率对比（CLIP 10 天 256×TPUv3 vs SigLIP 5 天 32×TPUv4）**、并行化（文本嵌入轮转交换、覆盖非对角块）、**batch size 与损失解耦**（<16K 更优、100 万无益、临界值约 32K） |
| [17 · LLaVA 与 LLaVA OneVision：把图像注入语言模型](04-llava.md) | VLM 模板（视觉编码器 + 投影器 + 语言模型）、LLaVA 2023（CLIP + Vicuna + **线性投影 $W$**）、**GPT-4 合成数据（MS COCO + 158K 例子）**、**两阶段训练（对齐 $W$ → 微调语言模型）**、"熨斗熨在面包车后座"的例子、LLaVA OneVision 2024（SigLIP + Qwen2 + 两层 MLP）、**AnyRes 高分辨率方案（为 OCR）**、单图/多图/视频的 token 预算分配、质量优先数据、由易到难三阶段训练、跨模态迁移（图表→多图、OCR/关系推理→GUI 智能体、视觉提示→视频）、开源模型与数据 |
| [17 · Qwen-VL 系列：从 Qwen-VL 到 Qwen3-VL](05-qwen-vl.md) | Qwen-VL（OpenCLIP ViT、一层交叉注意力适配器、特殊 token、三阶段训练、中文能力示例）、Qwen2-VL（**动态分辨率**、675M ViT、2×2 压缩、视频 2 帧/秒、**MRoPE 三维旋转位置编码**）、Qwen3-VL（Qwen-3 底座、256K 上下文、SigLIP-2、**交错式 MRoPE**、显式视频时间戳、**平方根归一化逐 token 损失**、DeepStack 跨层融合、4 阶段预训练 + 3 阶段后训练、对 Gemini/GPT-5/Opus 4.1 的 SOTA 成绩）、课堂问答（视频生成、系统与数据加载、对齐预算、参数规模） |
| [17 · Chameleon：把一切变成离散 token 与本讲总结](06-chameleon-and-summary.md) | VLM 只能生成文本的局限、**Chameleon 的思路（一切皆为离散 token、文本图像同处一个空间）**、**VQ-VAE（码本量化、重建损失 + 码本/commitment 项、512×512→1024 token、词表 8192）**、新 BPE tokenizer、两阶段训练（2.9T 文本 + 1.5T 图文 + 400B 交错）、**训练稳定性（文本低熵 vs 图像高熵、范数增长、QK-norm + z-loss 修复）**、离散化丢失信息（OCR）、VQ-VAE 为何被 diffusion 取代、**全讲总结（连续编码器 + Transformer + diffusion、语义 vs 细节、模态加权）** |

## 本讲要点

- **多模态是必答题**：前沿模型都被期望是原生多模态、甚至全模态（omni）的——输入/输出任意模态的任意组合。但本讲只解决"输入"这一半：如何把图像等非文本数据转换成 Transformer 能消化的 token；
- **token 的本质是语义单元**：文本里有子词（BPE），一个像素却没有语义；所以非文本模态需要一个"图像的 BPE tokenizer"。Transformer 只会说 token（离散或连续），一切模态都得先变成 token；
- **CLIP 是对比学习（InfoNCE）**：批量编码（图像，文本）对，点积构造 $N\times N$ 相似度矩阵，两个方向的交叉熵让对角线远高于非对角线。它天然需要**大 batch**（负样本越多信号越强），且 softmax 覆盖全 batch、难以分解；
- **CLIP 的表示是"语义"而非"细节"**：因为配对的是自然语言 caption（高层的、语义的），设计决策也都围绕图像分类——所以它捕获高层语义、对细粒度（OCR 之类）不友好，但作为 VLM 的起点足够稳健；零样本 ImageNet 超过 ResNet-50 是它成名的头条结果；
- **ViT 是图像编码器的赢家**：图像切 14×14 patch、加位置嵌入、过标准 Transformer；CLIP 用 attention pooling（全局平均做 query 再对每个位置做一轮注意力）取整图表示；2D 位置嵌入对分类影响不大，直到 Qwen 的 MRoPE 才真正利用空间结构；
- **SigLIP 把多分类改成逐对二分类**：每个（图像，文本）对独立判断"对齐与否"（$\log\sigma(y_{ij}z_{ij})$），于是**batch size 与损失解耦**、并行化可行（轮转交换文本嵌入覆盖非对角块）——CLIP 的 10 天 256×TPUv3 变成 SigLIP 的 5 天 32×TPUv4，临界 batch 约 32K；
- **VLM 的标准模板是"视觉编码器 + 投影器 + 语言模型"**，本质是中期/后训练式的"缝合"：取现成 CLIP/SigLIP 与现成 LLM，分阶段训练。LLaVA 投影器只是一层线性 $W$，先冻结一切只训 $W$ 对齐空间，再放开语言模型微调；
- **数据是 VLM 最大的工作量**：LLaVA 用 GPT-4 合成 158K 指令数据（毫不掩饰的蒸馏）；LLaVA OneVision 坚持"质量优先"、任务化数据、由易到难三阶段训练，并展示了**跨模态迁移**（单图图表→多图问答、OCR+关系推理→GUI 智能体、视觉提示→视频）；
- **分辨率与长上下文是两条主线**：AnyRes/动态分辨率让高分辨率、任意尺寸图像可被处理（OCR 需要）；视频推高了上下文需求（Qwen3-VL 做到 256K），Qwen2-VL 的视频上限是 16K token、每秒 2 帧；
- **Qwen3-VL 的 SOTA 打磨**：交错式 MRoPE（让时间/宽/高轴都暴露在低高频）、显式视频时间戳 token、平方根归一化逐 token 损失（防止长视频主导训练）、DeepStack 跨层把视觉嵌入注入残差流；闭源模型（Gemini/GPT-5/Opus 4.1）的每行最优常被 Qwen 拿下；
- **Chameleon 的"一切皆离散 token"**：VQ-VAE 把图像量化进 8,192 的码本（512×512→1024 token），然后就是纯自回归语言模型训练。优雅、统一，但**离散化丢信息**（OCR 读不出小字）、**训练不稳定**（文本低熵 vs 图像高熵导致范数增长，用 QK-norm 与 z-loss 修复）、性能也不如连续编码方案；
- **本讲的路线判断**：理解与生成要求不同——CLIP 式小向量装高层语义即可，生成/OCR 需要细粒度细节（diffusion 擅长）；多模态训练要小心加权（视频信息密度低，别让它压过文本）。当下最佳猜测组合是**连续编码器 + Transformer + diffusion 做生成**。

## 课程导航

- [上一讲：16 Post-Training - RLVR](../16/)
- [下一讲：18](../18/)
