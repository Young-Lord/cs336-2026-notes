---
title: "第 1 讲差异:Overview, Tokenization"
lecture: "1-2"
---

# 第 1 讲差异：Overview, Tokenization

**讲师**: Percy Liang · **对比对象**: 2025 版 L1 vs 2026 版 L1

## 2025 第 1 讲骨架回顾（唤起记忆）

- **开场与课程动机**：研究者与技术日益脱节（从自己训模型 → 下载 BERT 微调 → 只 prompt 专有模型）；抽象是泄漏的（leaky），要支撑基础研究必须"拆开整条技术栈"，因此这门课的理念是 **understanding via building**。
- **工业化困境**:GPT-4 据传 1.8T 参数、$100M 训练成本；xAI 建 20 万张 H100 的集群；前沿模型闭源、无公开细节——小模型未必能代表前沿。
- **两个"小模型不代表前沿"的证据**:attention 与 MLP 的 FLOPs 占比随规模变化；涌现（emergent）行为需要足够规模才出现。
- **三类知识**：机制（mechanics）、心态（mindset）、直觉（intuitions）；后两者里只有心态能可靠地教。
- **苦涩的教训正确解读**：不是"规模至上、算法无用"，而是 **accuracy = efficiency × resources**，算法效率同样重要（OpenAI 2020:2012–2019 年 ImageNet 算法效率提升 44×）。
- **语言模型简史**:Shannon 熵 → 2007 年谷歌 5-gram(2 万亿 token)→ 2010 年代神经成分（2003 Bengio、seq2seq、Adam、attention、Transformer 2017、MoE、模型并行）→ ELMo/BERT/T5 → OpenAI 拥抱 scaling(GPT-2/GPT-3)→ 开放模型（EleutherAI、OPT、BLOOM、Llama、Qwen、DeepSeek、OLMo）。
- **课程结构**：可执行讲义、5 单元 5 作业、Together AI 提供 H100 集群、A1 leaderboard（90 分钟 H100）、AI 工具"风险自负"。
- **五大单元预览**:basics / systems / scaling laws / data / alignment，一切围绕**效率**。
- **分词**：分词器接口（encode/decode 往返）；**GPT-2 分词器**（`tiktoken.get_encoding("gpt2")`）观察；"Hello, 🌍! 你好！" 压缩率 1.6；字符/字节/词级分词缺陷；**BPE**(1994 数据压缩 → 2016 神经机器翻译 → GPT-2)；在 "the cat in the hat" 上逐步合并；encode 按合并顺序回放；"the quick brown fox" 往返；作业 1 扩展点（只回放相关合并、特殊 token、预分词、提速）。

---

## 2026 新变化详解

> 标注说明:**〔2026 新增,需学习〕** 表示 2025 没有的内容;**〔年份刷新,了解即可〕** 表示结构相同、只换了数字/例子;**〔相同,可跳过〕** 表示可完全略过。

### 1. 课程第三版的组织变化〔2026 新增,需学习〕

- **版本次数**:2025 是"第二次开课(扩招 50%、3 个 TA、讲座上 YouTube)",2026 是**第三版**,开场即说 "bringing you the 3rd offering"、"last year we decided to put all our lectures on YouTube"。
- **TA 阵容换血**:
  - **Marcel**:去年上过这门课,"太好玩所以回来了",研究高阶梯度与训练;
  - **Herman**:一年前完全不懂 LLM("tokens 是游戏里收集的东西,attention 是注意力经济"),上完这门课后现在做 LLM 研究,今年回来当 TA——课程的活广告;
  - **Steven**:第一次当 CA,研究语言模型理论、数据效率。
- **课程目标的新表述(重要)**。讲义 `welcome()` 里"What's new?"三行:
  - 不变:**from-scratch 哲学**;
  - 新增:**"Prioritize high value-per-time concepts, don't lose the forest for the trees"(更注重单位时间的价值密度,别只见树木不见森林)**;
  - 新增:**"More coverage of modern LM ingredients (mixture of experts, long-context, agents)"**。课上原话:"this year we're going to spend maybe a bit more time on mixture of experts, and of course agents are very popular these days, so getting a handle on long context and what is needed for that"。
- 还有一处背景变化:Percy 说"尤其是现在,一个 coding agent 可能已经能零样本(zero-shot)完成一门课作业",因此下面第 4 点的 AI 政策是配套产物。

### 2. 前沿模型清单与开放权重模型的分类变化〔2026 新增,需学习〕

这是 2026 第 1 讲**内容层面**最大的改动。两版都讲"开放生态",但结构完全不同:

- **2025 版**:开放模型是一个平铺列表(EleutherAI / OPT / BLOOM / Llama / Qwen / DeepSeek / OLMo),外加"开放程度三档":closed(GPT-4o)、open-weight(DeepSeek)、open-source(OLMo);前沿模型清单为 o3、Claude Sonnet 3.7、Grok 3、Gemini 2.5、Llama 3.3、DeepSeek R1、Qwen 2.5 Max、Hunyuan-T1。
- **2026 版**把开放模型重排为**三组**:
  1. **早期 GPT-3 复刻尝试**(early attempts to replicate GPT-3):EleutherAI(The Pile + GPT-J)、Meta OPT(175B)、BigScience BLOOM(176B)——课上点评"这些模型并不很强"(not very strong),因为没有足够算力/踩了很多硬件坑;
  2. **可信开放权重模型(weights + paper)**:Meta Llama、Mistral、DeepSeek、Alibaba Qwen、Moonshot **Kimi**、Z.ai **GLM**、**Minimax**、**小米 MIMO**——结论:**"开放权重模型正在逼近闭源模型"**(open weight models are approaching closed models),差距可能很小或相当;
  3. **开源模型(weights + paper + code + data)**:AI2 **OLMo**、NVIDIA **Nemotron**、**Marin(Percy 本人参与的项目)**——不仅给权重,还给论文、代码、数据,以便真正理解模型如何构建。
- **为什么 2026 更强调开放生态(新增的论证)**:这门课能讲下去,全靠这些开源论文与复现——即使 Qwen 等论文也缺数据配比等细节、无法完全复现,但比闭源"完全看不见"已经好得多,研究者得以"拼凑(triangulate)前沿模型是怎么造的"。
- **前沿模型引文更新(以讲义脚本 `references.py` 为准)**:2026 引用 **Kimi K2(2026 年初)、GLM-5、Minimax M2、小米 MIMO V2、Qwen 3.5、OLMo 3、Nemotron 3、Marin 8B/32B** 等。注意:转录口语里只点名了 "DeepSeek 和 Qwen"(以及 Kimi 相关的一两处),完整清单在讲义脚本的引文表中,建议以文档为准。
- **附带的数字刷新**:工业化叙事中,GPT-4 训练成本从"约 $100M"更新为"现在可能已到十亿美元量级(推测,speculative)";xAI 集群从 20 万 H100 更新为 **2025 年的 23 万 GPU**。

### 3. 分词器演示:从 GPT-2 换成 GPT-5(o200k_base)〔2026 新增,重点学习〕

这是本讲**唯一需要动手看的新演示**。两版的演示代码几乎只有一行不同:

- 2025 版:

```python
def get_gpt2_tokenizer():
    # Code: https://github.com/openai/tiktoken
    # You can use cl100k_base for the gpt3.5-turbo or gpt4 tokenizer
    return tiktoken.get_encoding("gpt2")
```

- 2026 版:

```python
def get_gpt5_tokenizer():
    # Code: https://github.com/openai/tiktoken
    return tiktoken.get_encoding("o200k_base")
```

Percy 在课上把 `o200k_base` 直接称为 **"the GPT-5 tokenizer"**。用同一个字符串 `"Hello, 🌍! 你好!"`(UTF-8 下恰好 **20 个字节**)做往返与压缩率:

| 分词器 | encode 后 token 数 | 压缩率(bytes/token) |
|--------|------------------|---------------------|
| GPT-2(`gpt2`,2025) | —(转录只报告压缩率) | **≈ 1.6** |
| GPT-5(`o200k_base`,2026) | **8**(转录明确给出) | **2.5**($20/8$) |

```python
tokenizer = get_gpt5_tokenizer()  # @stepover
string = "Hello, 🌍! 你好!"
indices = tokenizer.encode(string)
reconstructed_string = tokenizer.decode(indices)
assert string == reconstructed_string   # 往返
compression_ratio = get_compression_ratio(string, indices)  # 2.5
vocabulary_size = tokenizer.n_vocab     # 约 20 万(o200k_base)
```

- 2026 的讲解**新增了几个点**:
  - **压缩率的意义**:压缩率越大 → 序列越短 → 越好,因为 attention 对序列长度是**二次方**的(compression ratio is larger → shorter sequence, good since attention is quadratic)。
  - **词表与稀疏性的权衡**:可以靠加大词表提高压缩率,但词表每个元素都被模型当作独立类别处理,词表越大越稀疏;如今**多语言分词器普遍 100k–200k 词表**。
  - 讲义还会把整个词表**导出成文件**(`var/gpt5_tokenizer_vocab.txt`,每行一个 token)供浏览,并展示 `n_vocab`。
- 其余观察两版相同,可跳过:空格属于前导 token(" world" 一个 token)、"hello" 与 " hello" 是毫不相干的两个索引、数字被切成若干位一组(左到右,无千分位语义)。
- 另一个 2026 新增的小细节:演示时现场**没网**,所以交互式站点没跑起来,改成纯代码演示(转录里 "I guess I don't have internet")——不影响内容,你自测时可直接跑讲义脚本。

### 4. 其它年份刷新〔年份刷新,了解即可〕

- **课程计算资源**:Together AI H100 集群 → **Modal(云平台,按 API 使用)**,Percy 自述"去年是 SSH 进集群,今年更像用 API,一开始我有点怀疑,实际体验很舒服";A1 leaderboard 从"90 分钟 H100"改为"**45 分钟 B200**"(算力翻倍、时间减半)。
- **AI 政策(新增小节)**:2025 只有一句"AI 工具(CoPilot/Cursor)可能拿走学习,风险自负";2026 升级为正式政策——**必须使用课程提供的 `AGENTS.md`(或等价 prompt,要求 AI "pedagogically-minded" 教学心态)**,并附 AI policy guide 链接。理由是"coding agent 已经强到能直接做完所有作业,但那样你什么也学不到"。
- **"什么是语言模型"的时间线补第四格(2026 新增)**:2018(BERT)something you fine-tune → 2020(GPT-3)something you prompt → 2022(ChatGPT)something you talk to → **2026(agents)something that acts autonomously**,并现场演示一段巨大的 agent trace(给一页文本,agent 完成很复杂的 agentic 编码任务)。结论:**基础没变(GPU/内核/梯度/Transformer),变的是规格——更长的上下文、推理效率更重要**。
- **FLOPs 分布例子的表述更新**:2025 定性说"小模型 attention/MLP 相当,175B 时 MLP 主导";2026 给出数字——**小模型 MLP 约占 44%,175B 时到 80%**,并预告"接下来会花更多时间在 FLOPs 计数上"。
- **架构预览点名新成员(2026 新增)**:除了既有的激活函数/位置编码/归一化/GQA/MLA,新增 **线性注意力 / 状态空间模型(Mamba、Gated DeltaNet)**,并说"线性注意力与 attention 的混合(hybrid)通常效果不错"——这是给第 4 讲(Attention Alternatives)埋伏笔;分词器-free 路线的参考列表也加了 **H-Net(2025)**。
- **优化器与损失预览(2026 新增)**:除了 Adam,新增 **"Muon 已越来越多用于最新开源模型,比如 Kimi K2"**(为第 11 讲的优化器专题埋伏笔);损失函数提及 **multi-token prediction**。
- **系统预览(2026 新增)**:给出 **B200 数字(bf16 下 2.25 PFLOP/s,内存带宽 8 TB/s)**、**DGX B200(8 GPU,NVLink;上千 GPU 用 InfiniBand/Ethernet 互联)**、并预告 **roofline analysis**;推荐阅读 Google 出的 **《How to Scale Your Model》**(概念好,虽然偏 TPU,但新增了 GPU 章节)。
- **数据/对齐预览小幅刷新**:数据预览新增 **mid-training 数据(高质量、含 long-context,如大代码库/书)** 与 **post-training 数据(对话、带工具调用的 agentic trace)** 的分层;对齐预览强调 RL 规模化时的系统挑战(推理服务器 + 训练服务器、异步 rollout、on-policy 与吞吐量的拉扯)。

### 5. 相同,可跳过

以下内容两版几乎逐字相同,**直接跳过**:

- 开场整套叙事:研究者与技术脱节、抽象是泄漏的、"要理解就必须亲手构建";
- 三类知识(机制/心态/直觉)与 SwiGLU 的 "divine benevolence";
- 苦涩的教训与 accuracy = efficiency × resources、44× 算法效率改进;
- 2010s 神经成分简史、ELMo/BERT/T5、GPT-2/GPT-3/PaLM/Chinchilla 的缩放叙事;
- 课程 5 单元 5 作业结构与"为什么上/不上这门课"、可执行讲义的形态;
- 分词器接口与往返、字符/字节/词级分词及其缺陷、**BPE 算法与 "the cat in the hat" 逐步合并推导**、encode 按合并顺序回放、"the quick brown fox" 往返、作业 1 的扩展点(提速/特殊 token/预分词)——代码与讲解完全一致。

---

## 小结

第 1 讲是两版之间**质量最稳定的一讲**：结构零改动，内容是整体"年份刷新"。真正需要你补的就三件事：(1) **GPT-5 分词器（o200k_base）与压缩率 2.5**;(2) **开放模型三档分类与 Kimi K2/GLM-5/Minimax M2/Qwen 3.5/OLMo 3 等新清单**;(3) **课程第三版的目标表述与 Modal/AGENTS.md 等新安排**。其余都可以沿用 2025 的理解，放心跳到第 3 讲。
