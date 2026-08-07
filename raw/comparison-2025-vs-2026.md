# CS336 "Language Modeling from Scratch":2025 春季版 vs 2026 春季版(第三版)深度对比报告

> 对比依据:两版全部转录(2025 年 17 讲 23.7 万词、2026 年 18 讲 24.9 万词)逐讲通读,并交叉核对两版仓库的可执行讲义脚本(lecture_XX.py)与 PDF 讲义清单。所有结论均标注了转录/文件中的具体证据。

---

## 1. 总评

**2026 版相比 2025 版有非常明显、且方向明确的改进。** 这不是一次"换皮",而是一次基于两年授课经验的系统性迭代:课程在保留"从零构建 + 效率至上"内核不变的前提下,(a) 把 2025 年没有的**线性注意力/状态空间模型/混合架构**正式纳入正课;(b) 新增了**多模态**与**推理服务(guest lecture)**两个全新讲次;(c) 把对齐/RL 部分重组为"SFT/RLHF/DPO → RLVR",并塞进了 2025 年分散在三个讲次里的内容;(d) 大规模刷新了每个讲次引用的前沿模型与数字(GPT-5 分词器、B200、Kimi K2、Qwen 3.5、DeepSeek R1 等);(e) 工具链从 `requirements.txt + execute.py + Together AI 集群` 全面切换到 `pyproject.toml + uv + modal_execute.py(B200)`,并首次引入面向 AI 助手的教学政策(AGENTS.md)。教师本人在转录中反复出现"这是第一次讲线性注意力""今年加了优化器部分"这类自我迭代痕迹,是课程在主动追赶领域变化的最直接证据。

---

## 2. 课程结构变化表

| # | 2025 讲次 | 2026 讲次 | 变化类型 | 说明 |
|---|-----------|-----------|----------|------|
| 1 | 01 Overview, Tokenization | 01 Overview, Tokenization | 内容更新 | 结构相同;分词器演示从 GPT-2 换成 **GPT-5 (o200k_base)**;前沿模型清单、AI 政策全新 |
| 2 | 02 PyTorch, Resource Accounting | 02 PyTorch, Resource Accounting | 改名/微调 | 标题标注 einops;内容几乎不变,算术强度/roofline 前置,新增 FP4/NVFP4/NeMo-3 Super |
| 3 | 03 Architectures, Hyperparameters | 03 Architectures | 改名(标题精简) | **超参数内容并没有被删**:2026 转录原话仍叫"everything you didn't want to know about **architectures and hyperparameters**",只是课程表标题简化 |
| 4 | 04 Mixture of Experts | 04 Attention Alternatives + MoE | **内容扩容/替换** | 变化最大处之一:2025 整讲只讲 MoE;2026 前半新增线性注意力/SSM/混合架构(此前从未讲过),MoE 移到后半,篇幅被压缩但保留 |
| 5 | 05 GPUs | 05 GPUs, TPUs | 更名/内容更新 | 两版都讲"两张 PPT 的 TPU",2026 在标题中显式标出;硬件示例从 A100/H100 更新到 B200 |
| 6 | 06 Kernels, Triton | 06 Kernels, Triton | 内容更新 | 结构相同(GELU→softmax→matmul→FlashAttention);硬件更新到 B200、提及 tensor memory |
| 7 | 07 Parallelism 1 | 07 Parallelism | 重组(拆分比例对调) | 两版都是"collectives + 数据/张量/流水线并行"的可执行讲座;2026 本讲更短(1.1 万词 vs 1.7 万词),把深度内容移入第 8 讲 |
| 8 | 08 Parallelism 2 | 08 Parallelism | 内容更新 | 2026 本讲更长(1.6 万词),TPU toroidal mesh 对比 GPU fat-tree 的篇幅更足 |
| 9 | 09 Scaling Laws 1 | 09 Scaling Laws | 微调 | 结构相同(历史线:Bell Labs 1993 → Hestness 2017 → Kaplan/Chinchilla);预告"今年加优化器" |
| 10 | 10 Inference | 10 Inference | 内容更新 | 增加 agentic workload 框架、连续批处理、OpenAI 日产 8.6T tokens 等新素材 |
| 11 | 11 Scaling Laws 2 | 11 Scaling Laws | **新增板块** | 2025 案例为 Cerebras-GPT/MiniCPM/DeepSeek+μP 推导;2026 案例更新为 MiniCPM/DeepSeek/**Kimi K2**,并**新增一整块优化器专题(Muon、SOAP、AdamW、Newton-Schulz)** |
| 12 | 12 Evaluation | 12 Evaluation | 微调 | 结构相同;榜单示例更新到 GPT-5.5 时代 |
| 13 | 13 Data 1 | 13 Data (Sources, Datasets) | 内容更新 | 新增**美国版权法第 107 条四要素合理使用分析、Creative Commons 许可详解**(2025 只一句带过);数据源示例更新 |
| 14 | 14 Data 2 | 14 Data | **新增板块** | 2025 只讲过滤+去重;2026 新增**后训练/合成数据专题(Open Thoughts、教师模型、answer filtering 实验结论)**及 FineWeb PDFs |
| 15 | 15 Alignment - SFT/RLHF | 15 Mid/Post-Training | **重组** | 2026 把 SFT + RLHF/PPO + DPO + overoptimization 全部压进第 15 讲,新增 mid-training/decay 阶段混合、"base model 是谎言"、注释者经济/人口统计学偏差等全新内容 |
| 16 | 16 Alignment - RL 1 | 16 Post-Training - RLVR | **重组/加厚** | 2026 把 2025 第 16、17 两讲的 RLVR 内容合并为完整一讲,并整合 R1/R1-Zero/Kimi K1.5 案例与 RL 基础设施讨论 |
| 17 | 17 Alignment - RL 2 | 17 Alignment - Multimodality | **内容替换** | 2025 第 17 讲是策略梯度/GRPO 代码深挖(该内容被并入 2026 第 16 讲);2026 第 17 讲整讲换成**多模态模型概述(CLIP/ViT/VLM)**,教师原话:"原计划继续讲 RL,但只剩一讲,不讲多模态这门课就不完整" |
| — | — | 18 Guest Lecture: Dan Fu (UCSD/Together, 推理服务) | **新增** | 全新 guest lecture,讲"一个 token 的一生"、prefill/decode 分离、KV cache、推理工作负载分析 |

**结构结论**:新增 2 个讲次(多模态、推理服务 guest),1 个讲次被整体替换(2025 的 L17 RL-2 → 2026 的 L17 多模态),1 个讲次被扩容替换(L4 从纯 MoE 变为"注意力替代 + MoE"),对齐/RL 板块从 3 讲重组为 2 讲;其余约 10 讲是"结构不变、内容刷新"。

---

## 3. 各主题深度对比

### 3.1 分词与总览(讲 1)

- **内核未变**:两版开头 ~2000 词几乎逐句对应——"researchers disconnected from underlying technology"、GPT-4 训 1.8T 参数/$100M、"bitter lesson 的正确解读是 algorithms that scale"、"accuracy = efficiency × resources"、三类知识(mechanics/mindset/intuitions)。这说明课程叙事是稳定的。
- **硬更新点**:
  - 分词器现场演示:2025 用 `tiktoken.get_encoding("gpt2")`(讲义脚本 lecture_01.py 574-575 行);2026 用 `tiktoken.get_encoding("o200k_base")`,并称"Here's the **GPT-5** tokenizer"(转录中两次提到 GPT-5)。
  - 前沿模型清单:2025 列 o3、Claude Sonnet 3.7、Grok 3、Gemini 2.5、Llama 3.3、DeepSeek R1、Qwen 2.5 Max;2026 重写为 **Kimi K2(2026 年初)、GLM-5、Minimax M2、小米 MIMO V2、Qwen 3.5、OLMo 3** 等,并把开放权重模型细分为"早期 GPT-3 复刻尝试(OPT/BLOOM/GPT-J)"与"可信开放权重模型(Llama/Mistral/DeepSeek/Kimi/Qwen)"两档——分类更精细。
  - 课堂规模:2025"第二版开课、扩招 50%、3 个 TA、讲座上 YouTube";2026"第三版",TA 换为 Herman(去年学生,现做 LLM 研究)、Steven 等。
- **总评**:结构零改动,内容做"年份刷新",是质量最稳定的讲次。

### 3.2 架构与超参数(讲 3)

- 用户提示里问"2026 是否把超参数并进去了?"——**答案:没有并走,超参数本来就还在**。2026 转录开场原话:"I've titled this lecture **everything you didn't want to know about architectures and hyperparameters**",并明确说"after we've established the standard building blocks… we're going to talk about hyperparameters(FF dim、vocab 数)"。课程表把 2025 的"Architectures, Hyperparameters"简写为"Architectures"纯属标题精简。
- **真正的差异在调查表**:2025 年 Tatsu 抱怨"去年我以为是几篇论文,结果发现 19 个新稠密模型(Command A、Gemma 3、Qwen 2.5、InternLM 等)";2026 年他说"今年稠密模型少了,但 **Qwen 3、Gemma 4(上周四刚发布)、Llama 4、Nemotron**,还有 Percy 自己用 Marine 训的 8B 模型",且**新增模型大多是 MoE**,并预告"明天(L4)讲 MoE"。
- **主题迁移**:2025 讲的是"去年趋势=训练稳定性改进(double norm 等)";2026 讲的是"**今年的趋势=支持更长上下文的架构变体**"——直接呼应 2026 总览里"long-context、agents"的新重点。
- 两版都覆盖 LayerNorm 位置(prenorm 共识、OPT-350M 例外)、SwiGLU、RoPE、double-norm 等,内容深度相当。

### 3.3 MoE 与注意力替代(2025 L4 vs 2026 L4)——全课程变化最大的一处

- **2025 L4(纯 MoE,1.6 万词)**:开场即"去年这是 bonus 讲座,今年变成必修";整讲围绕 MoE 原理、FEDUS/OLMo 消融、"专家并行=额外并行轴"、并最终**以 DeepSeek V3 架构逐部件走查收尾**。
- **2026 L4(注意力替代 + MoE,1.7 万词)**:
  - 前半全新讲**线性时间注意力**:从"乘法的结合律"重排 QK^T V 的括号、N²→ND 复杂度、线性注意力与 RNN 形式的等价、KV 状态携带;然后讲**混合架构**(如 Minimax M1 的 7:1 线性+softmax 混合),并以"尚未有人把纯线性注意力做到 scale,目前全是 hybrid"收束。教师原话:"**这是第一年我讲线性时间注意力**,因为现在这些方法已经在 scale 和 production 中被验证了"——这是全课程最直白的"教学迭代"证据。
  - 后半 MoE 被压缩但保留核心:多专家收益、expert parallelism、**structured sparsity 与硬件协同设计**、Nemotron 3 的"先 down-project 再通信降开销"技巧、load balancing/随机性。
  - DeepSeek V3 的逐部件走查在 2026 被弱化(不再整讲收尾),让位给注意力替代内容。
- **总评**:2026 用"注意力替代(半讲)+ MoE(半讲)"替换 2025 的"纯 MoE",覆盖了 2025 完全缺席的 Mamba/SSM/线性注意力/混合架构一族,是本课程"拥抱 2024-2026 领域变化"最典型的一讲。

### 3.4 GPU / Kernel / TPU / XLA(讲 5、6)

- 两版 L5 都是"GPU 硬件模型 + 六个加速技巧 + 用 FlashAttention 收官";TPU 在两版都只有约"两张 PPT"的篇幅(2025 是"very briefly about TPUs",2026 是"two slides about TPUs")。
- 2026 的硬件叙述从 H100 更新到 **B200(192GB HBM、2.25 PFLOP/s bf16、tensor memory)**,并提到 NeMo-3 Super 用 FP4 训练。
- **XLA 的核实**:映射表写 2026 L6 为"Kernels, Triton, XLA",但转录核查显示 **XLA 在 2026 转录中一次都没出现**;反倒是 2025 L6 有 1 处(讲 torch.compile/JAX 的 XLA 编译器能做到 FlashAttention 式优化)。2026 L6 仍以 Triton 为主线(GELU/softmax/matmul/FlashAttention),新增 CUTLASS、ThunderKittens 各 1 处提及。**结论:此条映射不准,2026 并未扩讲 XLA。**
- 总评:该板块两版差异不大,主要是硬件/数字刷新。

### 3.5 并行(讲 7、8)

- 结构完全对应:两版都是"可执行讲座(collectives + DDP/TP/PP 代码)"+"深度讲座(网络拓扑、all-reduce 算法、ZeRO/Megatron 案例)"。2026 第 7 讲与 2025 的对应可执行脚本函数名几乎一一对应(`collective_operations`/`data_parallelism`/`tensor_parallelism`/`pipeline_parallelism` 等)。
- 2026 唯一明显增量:第 8 讲把 **TPU toroidal mesh vs GPU fat-tree** 的对比讲得更足(附 3D 可视化链接),并保持 2025 的 JAX/FSDP/DeepSeek H800 低带宽集群案例。
- 总评:迭代痕迹弱,属"微调"。

### 3.6 缩放定律(讲 9、11)

- L9(基础)两版几乎相同:都从"10,000 张 B200(2026)/H100(2025)"的土豪朋友场景切入,走 Bell Labs/Cortes-Vapnik 1993 → Banko & Brill → Collobert 2012 → Hestness 2017 → Kaplan/Chinchilla 的历史线,强调"scaling law 是工程曲线拟合,不是自然定律"。
- **L11(进阶)是 2026 增厚最明显的讲次之一**:
  - 2025 案例:Cerebras-GPT、MiniCPM、DeepSeek LLM,配 μP 数学推导;顺带提 Llama 3、Hunyuan、MiniMax-01。
  - 2026 案例:MiniCPM、DeepSeek,终点提到 **Kimi K2**;μP 仍为核心。
  - **新增优化器大板块**(2025 没有):Muon(基于 Newton-Schulz 的矩阵正交化、谱范数归一)、AdamW/SOAP 对比,并以 **Kimi K2 全程用 Muon 训练**作为"小规模结论能否迁移到大 scale"的叙事:先"Muon 在 nanoGPT speedrun 上很神"→ 有人 scaling 后说不行 → 结果 Kimi K2 用 Muon 训出顶级模型。教师原话:"今年我要在进阶讲里塞更多优化器内容"。
- 总评:2026 让缩放定律板块从"纯曲线拟合"拓展到"优化器+初始化+超参的 scale 稳定性",更贴近一线训练实操。

### 3.7 推理(讲 10)

- 两版主线一致:prefill vs decode、KV cache、量化、剪枝、投机解码。2026 增量:
  - 开场动机更新:"OpenAI 日产约 8.6 万亿 token(不到 4 天就超过 GPT-4 的训练 token 量)",并用 **agentic 工作负载**重新论证推理重要性的上升("agent 产出的大部分 token 不是给人读的")。
  - 工程栈列举新增 **SGLang(适合 agentic)**、连续批处理(continuous batching);vLLM/TensorRT/llama.cpp 保留。
- 总评:结构未动、素材年份刷新,并把"agents"嵌入推理叙事。

### 3.8 评估(讲 12)

- 两版结构一致:perplexity → 涌现/leakage → 基准(GPT-2→HellaSwag→MMLU 等)→ 人工偏好(Arena)→ **agent 基准(SWE-bench、CyberSecBench、MLE-bench、ARC-AGI)**。
- 2026 的榜单示例更新到 "GPT-5.5" 时代,安全性评估(dual use、幻觉、越狱)叙述保留;其余与 2025 高度相似。

### 3.9 数据(讲 13、14)

- L13(数据来源):主线一致(Common Crawl、books、Wikipedia、shadow libraries、"模型不是训练在互联网上"的纠偏)。2026 增量:
  - **版权/法律板块大幅加厚**:2025 只在讲 shadow libraries 时提一句"legal restriction"与 NYT;2026 系统讲**美国版权法第 107 条四要素测试(目的与性质/转换性、作品性质、使用量、市场影响)、Creative Commons 许可(多次出现)、OpenAI-NYT 等诉讼语境**,并引用 Shane Lamprey 的 "Consent in Crisis" 研究展示 robots.txt 与 ToS 限制随时间收紧。
  - 数据源示例更新:提及 **Qwen 3.5 397B"没有 base model,只有最终模型"** 的现象。
- L14(清洗/去重/混合):主线一致(fastText/kenlm/classifier 过滤、minhash 类去重)。**2026 新增压轴板块——后训练合成数据**:Open Thoughts 数据管线、教师模型选择(结论:更强的模型不一定是更好的教师,**QWQ-32B 比当时的 DeepSeek R1 更优**)、采样多代(16 代)比多源更有效、answer filtering 无效等具体实验结论;顺带讲 FineWeb PDFs(OCR/VLM 转文本)。
- 总评:数据讲次在 2026 补上了"合规(fair use)与后训练合成数据"两个 2025 缺失的维度,是"紧跟 2024-2025 行业关注点"的又一例。

### 3.10 对齐 / RLVR / 多模态(讲 15、16、17)——重组幅度最大的板块

- **2025**:L15=SFT+RLHF(镜像 InstructGPT 三阶段);L16=DPO/SimPO+overoptimization+RLVR 引入;L17=策略梯度/GRPO 代码与数学深挖。
- **2026**:
  - **L15 "Mid/Post-Training"** 一口吞下 SFT + RLHF/PPO + DPO(含 DPO 闭式解推导)+ overoptimization,并新增:
    - **mid-training / decay 阶段混合指令数据**的讨论("base model 是谎言——今天所谓 base model 其实混了 UltraChat 等 chat 数据"),配 Minimax 的两阶段数据配比图;
    - **SFT 数据向 agent/tool-use 迁移**的新代际;
    - **注释者经济专题**:专家注释者时薪>$100、防 AI 作弊(annotators 用 ChatGPT 答题)、人口统计学偏差对模型行为的影响(引用其本人早期意识形态对齐研究);
    - 开头的新框架:GPT-3 →(本讲)ChatGPT →(下讲)o1 类 thinking models。
  - **L16 "Post-Training - RLVR"** 把 2025 L16 尾 + L17 全部并进来:PPO 的"37 个实现细节"陷阱、GRPO、长度归一化,然后串讲 **DeepSeek R1/R1-Zero("极简配方:base+GRPO+accuracy/format reward 就接近 o1")、aha moment 的批评(长 CoT 是长度归一化的副作用、aha 在 base 模型就存在)、Kimi K1.5**,最后讲 **RL 基础设施难点**(长 CoT 负载不均衡、答案等价性检查的"兔子洞"——数学最终也要用 reward model 而非真验证)。
  - **L17 换为多模态**:CLIP 对比学习目标、ViT 分块、attention pooling、开放复现(OpenCLIP/LAION-5B),并展望"omnimodel"与音频/视频;教师自述"本来只安排了一讲 RL,但我觉得不讲多模态这门课不完整"。**2025 的 L17(GRPO 代码深挖)在 2026 没有独立讲次,但其内容已并入 L16。**
  - **L18 为 Dan Fu guest lecture**:推理服务全景("把 token 的一生"走一遍:调度、KV cache 命中、prefill/decode 分离、跨节点并行),并给"inference engines 是 full-stack innovation 的杠杆"的观点。
- 总评:对齐板块从"3 讲偏学院派"重组为"2 讲更贴近当下(RLVR 为主角)+ 1 讲多模态 + 1 讲推理服务",知识密度与时效性都明显提升,代价是 RLHF/DPO 的展开节奏变快。

---

## 4. 课程文件与工具链对比

| 维度 | 2025 | 2026 |
|------|------|------|
| 依赖管理 | `requirements.txt`(torch/numpy/sympy/requests/sqlitedict/warcio/markdownify/tiktoken/openai/wandb/einops/jaxtyping/triton/kenlm/fasttext/mmh3/bitarray/matplotlib) | `pyproject.toml` + `uv.lock`(核心依赖仅 edtrace/einops/mmh3/modal/tiktoken,重依赖由 Modal 镜像安装) |
| 执行方式 | 本地 `python execute.py -m lecture_XX`,本地自建 trace-viewer | `modal run modal_execute.py`,把整个 lecture 目录挂进 **Modal 云镜像(nvidia/cuda:13.2.0 + B200×4)**,跑完回传 trace/PTX/profile |
| 可执行讲次 | lecture_01/02/06/08/10/12/13/14/17.py(9 个)| lecture_01/02/06/07/10/12/13/14/17.py(9 个,第 7 讲从 PDF 改回可执行,第 8 讲改 PDF) |
| 计算资源赞助 | Together AI 集群(A1 榜:H100 × 90 分钟) | **Modal(B200)**(A1 榜:**B200 × 45 分钟**,时间减半、硬件升级) |
| AI 政策 | "AI 工具可能影响学习,风险自负"一句话 | **专设 AI 政策小节**:要求使用课程提供的 **AGENTS.md**(要求 AI "pedagogically-minded")并附政策文档链接——直接回应"coding agent 能零样本完成作业"的现实 |
| 其他 | `execute_util.py/lecture_util.py`、slurm 脚本、remote_execute.sh | 移除 slurm/remote 脚本,新增 `gpu_util.py`;README 变为 uv 工作流 |

**解读**:工具链迁移(uv + Modal 云执行)意味着"可执行讲座"从"需要学生自己配齐 CUDA 环境"变成"一键在云端 B200 上跑",既降低了门槛,也配合了 2026 讲师在 L1 里反复强调的"B200/效率"主题。AI 政策从一句话升级为正式小节,是 2025→2026 之间"coding agent 能力跃迁"逼出来的教学管理迭代。

---

## 5. 教学迭代痕迹(转录中的自指证据)

1. **2026 L1**:"这是第三版… 去年我们把讲座放上 YouTube"、What's new?"更注重单位时间的价值密度,别只见树木不见森林;**更覆盖现代 LM 组件(MoE、long-context、agents)**"——总览层直接声明了本版改动方向。
2. **2026 L4**(最强烈):"**这是第一年我讲线性时间注意力**,因为现在这些方法已经在 scale 和 production 中被验证了"——明确承认该主题是 2025 的缺口。
3. **2026 L3**:"去年我以为是几篇论文… 结果 19 个新稠密模型;今年我以为会放缓,结果 Qwen 3、Gemma 4(上周四才发)… 而且大多是 MoE";并总结"去年趋势是训练稳定性,今年趋势是长上下文架构"。
4. **2026 L9**:"进阶版缩放定律讲次里,**今年我要加更多优化器内容**"——兑现为 L11 的 Muon/Kimi K2 板块。
5. **2026 L17**:"原计划是继续讲 RL,但我只剩一讲,不讲多模态这门课就不完整"——解释了 L17 被替换的直接动机。
6. **2026 L2**:分享 Marine 项目用缩放定律预测的 loss 与实际训练 loss 只差 0.05("跟预测匹配上了")——用自家实验给学生示范缩放定律的实用性,是 2025 没有的"现身说法"。
7. **2026 L16**:讲 R1-Zero 时说"这个极简配方你会在作业里复现";2025 L11 曾说"去年我还得费力论证为什么讲 DeepSeek/中国开源模型,今年大家已经自己期待了",2026 L11 则说"我再也不用为讲 DeepSeek 辩护了,太好了"——同一句话的升级,侧面反映开源模型生态 2025→2026 的变化。
8. **2025 L1(对照组)**:"这是第二次开课,扩招 50%,3 个 TA"——可见该课程每年都在做显式的版本自述,2026 的自述比 2025 更聚焦于"内容取舍"而非"规模增长"。

---

## 6. 结论:哪些改进最明显,对学习者价值提升在哪

### 最明显的五项改进

1. **补齐"2024-2026 语言模型版图":** 线性注意力/SSM/混合架构(L4)、RLVR 一线配方与 Kimi/DeepSeek 案例(L16)、多模态(L17)、推理服务(L18)、优化器前沿 Muon/L11——这五块在 2025 版里要么完全缺席(线性注意力、多模态、推理服务、优化器),要么只存在于二手引用。2026 把它们全部纳入正课,课程的"从零构建"范围从"预训练 Transformer + 经典对齐"扩展到"整个现代 LLM 栈"。
2. **对齐/RL 板块从 3 讲压缩重组为 2 讲,信息密度显著提高:** SFT/RLHF/DPO/overoptimization 一次讲完,RLVR 成为真正的主角(一讲内完成"算法→开源复现→基础设施"闭环)。
3. **数据板块补上法律与合成数据两课:** 版权/合理使用四要素、注释者经济、后训练合成数据管线——2025 只讲清洗/去重,2026 补齐了数据工作"真实世界"的另一半。
4. **工具链与算力换代:** uv + Modal 云执行 B200、作业时间减半,配合 AI 政策(AGENTS.md),让课程在"coding agent 时代"仍能保住"亲手构建"的教学目标。
5. **全课程内容年份刷新:** GPT-5 分词器演示、B200 硬件叙述、Qwen 3.5/Kimi K2/Gemma 4/GLM-5/小米 MIMO 等模型清单、OpenAI 日产 8.6T token 等数字,让每讲都锚定在最新领域状态上。

### 需要留意的代价/保留项

- 2025 整讲的 **DeepSeek V3 MoE 逐部件走查**和 **GRPO 独立代码深挖**在 2026 被压缩/并入,偏好"慢节奏代码教学"的学习者会觉得 2026 更快;RLHF/DPO 的推导节奏也相应变紧。
- 映射表中"2026 L6 = Kernels, Triton, **XLA**"与"2026 L2 = PyTorch (**einops**)"两条,经转录核实前者不成立(XLA 在 2026 转录零出现),后者也仅是标题标注(einops 内容两版都有)。
- 缩放定律基础讲次(L9)、评估(L12)、GPU(L5)、并行(L7/8)的实质性变化相对温和,主要是数字与模型引用刷新。

### 对学习者的价值提升

一句话:**2026 版把"2025 版的课程"升级成了"覆盖 2026 年 LLM 全景的课程",而且是用同样的"从零构建"方法论。** 学习者花同样的时间,能拿到的知识版图从"如何训练并对齐一个 Transformer"扩展为"如何训练、对齐、扩展上下文、引入多模态、并把它高效地服务出去"。如果你只能看一版,2026 版的信息量与时新性全面占优;如果你已经在 2025 版投入过时间,2026 版值得重点补看的讲次是 **L4(注意力替代+MoE)、L11(优化器)、L14 后半(合成数据)、L15-L17(对齐重组+多模态)、L18(推理服务)**。
