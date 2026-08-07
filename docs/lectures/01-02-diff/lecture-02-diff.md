---
title: "第 2 讲差异:PyTorch"
lecture: "1-2"
---

# 第 2 讲差异：PyTorch, Resource Accounting

**讲师**: Percy Liang · **对比对象**: 2025 版 L2 vs 2026 版 L2（课程表标题为 "PyTorch (einops)")

## 2025 第 2 讲骨架回顾（唤起记忆）

- **两个动机题（餐巾纸数学）**:① 1024 张 H100 训练 70B 参数 / 15T token 要多久（≈144 天，用到 $6\times\text{参数}\times\text{token}$）；② 8 张 H100 + AdamW 能装下多大模型（≈40B，每参数 16 字节）。
- **内存核算**：张量是存储一切的积木；**内存 = 元素数 × 每元素字节数**;fp32(1+8+23 位)/ fp16 / bf16(2018，动态范围同 fp32)/ fp8（2022,E4M3/E5M2）的位布局与动态范围；混合精度概念。
- **张量在 GPU 上**:storage/stride/view/contiguity、逐元素操作、matmul 对 batch/seq 的隐式广播。
- **einops 插曲**：动机（$x @ y.transpose(-2,-1)$ 容易写错）、**jaxtyping**（`Float[torch.Tensor, "batch seq hidden"]`）、`einsum` / `reduce` / `rearrange` 三个例子。
- **计算核算**:FLOPs vs FLOP/s 两个混淆的缩写；matmul ≈ $2\times$ 三维尺寸乘积；**MFU = 实际 FLOP/s ÷ 承诺 FLOP/s**,$\ge 0.5$ 算不错；H100 峰值 1979 teraFLOP/s（稀疏，实际取一半）。
- **梯度**：简单线性模型链式法则；w2.grad、h1.grad 各是一个 matmul；前向 2×、反向 4×、**总 6×数据点×参数**。
- **模型与训练**:nn.Parameter、Xavier 初始化（除以 $\sqrt{d}$）、Cruncher 深线性网络、随机种子、memmap 数据加载、SGD/AdaGrad/RMSProp/Adam 直觉、AdaGrad 实现、训练循环、**检查点（把模型+优化器存盘）**、混合精度总结。

---

## 2026 新变化详解

> 标注说明:**〔2026 新增,需学习〕** 表示 2025 没有(或只在别处提到);**〔年份刷新,了解即可〕** 表示结构相同、只换数字/例子;**〔相同,可跳过〕** 表示可完全略过。

### 1. Marine 项目:"缩放定律预测命中"的现身说法〔2026 新增,重点〕

2025 第 2 讲直接以两个动机题开场;2026 则**先晒一个实验结果**再进入正题:

- **Marin 项目**(Percy 参与的开放开发项目)有一次 **$10^{23}$ FLOPs 量级的训练**跑完了,**用缩放定律预注册(preregistered)预测的 loss 与实际训练 loss 只差 0.05**,"跟预测匹配上了"(matched the forecast)。
- 课上展示了若干条 **iso-FLOP 曲线**:每条曲线是一组小模型跑出来的,在曲线上找 compute-optimal 点、拟合缩放定律、外推到目标规模——这次真的被验证了。
- Percy 还顺带说:如果把这条定律外推到 GPT-5 级别,就能得到对应的 loss(当然,缩放定律的外推结果会因方法而异,mileage may vary)。
- **为什么值得记**:这是给"缩放定律能预测、能预注册"提供**第一方真实证据**——呼应第 3/9/11 讲的缩放定律单元,也是整个课程"效率至上"叙事的锚点。

### 2. 算术强度(arithmetic intensity)与 roofline 分析前置〔2026 新增,重点学习,完整推导〕

**这是本讲最大的一块新增内容。** 在 2025 版里,算术强度/roofline 属于后续 GPU 讲座(第 5 讲)的内容;2026 把它**前移到第 2 讲**,用来当场回答上一节留下的问题——"为什么 MFU 通常只有 0.5,而不是接近 1?"。

#### 2.1 硬件视角:计算与通信

硬件可以抽象成两层:离片外的**高带宽内存(HBM)** 与真正的**计算单元(加速器)**。做任何计算都要三步:

1. 把输入从内存搬到加速器;
2. 执行计算;
3. 把输出从加速器搬回内存。

于是一次计算的总耗时由**两个速度**共同决定:

- 计算速度:加速器峰值 $\Pi$(FLOP/s);
- 通信速度:内存带宽 $\beta$(bytes/s)。

记读入/写出的字节数为 $B_{\text{bytes}}$,总 FLOPs 为 $F$,则

$$
T_{\text{comm}} = \frac{B_{\text{bytes}}}{\beta}, \qquad
T_{\text{compute}} = \frac{F}{\Pi}.
$$

课程的关键假设:**通信与计算可以完美重叠(overlap)**——数据一到就开始算,算完就搬回去,两边同时进行。因此

$$
T_{\text{total}} = \max(T_{\text{comm}},\ T_{\text{compute}}).
$$

- 若 $T_{\text{comm}} > T_{\text{compute}}$:瓶颈是"等字节搬过来",称为 **memory-bound**;
- 若 $T_{\text{compute}} > T_{\text{comm}}$:瓶颈是真正做算术,称为 **compute-bound**。

#### 2.2 加速器强度与算法算术强度

把上面两个时间比改写成分数形式,可以得到更顺手的判据:

- **加速器强度**(accelerator intensity):该硬件每搬 1 字节能做多少 FLOP,

$$
I_{\text{accel}} = \frac{\Pi}{\beta}.
$$

对 H100:$\Pi = 1979\times10^{12}/2 \approx 9.9\times10^{14}$ FLOP/s(手册给的 1979 是**含稀疏**的峰值,密集要减半;转录里说"这个数字除以二"),$\beta = 3.35\times10^{12}$ bytes/s,于是

$$
I_{\text{accel}} \approx \frac{989.5\times10^{12}}{3.35\times10^{12}} \approx 295,
$$

**约等于 300**——H100 每搬 1 字节能做约 295 次浮点运算,这个数值得记在脑子里。

- **算法算术强度**(arithmetic intensity):这个工作负载本身每搬 1 字节做了多少"有用"的 FLOP,

$$
I_{\text{alg}} = \frac{F}{B_{\text{bytes}}}.
$$

判据($\approx$ 两个分式交叉相乘):

$$
I_{\text{alg}} < I_{\text{accel}} \iff \text{memory-bound}, \qquad
I_{\text{alg}} > I_{\text{accel}} \iff \text{compute-bound}.
$$

#### 2.3 逐个操作算一遍(重要结果)

以下都按 bf16(每数 2 字节)计,设 $n$ 为向量长度:

**ReLU(逐元素激活)**:读 $x$(2n 字节),写 $y$(2n 字节),$B_{\text{bytes}} = 4n$;$F = n$(n 次与 0 比较)。

$$
I_{\text{alg}} = \frac{n}{4n} = 0.25 \ll 295 \implies \text{memory-bound}.
$$

**GELU(逐元素激活)**:读 $x$、写 $y$ 同样是 $4n$ 字节,但每元素要做约 20 次 FLOP(tanh 用多项式近似,公式为 $\mathrm{GELU}(x)=0.5x(1+\tanh(\sqrt{2/\pi}\,(x+0.044715x^3)))$),$F \approx 20n$,

$$
I_{\text{alg}} \approx 5 \quad \text{仍远小于 295} \implies \text{memory-bound}.
$$

要点:**GELU 每字节做的活比 ReLU 多,算术强度更高,但两者都是 memory-bound——所以"单独看"时 GELU 并不比 ReLU 慢。** 平时优化激活函数省下的时间微不足道,瓶颈根本不在这。

**向量内积(dot product)**:读 $x$(2n)、读 $w$(2n)、写一个标量(可忽略),$B_{\text{bytes}}\approx 4n$;$F = 2n-1 \approx 2n$(n 次乘、n-1 次加),

$$
I_{\text{alg}} \approx \frac{2n}{4n} = 0.5 \implies \text{memory-bound}.
$$

**矩阵-向量乘(matvec)**:$w$ 是 $n\times n$,$B_{\text{bytes}} = 2n + 2n^2 + 2n \approx 2n^2$;$F = n(2n-1)\approx 2n^2$,

$$
I_{\text{alg}} \approx \frac{2n^2}{2n^2} \approx 1 \ll 295 \implies \text{memory-bound}.
$$

**矩阵-矩阵乘(matmul)**:两个 $n\times n$ 矩阵,$B_{\text{bytes}} = 6n^2$(读两个、写一个);$F = n^2(2n-1)\approx 2n^3$,

$$
I_{\text{alg}} \approx \frac{2n^3}{6n^2} = \frac{n}{3}.
$$

$n=1024$ 时 $I_{\text{alg}}\approx 341 > 295$,**compute-bound**!

**为什么**:matvec/matmul 都在搬 $O(n^2)$ 的数据,但 matmul 算了 $O(n^3)$ 次运算,所以算术强度随 $n$ 增长——**矩阵越大越能饱和加速器**。这就是"为什么需要大 batch、大矩阵"的数学解释;也是为什么课程说 **Transformer 训练本质是"大 matmul + 中间撒点逐元素操作"**,整体 compute-bound、能把 MFU 做到接近 0.5 甚至更高。

#### 2.4 联系推理:为什么 decode 是 memory-bound

这一点直接为第 10 讲(推理)埋伏笔:

- **训练/预填充(prefill)**:一次处理整条序列,是 matmul → compute-bound;
- **推理解码(decode)**:一次只生成一个 token,每个 token 本质上是"一个向量去乘权重矩阵"——**matvec → memory-bound**。

所以在训练里"矩形象限"的漂亮性质,在推理时享受不到;这也是推理优化(量化、KV cache、批处理、投机解码)存在的根本原因。

#### 2.5 roofline 图与 MFU 的关系

把"算术强度 vs 可实现性能"画成图就是 **roofline 图**:横轴为算术强度 $I_{\text{alg}}$(每个算法对应一个竖切面),纵轴为可实现 FLOP/s,每条折线对应一个硬件:

- 在拐点($I_{\text{accel}}$)左侧,性能受带宽限制,呈斜率为 $\beta$ 的斜线(升不上去);
- 在拐点右侧,性能受峰值算力限制,是水平的天花板 $\Pi$。

于是可以写出**算数强度版的 MFU**:

$$
\text{MFU} = \min\left(1,\ \frac{I_{\text{alg}}}{I_{\text{accel}}}\right).
$$

这就是"MFU 为什么通常 0.5 左右"的答案:很多操作(或操作组合)的算术强度达不到加速器强度,GPU 有一半时间在等内存。课上还留了个思考题:能不能设计算术强度比更好的加速器?(答案:去跟 Jensen 说。)

### 3. FP4 / NVFP4 与 NeMo-3 Super〔2026 新增,重点〕

2025 的精度谱系只讲到 FP8(E4M3 / E5M2 两种变体);2026 往下多走了一步:

- **NVFP4**:2025 年 NVIDIA 开发,每个值只有 **4 位**。可表示的值基本上就一行:

$$
-6,\ -4,\ -3,\ -2,\ -1.5,\ -1,\ -0.5,\ 0,\ 0.5,\ 1,\ 1.5,\ 2,\ 3,\ 4,\ 6.
$$

- **关键技巧——按块缩放(block scaling)**:如果只允许这 15 个值,表达力显然不够。真实做法是**每个块(block)共用一个 scale 因子**:块内每个值仍有 4 位自由,但整块可以整体放大/缩小,从而覆盖更大的动态范围——代价是**块内各值不能独立地"自由变化"**(转录里有个学生追问,Percy 确认:块内比例关系仍是 FP4 那 15 个值)。
- **现身说法**:2026 年发布的 **NeMo-3 Super**(转录语音为 "NeMo-3 Super",讲义脚本引文写作 **Nemotron 3 Super**)就是**用 FP4 训练的**,"我觉得这挺酷的"。
- **实操提示**:FP4 这类精度大多藏在 **NVIDIA 软件栈底层**(Transformer Engine 等)自动处理,你没法在 PyTorch 里 `torch.zeros(..., dtype=torch.float4)` 直接创建——知道原理即可,不必纠结 API。
- 补一句:2026 也保留了 FP8 的 E4M3(范围 $[-448,448]$)/ E5M2($[-57344,57344]$)介绍,与 2025 相同,可跳过。

### 4. einops 的讲授方式变化〔2026 变化〕

- **地位提升**:2026 课程表把第 2 讲标题写成 **"PyTorch (einops)"**,einops 从"插曲"变成了正式卖点;转录里 Percy 还现场调查"多少人用过 einsum?"(约三分之二)。
- **删掉了 jaxtyping**:2025 有一小节 `jaxtyping_basics`(用 `Float[torch.Tensor, "batch seq hidden"]` 做维度标注);2026 **不再讲 jaxtyping**,只保留 `einsum` / `reduce` / `rearrange` 三个工具,例子本体几乎不变。
- **例子对象更新**:2026 用 **DeepSeek v3.2** 的 `safetensors` 展示"一个模型就是一堆不同 shape、不同精度的张量";并在张量基础部分新增 **rank-4 张量示例** $[B,S,H,D]$(batch/sequence/heads/hidden),直接对应 attention 的形状。

### 5. 硬件数字更新〔年份刷新,了解即可〕

- **动机题第 2 问的答案变了:40B → 53B**。原因是每参数字节数的口径更新:
  - 2025:全用 fp32,每参数 $4+4+(4+4)=16$ 字节 → $8\times 80\text{GB}/16\approx 40\text{B}$ 参数;
  - 2026:参数、梯度用 **bf16(2+2)**,优化器状态用 fp32(4+4),每参数 $2+2+(4+4)=12$ 字节 → $8\times 80\text{GB}/12\approx 53\text{B}$ 参数。
  - 这更贴近 2026 默认的 bf16 混合精度训练惯例。两道题都仍注明"不含激活,是上界"。
- **硬件焦点**:2025 同时列 A100(312 teraFLOP/s)与 H100;2026 只聚焦 H100(峰值 $1979\times10^{12}/2$,含稀疏减半),并新增 H100 内存带宽 **3.35 TB/s**(这就是上面 $I_{\text{accel}}\approx 295$ 的来源);**B200(2.25 PFLOP/s bf16、8 TB/s)在上一讲的系统预览里预告**,本讲的具体计算仍以 H100 为例。
- **FLOPs 直觉刷新**:保留 GPT-3 3.14e23 FLOPs,新增"GPT-4 推测约 2e25 FLOPs";**删掉了 2025 的"美国行政命令:$\ge 10^{26}$ FLOPs 的模型需上报(2025 已撤销)"** 那段。
- **8×H100 两周总 FLOPs** 的例子两版都有,数值相同。

### 6. 梯度累积与激活检查点〔2026 新增,重点〕

2025 只在结尾**一句话**带过 activation checkpointing("我们以后再讲"),而且 2025 的 "checkpointing" 小节讲的是**把模型存盘**(训练崩溃保护)。2026 把两个真正的**显存优化技巧**完整展开(并保留存盘检查点的讨论):

#### 6.1 梯度累积(gradient accumulation)

动机:**大 batch 提升训练稳定性**,但激活内存随 batch 大小线性增长,容易 OOM。做法:

- 把大 batch 拆成若干个 micro-batch;
- 对每个 micro-batch 前向 + 反向,**累积梯度(不清零)**;
- 每 `batch_size / micro_batch_size` 步,才真正 `optimizer.step()` 并清零梯度。

一个非常简单的代码改动,就能在不改变有效 batch 的前提下把激活内存从 $O(B)$ 降到 $O(B/m)$。

#### 6.2 激活检查点(activation checkpointing / gradient checkpointing / rematerialization)

背景:训练需要**所有层的激活**(反向传播要用),内存为 $O(L\cdot B\cdot D)$(bf16 下每层 $2BD$ 字节);推理不需要梯度,只需当前层激活。

核心思想:**用计算换内存**——前向只存部分层的激活,反向时从最近一个检查点**重新计算**缺掉的激活。用 `torch.utils.checkpoint.checkpoint` 包一层即可:

```python
# KEY: only store activations at checkpoints, recompute the rest
x = torch.utils.checkpoint.checkpoint(layer, x)
```

检查点密度的权衡(设 $L$ 层):

| 策略 | 激活内存 | 重算开销 |
|------|----------|----------|
| 每层都存 | $O(L)$ | 无 |
| 一层都不存 | $O(1)$ | $O(L^2)$(每层都从输入重算) |
| **每 $\sqrt{L}$ 层存一个** | $O(\sqrt{L})$ | $O(L)$ |

直观解释"甜点"为何是 $\sqrt{L}$:检查点个数为 $L/\sqrt{L}=\sqrt{L}$,每个检查点区间内反向要重算 $\sqrt{L}$ 层,总重算 $\approx\sqrt{L}\times\sqrt{L}=L$(转录里的表述是"内存与重算都是 $\sqrt{L}$ 量级、达到平衡",指每段区间;讲义脚本则记为总重算 $O(L)$)。$O(1)$ 内存的方案在极端深网络里也可以接受,只是慢很多。

### 7. 其它更新〔年份刷新 / 新增〕

- **混合精度给了实际代码**:2025 只讲概念;2026 给出 `torch.amp.autocast("cuda", dtype=torch.bfloat16)` 的用法(matmul 安全降 bf16,`exp` 等保持高精度)。
- **优化器内存口径**:2026 明确 AdaGrad 每参数 4 字节(存二阶矩)、**Adam 每参数 8 字节**(一阶 + 二阶矩),并强调优化器状态"不卡计算速度,但决定模型能不能塞进显存"——和动机题第 2 问的 12 字节/参数 是同一套口径。
- **梯度计数示例升级**:2025 用 $x\!\to\!h_1\!\to\!h_2$ 两层线性网络手推 w2.grad / h1.grad;2026 改用 **einsum 记法**推同一个结论(h1.grad 对 out 维求和、w2.grad 对 batch 维求和,FLOPs 都是三维乘积),并强调"反向是前向的 2 倍,因为要算两个梯度(对输入 + 对参数)"——结论 $6\times\text{数据点}\times\text{参数}$ 不变。
- **新增引文**:Adagrad(2011)、Nemotron 3 Super、DeepSeek v3.2 等。
- **开场的课程公告**:加入 Slack、用 Stanford 邮箱注册 Modal、读 AI policy / cluster 指南——与第 1 讲的新安排一致。

### 8. 相同,可跳过

- 张量创建、内存公式(元素数 × 每元素字节)、fp32/fp16/bf16 位布局与动态范围对比、1e-8 在 fp16 下溢出的例子;
- 张量 storage/stride/view/contiguity、逐元素操作、matmul 广播;
- einops 的 `einsum` / `reduce` / `rearrange` 三个例子本身(2026 只是删了 jaxtyping、换了展示对象);
- FLOPs 计数:matmul ≈ $2\times$ 三维乘积;前向 $2\times$、反向 $4\times$、总 $6\times$ 的推导;
- MFU 的定义与"$\ge 0.5$ 不错"的经验值(新增的只是与算术强度的联系);
- AdaGrad 的 `step()` 实现、训练循环、随机种子、memmap 数据加载;
- 参数初始化(Xavier、除以 $\sqrt{d}$、截断正态)与 Cruncher/DeepNetwork 深线性模型本身。

---

## 小结

第 2 讲是"**内容前移 + 局部增厚**"的一讲：骨架（张量 → 内存/计算核算 → 梯度 → 优化器 → 训练）完全不变，但 2026 把**算术强度/roofline** 从 GPU 讲座前置进来（需要完整掌握：加速器强度 ≈ 295、四种操作的算术强度与 memory/compute-bound 判据、decode 是 matvec 所以 memory-bound、MFU = min(1, $I_{\text{alg}}/I_{\text{accel}}$)），并新增 **FP4/NVFP4 + NeMo-3 Super**、**梯度累积**、**激活检查点** 与 **Marine 0.05 案例**。einops 少了 jaxtyping，动机题数字从 40B 更新到 53B（bf16 口径）。其余内容与 2025 相同，可直接沿用。
