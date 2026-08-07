---
title: "06 Kernels, Triton"
lecture: 6
---

# Lecture 6: Kernels, Triton

**讲师**：Percy Liang · **主题**：Kernel 编写与 Triton（Kernels, Triton）

## 本讲内容

这一讲是系统部分的第二讲，紧接 Tatsu 在上一讲给出的 GPU 高层概览。本讲的目标是把视角从“想”转向“做”：**深入代码**，亲手用 **Triton** 写 kernel，并用**基准测试（benchmarking）与性能剖析（profiling）**来验证性能。前半讲快速回顾 GPU 的硬件与编程模型（内存层级、线程/线程块/网格），随后深入**编程模型与硬件交互**的五个性能考量——**warp 与控制发散、warp 占用率、bank 冲突、内存合并（coalescing）、块占用率（波次量化）**。第二部分讲方法论：**基准测试**（如何正确地计时）与**性能剖析**（时间到底花在哪里），并用 **GeLU** 作为案例（naive vs builtin vs `torch.compile`）说明**算子融合（kernel fusion）**的价值。第三部分是重头戏：依次手写四个难度递增的 **Triton kernel**——逐元素的 **GeLU**（顺带读懂它编译出的 **PTX** 汇编）、按行的 **softmax**（归约）、行放不进块的 **row sum**（“婴儿版 tiling”）、以及终极示例**矩阵乘法（matmul）**（真正的 tiling + 用 shared memory 复用数据 + kernel 融合）。

| 页面 | 内容 |
|------|------|
| [06 · 开场与回顾：GPU 硬件与编程模型](overview.md) | 承接上一讲、内存层级回顾（寄存器/L1/shared/L2/HBM）、A100/H100/B200 规格表、编程模型（线程/线程块/网格）、为什么需要线程块、编程模型与硬件的两难 |
| [06 · 五个性能考量：warp、占用率、bank 冲突与合并访存](hardware-considerations.md) | warp 与锁步执行、控制发散、零成本切换、warp 占用率（含 18.75% 的完整计算）、bank 冲突与 swizzling、内存合并（128 字节 cache line）、块占用率与波次量化、课堂问答 |
| [06 · 基准测试与性能剖析](benchmarking-profiling.md) | 成功配方（benchmark → 改 → 再 benchmark）、正确的计时方法（warmup、CUDA events、同步）、随维度扩展的曲线、profiler 看 add/matmul 的 kernel 名（CUTLASS、SM100、tile 形状）、GeLU 三实现对比与算子融合 |
| [06 · Triton 入门：GeLU kernel 与 PTX](triton-gelu.md) | CUDA vs Triton、以线程块为单位思考、GeLU kernel 逐行讲解（指针、program_id、mask、load/compute/store）、看 PTX（ld/st.global、%ctaid.x/%tid.x、线程粗化）、课堂问答 |
| [06 · 归约与“婴儿 tiling”：softmax 与 row sum](softmax-row-sum.md) | softmax 回顾、naive softmax 的读写计数（$5MN+M$ 读、$3MN+2M$ 写 vs 理想的 4 倍加速）、每行一个块的 Triton softmax（mask 与 $-inf$）、行放不进块时怎么办、row sum 的 tile 迭代累加与最终归约、块 vs tile 的区别 |
| [06 · 矩阵乘法：tiling 与 kernel 融合](matmul-tiling.md) | naive matmul 的 $O(1)$ 算术强度与冗余读、理想化（全量载入 shared memory → $O(N)$）、tiling 的 $O(\text{tile\_size})$、kernel 融合 bonus、stride 回顾、完整 matmul kernel 讲解、全课小结与下讲预告、课堂问答 |

## 本讲要点

- 本讲从“读图”进入“写代码”：目标不是再讲一遍 GPU，而是亲手写 kernel、做基准测试，理解**正确性与性能是两个层面的事**——编程模型保证正确，硬件细节决定快慢；
- **GPU 的内存层级**：寄存器最快最小（每 SM 256 KB，B200 为 65,536 个 32 位寄存器），L1/shared 次之（同一块物理内存，shared 可编程、L1 不可控），L2 全芯片共享，HBM 最慢最大——**大而远的内存慢，近而小的内存快**，容量与带宽大体成反比；
- **编程模型三要素**：线程（执行一小段数据的代码）、线程块/CTA（一组线程，保证被调度到同一个 SM 上、共享同一块 shared memory）、网格（线程块的集合）；逐元素运算（如 GeLU）只需线程，而 softmax、matmul 这类需要线程间通信的运算必须靠线程块 + shared memory；
- **性能的五个考量**：① **warp**——32 个线程为一组锁步执行同一条指令，分支会导致控制发散、被串行化；② **warp 占用率**——寄存器数量决定能驻留多少线程，128 线程/块 × 160 寄存器/线程 → 每 SM 只能放 3 个块（12 个 warp），占用率仅 18.75%，但低占用率未必是坏事（线程粗化）；③ **bank 冲突**——shared memory 分 32 个 bank、每 bank 4 字节宽，同 bank 并发访问被串行化，matmul 访问行与列时几乎无法避免，用 **swizzling** 重排 shared memory 缓解；④ **内存合并**——warp 访问 HBM 时按 128 字节 cache line 合并事务，沿主序连续访问才能全合并；⑤ **块占用率**——线程块按波次调度，最后一波不满导致部分 SM 空转（波次量化），应让块数尽量整除 SM 数；
- **基准测试与性能剖析是“成功的配方”**：先测量、再改动、再测量；benchmarking 只看端到端耗时（要 warmup、用 CUDA events 计时、显式 `torch.cuda.synchronize()`、多次取平均），profiling 才能看到时间花在哪个 kernel 上——**profiler 里 kernel 的名字会泄露实现**（`cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x64x16_1x1x1_3_nnn_align1_...` 表明这是 CUTLASS 为 Blackwell SM100 写的、tile 为 64×64×16 的 FP32 矩阵乘 kernel），而且**同一运算不同维度会调用不同 kernel**；
- **GeLU 三实现的对比**：naive（多 kernel、无融合，数据在 HBM 与 SM 之间反复往返）很慢；builtin 与 `torch.compile` 都是单 kernel（算子融合，只读一次 HBM、写一次 HBM）；`torch.compile` 的秘密在于它把计算图编译成了 **Triton kernel**；
- **Triton 的思维框架**：在 CUDA 中你要指定“每个线程做什么”（粒度细但要自己管理同步与 shared memory 簿记）；在 Triton 中你指定“每个线程块做什么”——**把数据 load 进 shared memory、在上面计算（可顺带融合）、再写回全局内存**；
- **GeLU kernel 的四步**：醒来（`pid = tl.program_id(0)`）→ 定位（`start = pid * BLOCK_SIZE`）→ 读入（`tl.load`，用 `mask` 防止越界）→ 计算（tanh 近似）→ 写回（`tl.store`）；Triton 把它编译成 **PTX**（GPU 的汇编），从 PTX 可以看到 `ld.global`/`st.global`、`%ctaid.x`（块号）与 `%tid.x`（线程号）、浮点/整数寄存器，以及**线程粗化**（编译器让一个线程同时处理 8 个元素）；
- **softmax 是“行内归约”**：naive 实现逐 kernel 读写，共 $5MN+M$ 次读、$3MN+2M$ 次写；理想的融合实现只要 $MN$ 次读、$MN$ 次写（约 4 倍加速）。Triton 版本**每行一个线程块**，先求行内 max（数值稳定性）、减去 max、指数化、求和、归一化，mask 掉的列填 $-inf$；
- **行放不进块时用“婴儿版 tiling”**：把一行切成多个 tile，每个线程跨 tile 迭代、把部分和累进自己的累加器，最后做一次归约（`tl.sum`）得到标量——注意 tile 与块的区别：GeLU 把行分给**独立**的块，而 row sum 里这些 tile 属于**同一个**块；
- **matmul 是 tiling 的终局**：naive 每个输出元素对每个 $k$ 都要从 HBM 读 $A[m,k]$ 与 $B[k,n]$，共 $MKN$ 次读、算术强度只有 $O(1)$；理想化（全部载入 shared memory）可把算术强度提到 $O(N)$，但矩阵太大放不下；**tiling** 折中：把 $C$ 切成输出 tile（每个 tile 一个线程块），对每个（$A$ 的行 tile、$B$ 的列 tile）对载入 shared memory 并 `tl.dot` 累加，算术强度为 $O(\text{tile\_size})$，顺便还能融合 ReLU/GeLU 这类逐元素激活；
- 这一讲的三条主线：**编程模型（PyTorch/Triton/PTX）给你正确性，硬件细节（SM 数量、bank、寄存器、内存大小）决定性能**，而 **benchmark 与 profile** 是把两者连起来的桥梁——下讲进入多 GPU 并行。

## 课程导航

- [上一讲：05 GPUs, TPUs](../05/)
- [下一讲：07 Parallelism](../07/)
