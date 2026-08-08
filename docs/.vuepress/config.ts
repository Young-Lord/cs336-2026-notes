import { defineUserConfig } from "vuepress";
import { viteBundler } from "@vuepress/bundler-vite";
import { defaultTheme } from "@vuepress/theme-default";
import { markdownMathPlugin } from "@vuepress/plugin-markdown-math";

/** 每讲的元信息;`pages` 列出本讲拆分的子页面(按阅读顺序)。 */
const lectures = [
  { text: "2025 → 2026 第 1/2 讲差异", prefix: "/lectures/01-02-diff/", pages: ["lecture-01-diff.md", "lecture-02-diff.md"] },
  { text: "03 Architectures", prefix: "/lectures/03/", pages: ["overview.md", "normalization.md", "activations-ffn.md", "position-embeddings.md", "hyperparameters.md", "stability-and-attention.md"] },
  { text: "04 Attention Alternatives", prefix: "/lectures/04/", pages: ["overview.md", "linear-attention.md", "ssm-hybrid.md", "sparse-attention.md", "moe.md", "moe-training.md"] },
  { text: "05 GPUs, TPUs", prefix: "/lectures/05/", pages: ["overview.md", "gpu-hardware.md", "tpu-and-strengths.md", "gpu-performance.md", "matrix-mystery.md", "flash-attention.md"] },
  { text: "06 Kernels, Triton", prefix: "/lectures/06/", pages: ["overview.md", "hardware-considerations.md", "benchmarking-profiling.md", "triton-gelu.md", "softmax-row-sum.md", "matmul-tiling.md"] },
  { text: "07 Parallelism", prefix: "/lectures/07/", pages: ["overview.md", "collectives.md", "hardware.md", "torch-distributed.md", "data-and-tensor-parallelism.md", "pipeline-parallelism.md"] },
  { text: "08 Parallelism", prefix: "/lectures/08/", pages: ["01-motivation-and-networking.md", "02-data-parallelism-and-zero.md", "03-pipeline-and-tensor-parallelism.md", "04-activation-sequence-expert.md", "05-scale-and-case-studies.md"] },
  { text: "09 Scaling Laws", prefix: "/lectures/09/", pages: ["01-introduction-and-history.md", "02-data-scaling-laws.md", "03-model-engineering-scaling.md", "04-data-versus-model-size.md", "05-compute-optimal-and-chinchilla.md"] },
  { text: "10 Inference", prefix: "/lectures/10/", pages: ["01-inference-overview.md", "02-arithmetic-intensity.md", "03-reducing-kv-cache.md", "04-quantization-and-pruning.md", "05-speculative-decoding.md", "06-dynamic-workloads.md"] },
  { text: "11 Scaling Laws", prefix: "/lectures/11/", pages: ["01-introduction-and-motivation.md", "02-minicpm-scaling-recipe.md", "03-wsd-and-deepseek.md", "04-recent-scaling-recipes.md", "05-stepfun-and-optimizer-scaling.md", "06-muon-and-mup.md"] },
  { text: "12 Evaluation", prefix: "/lectures/12/", pages: ["01-what-is-good-and-perplexity.md", "02-exam-benchmarks.md", "03-chat-benchmarks.md", "04-agentic-benchmarks.md", "05-reasoning-and-safety-benchmarks.md", "06-realism-validity-and-conclusion.md"] },
  { text: "13 Data (Sources, Datasets)", prefix: "/lectures/13/", pages: ["01-why-data-matters.md", "02-origin-of-data.md", "03-copyright-and-fair-use.md", "04-source-infrastructure.md", "05-model-datasets-2019-2022.md", "06-model-datasets-2023-2024.md"] },
  { text: "14 Data", prefix: "/lectures/14/", pages: ["01-data-transformation.md", "02-filtering-framework.md", "03-filtering-in-practice.md", "04-deduplication.md", "05-data-mixing.md", "06-post-training-synthetic-data.md"] },
  { text: "15 Mid/Post-Training", prefix: "/lectures/15/", pages: ["01-introduction-and-sft-data.md", "02-sft-data-style-and-pitfalls.md", "03-midtraining-and-sft-method.md", "04-rlhf-data-and-annotation.md", "05-ppo.md", "06-dpo-and-side-effects.md"] },
  { text: "16 Post-Training - RLVR", prefix: "/lectures/16/", pages: ["01-introduction-rlvr.md", "02-ppo-revisited.md", "03-grpo.md", "04-deepseek-r1.md", "05-kimi-k15.md", "06-qwen3-and-agentic-rl.md"] },
  { text: "17 Alignment - Multimodality", prefix: "/lectures/17/", pages: ["01-introduction-and-omni-models.md", "02-clip.md", "03-siglip.md", "04-llava.md", "05-qwen-vl.md", "06-chameleon-and-summary.md"] },
  { text: "18 Guest Lecture: Dan Fu", prefix: "/lectures/18/", pages: ["01-introduction-and-motivation.md", "02-lifetime-of-a-token.md", "03-prefill-decode-and-batching.md", "04-kv-cache-and-parallelism.md", "05-serving-at-scale.md", "06-megakernels-and-parcae.md"] },
];

const sidebar = lectures.map(({ text, prefix, pages }) => ({
  text,
  collapsible: pages.length > 0,
  children: [prefix, ...pages.map((page) => `${prefix}${page}`)],
}));

export default defineUserConfig({
  base: "/cs336-2026-notes/",
  lang: "zh-CN",
  title: "CS336 2026 课程笔记",
  description:
    "Stanford CS336: Language Modeling from Scratch (Spring 2026) — 从零构建语言模型的中文详尽课程笔记",
  bundler: viteBundler(),
  theme: defaultTheme({
    navbar: [
      { text: "Home", link: "/" },
      { text: "Lectures", link: "/lectures/03/" },
    ],
    sidebar,
  }),
  head: [
    ["meta", { name: "keywords", content: "CS336, Language Model, LLM, Transformer, 课程笔记" }],
  ],
  plugins: [
    markdownMathPlugin({
      type: "katex",
    }),
  ],
});
