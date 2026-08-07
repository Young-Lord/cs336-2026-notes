#!/usr/bin/env python3
"""Append prev/next navigation between lectures.

- The LAST section page of each lecture gets a "→ 下一讲" footer linking to the
  next lecture's first page (its README/index).
- The FIRST section page of each lecture gets a "← 上一讲" footer linking to the
  previous lecture's index.
Idempotent: an existing `<!-- lecture-nav -->` block is replaced, not duplicated.
"""

import sys
from pathlib import Path

MARKER = "<!-- lecture-nav -->"

# (lecture dir, display title)
LECTURES_2026 = [
    ("03", "03 Architectures"),
    ("04", "04 Attention Alternatives"),
    ("05", "05 GPUs, TPUs"),
    ("06", "06 Kernels, Triton"),
    ("07", "07 Parallelism"),
    ("08", "08 Parallelism"),
    ("09", "09 Scaling Laws"),
    ("10", "10 Inference"),
    ("11", "11 Scaling Laws"),
    ("12", "12 Evaluation"),
    ("13", "13 Data (Sources, Datasets)"),
    ("14", "14 Data"),
    ("15", "15 Mid/Post-Training"),
    ("16", "16 Post-Training - RLVR"),
    ("17", "17 Alignment - Multimodality"),
    ("18", "18 Guest Lecture: Dan Fu"),
]

# pages per lecture, in reading order (must mirror docs/.vuepress/config.ts)
PAGES_2026 = {
    "03": ["overview.md", "normalization.md", "activations-ffn.md", "position-embeddings.md", "hyperparameters.md", "stability-and-attention.md"],
    "04": ["overview.md", "linear-attention.md", "ssm-hybrid.md", "sparse-attention.md", "moe.md", "moe-training.md"],
    "05": ["overview.md", "gpu-hardware.md", "tpu-and-strengths.md", "gpu-performance.md", "matrix-mystery.md", "flash-attention.md"],
    "06": ["overview.md", "hardware-considerations.md", "benchmarking-profiling.md", "triton-gelu.md", "softmax-row-sum.md", "matmul-tiling.md"],
    "07": ["overview.md", "collectives.md", "hardware.md", "torch-distributed.md", "data-and-tensor-parallelism.md", "pipeline-parallelism.md"],
    "08": ["01-motivation-and-networking.md", "02-data-parallelism-and-zero.md", "03-pipeline-and-tensor-parallelism.md", "04-activation-sequence-expert.md", "05-scale-and-case-studies.md"],
    "09": ["01-introduction-and-history.md", "02-data-scaling-laws.md", "03-model-engineering-scaling.md", "04-data-versus-model-size.md", "05-compute-optimal-and-chinchilla.md"],
}

LECTURES_2025 = [
    ("01", "01 Overview and Tokenization"),
    ("02", "02 Pytorch, Resource Accounting"),
    ("03", "03 Architectures, Hyperparameters"),
    ("04", "04 Mixture of Experts"),
    ("05", "05 GPUs"),
    ("06", "06 Kernels, Triton"),
    ("07", "07 Parallelism 1"),
    ("08", "08 Parallelism 2"),
    ("09", "09 Scaling Laws 1"),
    ("10", "10 Inference"),
    ("11", "11 Scaling Laws 2"),
    ("12", "12 Evaluation"),
    ("13", "13 Data 1"),
    ("14", "14 Data 2"),
    ("15", "15 Alignment - SFT/RLHF"),
    ("16", "16 Alignment - RL 1"),
    ("17", "17 Alignment - RL 2"),
]

PAGES_2025 = {
    "01": ["overview.md", "course-structure.md", "five-pillars.md", "tokenization-basics.md", "bpe.md"],
    "02": ["overview.md", "memory-accounting.md", "tensor-operations.md", "compute-accounting.md", "gradients.md", "models-and-training.md"],
    "03": ["overview.md", "normalization.md", "activations-ffn.md", "position-embeddings.md", "hyperparameters.md", "stability-and-attention.md"],
    "04": ["overview.md", "routing.md", "experts.md", "training.md", "systems-and-stability.md", "deepseek-v3.md"],
}


def build_block(prev_info, next_info) -> str:
    parts = [MARKER, ""]
    if prev_info:
        parts.append(f"**← 上一讲**:[{prev_info[1]}](../{prev_info[0]}/)")
    if next_info:
        parts.append(f"**→ 下一讲**:[{next_info[1]}](../{next_info[0]}/)")
    parts.append("")
    return "\n".join(parts)


def apply_nav(base: Path, order: list[tuple[str, str]], pages_map: dict[str, list[str]]) -> None:
    for idx, (lecture, title) in enumerate(order):
        pages = pages_map.get(lecture)
        if not pages:
            continue
        prev_info = order[idx - 1] if idx > 0 else None
        next_info = order[idx + 1] if idx + 1 < len(order) else None

        first_page = base / lecture / pages[0]
        last_page = base / lecture / pages[-1]

        for target, info, direction in (
            (first_page, prev_info, "prev"),
            (last_page, next_info, "next"),
        ):
            if info is None:
                continue
            text = target.read_text(encoding="utf-8").rstrip() + "\n\n"
            block = build_block(prev_info if direction == "prev" else None,
                                next_info if direction == "next" else None)
            if MARKER in text:
                # replace existing nav block
                start = text.find(MARKER)
                # find the enclosing block start (previous blank line + block)
                prefix = text[:start]
                # remove the trailing blank line before marker
                prefix = prefix.rstrip("\n")
                # drop a preceding '---' divider if present
                if prefix.endswith("\n---"):
                    prefix = prefix[: -len("\n---")]
                text = prefix.rstrip() + "\n\n" + block
            else:
                text = text.rstrip() + "\n\n---\n\n" + block
            target.write_text(text, encoding="utf-8")
            print(f"{target.relative_to(base)}: {direction} -> {info[0] if info else None}")


def main() -> None:
    root_2026 = Path("/home/niko/Projects/cs336-2026-notes")
    apply_nav(root_2026 / "docs" / "lectures", LECTURES_2026, PAGES_2026)
    root_2025 = Path("/home/niko/Projects/cs336-2025-notes")
    apply_nav(root_2025 / "docs" / "lectures", LECTURES_2025, PAGES_2025)


if __name__ == "__main__":
    sys.exit(main())
