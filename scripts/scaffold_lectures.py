#!/usr/bin/env python3
"""Create the per-lecture directory skeleton for the 2026 notes site."""

from pathlib import Path

DOCS = Path("docs/lectures")

LECTURES = [
    ("01-02-diff", "2025 → 2026 第 1/2 讲差异", "1", "01-02-diff"),
    ("03", "Architectures", "3", "03"),
    ("04", "Attention Alternatives", "4", "04"),
    ("05", "GPUs, TPUs", "5", "05"),
    ("06", "Kernels, Triton", "6", "06"),
    ("07", "Parallelism", "7", "07"),
    ("08", "Parallelism", "8", "08"),
    ("09", "Scaling Laws", "9", "09"),
    ("10", "Inference", "10", "10"),
    ("11", "Scaling Laws", "11", "11"),
    ("12", "Evaluation", "12", "12"),
    ("13", "Data (Sources, Datasets)", "13", "13"),
    ("14", "Data", "14", "14"),
    ("15", "Mid/Post-Training", "15", "15"),
    ("16", "Post-Training - RLVR", "16", "16"),
    ("17", "Alignment - Multimodality", "17", "17"),
    ("18", "Guest Lecture: Dan Fu", "18", "18"),
]

for folder, title, lecture, prev in LECTURES:
    out = DOCS / folder
    out.mkdir(parents=True, exist_ok=True)
    readme = out / "README.md"
    if readme.exists():
        continue
    readme.write_text(
        f"---\ntitle: \"{title}\"\nlecture: {lecture}\n---\n\n"
        f"# {title}\n\n> 本讲内容整理中,将按小节拆分为多个页面。\n",
        encoding="utf-8",
    )
    print(f"created {folder}/README.md")
