---
home: true
title: Home
heroText: CS336 2026 课程笔记
tagline: "Stanford CS336: Language Modeling from Scratch (Spring 2026) 中文详尽课程笔记"
actions:
  - text: 开始阅读
    link: /lectures/03/
    type: primary
---

## 关于本项目

本笔记整理自 **Stanford CS336: Language Modeling from Scratch (Spring 2026,第三版)** 课程:

- **视频来源**: [YouTube 播放列表](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)(18 讲)
- **内容来源**: 每讲视频的完整语音转录 + 官方课程文件(幻灯片 PDF、可执行 `lecture_*.py`、课件图片)
- **整理方式**: 在保留全部授课内容(含发散与口语表述)的基础上校对为完整、可流畅阅读的讲课内容;补全语音中未体现的公式与推导;公式使用 LaTeX 编写
- **组织结构**: 每讲划分为若干小节,包含详尽的讲解、适当的代码与图片

> 说明:本网站从**第 3 讲**开始。第 1、2 讲与 2025 版差异不大,已整理为
> [「2025 → 2026 第 1/2 讲差异」](./lectures/01-02-diff/README.md) 供已学过 2025 版的同学快速掌握增量内容。

## 课程目录

| # | 主题 | 讲师 |
|---|------|------|
| 01 | Overview, Tokenization | Percy Liang |
| 02 | PyTorch (einops) | Percy Liang |
| [03](./lectures/03/) | Architectures | Tatsunori Hashimoto |
| [04](./lectures/04/) | Attention Alternatives | Tatsunori Hashimoto |
| [05](./lectures/05/) | GPUs, TPUs | Tatsunori Hashimoto |
| [06](./lectures/06/) | Kernels, Triton | Percy Liang |
| [07](./lectures/07/) | Parallelism | Tatsunori Hashimoto |
| [08](./lectures/08/) | Parallelism | Percy Liang |
| [09](./lectures/09/) | Scaling Laws | Tatsunori Hashimoto |
| [10](./lectures/10/) | Inference | Percy Liang |
| [11](./lectures/11/) | Scaling Laws | Tatsunori Hashimoto |
| [12](./lectures/12/) | Evaluation | Percy Liang |
| [13](./lectures/13/) | Data (Sources, Datasets) | Percy Liang |
| [14](./lectures/14/) | Data | Percy Liang |
| [15](./lectures/15/) | Mid/Post-Training | Tatsunori Hashimoto |
| [16](./lectures/16/) | Post-Training - RLVR | Tatsunori Hashimoto |
| [17](./lectures/17/) | Alignment - Multimodality | Tatsunori Hashimoto |
| 18 | Guest Lecture: Dan Fu | Dan Fu (UCSD / Together) |

## 使用

```bash
npm install
npm run dev   # 本地预览 http://localhost:8080
npm run build # 构建静态站点
```
