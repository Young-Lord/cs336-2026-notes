---
title: "12 · 智能体基准：SWE-bench 到 MLE-Bench"
lecture: 12
---

# 智能体基准：SWE-bench 到 MLE-Bench

## 从"说什么"到"做什么"

到目前为止，我们评估的一直是语言模型**说什么**——那是聊天。这一节转向评估语言模型**做什么**，也就是**智能体（agent）**。但凡关注 AI 热潮的人都知道，智能体如今无处不在，而且讲者认为它们将改变我们思考语言模型的方式。

先厘清概念。**智能体 = 语言模型 + agent scaffold（智能体脚手架）**。scaffold 是"决定如何调用语言模型"的那部分逻辑：模型什么时候被调用、能访问哪些工具、如何把结果拼回去，等等。要注意，我们评估的不再仅仅是模型本身——下面会看到这有多重要。

这类任务有一个共同特征：**需要使用工具（比如运行代码），并且要在一段时间内反复迭代**。

## SWE-bench：提交一个能通过测试的 PR

**SWE-bench** 是最流行的智能体基准之一。任务是：给你一个**代码库**（codebase）和一份 **GitHub issue 的描述**，你要**提交一个 pull request**（PR）去修复它。而评估 PR 的方式非常简单直接：**看它能不能通过单元测试（unit tests）**。

这些测试是精心设计的：提交 PR 之前，某些单元测试不通过（这正是 issue 要修的 bug）；你的目标是修复它——让这些测试通过，同时**不破坏其他任何东西**（原有测试仍然通过）。

![SWE-bench 的任务示例：issue 描述与上下文代码](/lectures/12/swebench.png)

SWE-bench 包含 **2294 个任务、横跨 12 个 Python 代码仓库**。好的地方在于**评估非常明确**：没有主观性，测试过了就是过了。

SWE-bench 的成绩曲线很有戏剧性。下面是 SWE-bench **Verified**（后面会解释为什么会有这个版本）的成绩：2024 年还在 **16%** 左右，到现在已经涨到 **93%**——"干得漂亮，Mythos"。SWE-bench 可以说是这类基准的鼻祖：当大量注意力投向"智能体的编码能力"时，正是它开创了"在一个真实环境里评估智能体写代码能力"的思路。

## Terminal-Bench：通用的终端环境

**Terminal-Bench** 试图做得更通用。它的环境就是一台**计算机终端（computer terminal）**——用键入命令的方式完成任务。这个环境的好处是**简单且普适**：几乎任何"在电脑上能做的事"都可以表示成终端任务。

![Terminal-Bench 的任务示例](/lectures/12/terminal-bench.png)

它的任务由 **93 位来自世界各地的贡献者众包**而来，总共 229 个任务，其中 89 个构成了 Terminal-Bench 2.0。这些任务的耗时跨度很大——从 1 小时到超过一周，取决于你是专家还是新手。

![Terminal-Bench 任务的人类完成时间分布](/lectures/12/terminal-bench-human-time.png)

![Terminal-Bench 的排行榜](/lectures/12/terminal-bench-results.png)

看看榜单，最上面的自然是那些前沿模型。但有个值得注意的现象：**agent 本身在这里也很关键**——你可以用同一个模型配两个不同的 agent，它们的分数可以不一样。这一点下面会展开。

## CyBench：夺旗（CTF）挑战

**CyBench** 是网络安全方向的基准，包含 **40 个夺旗（Capture the Flag，CTF）任务**。设定是这样的：给 agent 一个环境，它可以运行各种命令（比如查看源代码）、访问某个 Web 服务器，目标是**攻入这台服务器、取出一个 flag**——flag 是一串唯一的字符串，用来证明你真的攻进去了。

![CyBench 的任务示例](/lectures/12/cybench.png)

这些本来是给人类做的网络安全竞赛练习。下面是一个**极简版 agent scaffold** 的示意——只有一个连续的上下文缓冲：

![CyBench 使用的简单 agent scaffold](/lectures/12/cybench-agent.png)

它的大致逻辑是：模型产生某个动作 → 得到环境反馈 → 把反馈拼进一个历史缓冲 → 继续。用伪代码表示：

```python
def run_agent(model, environment, max_steps: int) -> str:
    """极简 scaffold：一个连续历史 + 循环执行，直到拿到 flag。"""
    history = []
    for _ in range(max_steps):
        action = model.act("\n".join(history))   # 基于全部历史生成下一个动作
        observation = environment.execute(action) # 执行命令、返回环境反馈
        history.append(f">>> {action}\n{observation}")
        if "flag{" in observation:                # 拿到 flag，成功
            return observation
    return "failed"
```

讲者特意指出：这个简单版本在后面会成为一个反例——**随着任务推进，历史缓冲会迅速膨胀**，你很快就会需要更好的上下文管理方式。

![CyBench 的排行榜](/lectures/12/cybench-results.png)

CyBench 刚发布时的排行榜上，最好的模型也只有 **10% 左右**；而现在它**已经完全被解决了**。

## MLE-Bench：Kaggle 竞赛

**MLE-Bench** 把评估搬到机器学习本身：它包含 **75 场 Kaggle 竞赛**——涉及处理数据、训练模型等等。agent 要自己看数据集、读竞赛描述、写代码、训练模型，最后**提交结果、拿到评分**。

![MLE-Bench 的任务示例](/lectures/12/mlebench.png)

![MLE-Bench 的排行榜](/lectures/12/mlebench-results.png)

看现在的排行榜：模型那一侧还是那几个老面孔，但**不同 agent 之间的分数差异相当大**——又一次印证了 scaffold 的重要性。

## Agent scaffold 为什么这么重要

讲者强调，**scaffold 在很大程度上决定了成败**，这也是为什么"语言模型评估"这个话题比"语言模型本身"更宽。与前面那个简单的 SWE-bench/CTF scaffold 不同，解决真正复杂的任务现在需要**成熟得多的脚手架**。人们逐渐意识到这样几件事：

- **显式规划（explicit planning）很有用**：不能只是"意识流式"地链式思考、把上下文越堆越长——agent 很容易丢失自己走到哪了。更稳的做法是**维护一个待办清单（to-do list），每完成一项就划掉一项**；
- **层级委派（hierarchical delegation）**：agent 应该去调用子 agent，并只给它们**干净、裁剪过的上下文**；子 agent 干完活只把**结果**返回给主 agent，主 agent 不需要看到所有的"细枝末节"——这提供了一种很好的封装（encapsulation）；
- **持久记忆（persistent memory）**：尤其在上下文越来越长时，人们在实验**显式地读写文件**——你不能把所有东西都塞在上下文窗口里；
- **更极端的上下文工程（context engineering）**：管理整个过程的风格——什么时候该委派给子 agent、什么时候该换一种策略、该往持久记忆里写什么——而这些通常都是**针对特定语言模型调优**的。

## 小结

智能体之所以令人兴奋，是因为它们**极大地扩展了语言模型的能力面（capability surface）**。而 agent scaffold 极其重要——所以**评估智能体，实际上是在同时评估 agent scaffold 和语言模型两者**。这也意味着，同一份分数背后，你很难分清哪些来自模型、哪些来自脚手架。

<!-- lecture-nav -->

**← 上一节**：[聊天类基准：人类偏好、LLM 评委与 Arena]（03-chat-benchmarks.md）　**→ 下一节**：[纯推理基准与安全评估：ARC-AGI、HarmBench 与越狱](05-reasoning-and-safety-benchmarks.md)
