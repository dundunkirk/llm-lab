# LLM 具体学习计划（12 周）

> 开始日期：2026-04-03
> 策略：基础只补到够用 → 尽快进入源码 → RL 只学大模型真正常用的那一支

**每周执行标准**：不追求"看完多少资料"，而是每周留下 **一张图 + 一段可运行代码 + 一页总结**。

---

## Week 1：反向传播与训练最小闭环

**目标**：把"神经网络是怎么学会的"讲清楚。

**主资料**
- Karpathy：[Lecture 1 micrograd](https://github.com/karpathy/nn-zero-to-hero)
- 按需补：d2l.ai PyTorch 基础、modern_ai_for_beginners 数学基础

**任务**
- [ ] 安装环境：Python、PyTorch、Jupyter
- [ ] 理解计算图、链式法则、梯度
- [ ] 手写一个最小 autograd / backprop demo
- [ ] 跑一个极小训练例子（哪怕只有几个参数）

**本周输出**
- 一个最小 autograd 代码
- 一张"前向 → loss → 反向 → 更新"流程图
- 一页笔记：为什么梯度能指导更新

---

## Week 2：最小语言模型

**目标**：第一次真正理解"语言模型在学什么"。

**主资料**
- Karpathy：Lecture 2 `makemore`（bigram / char-level LM）

**任务**
- [ ] 手写 bigram / char-level 语言模型
- [ ] 跑通训练与采样
- [ ] 理解 negative log likelihood / cross entropy
- [ ] 记录训练前后输出差异

**本周输出**
- 一个最小语言模型
- 一份采样结果记录
- 一页总结：next-token prediction 到底是什么

---

## Week 3：训练稳定性与深层网络直觉

**目标**：理解"能训练"和"训练得稳"不是一回事。

**主资料**
- Karpathy：Lecture 3–5（MLP、BatchNorm、WaveNet）

**任务**
- [ ] 学会 train/dev/test split
- [ ] 掌握基本调参：学习率、batch size、hidden size
- [ ] 理解 underfit / overfit
- [ ] 理解激活饱和、梯度消失、BatchNorm 的作用

**本周输出**
- 一张 loss 曲线 + 调参实验记录
- 一页总结：深层网络为什么容易不稳

---

## Week 4：Transformer、GPT、Tokenizer 主干

**目标**：把大模型的基本骨架真正搭起来。

**主资料**
- Karpathy：Lecture 7–8（nanoGPT + BPE tokenizer）
- 论文：*Attention Is All You Need*（遇到不懂直接问大模型）
- CS336：tokenization / architectures 前几讲

**任务**
- [ ] 理解 self-attention、residual、MLP、layer norm
- [ ] 理解 GPT 的 autoregressive 训练方式
- [ ] 手写 BPE tokenizer，理解 encode / decode 流程
- [ ] 跑通 nanoGPT，理解每个模块

**本周输出**
- 一张 Transformer block 图
- 一张 tokenizer 流程图
- 一页总结：LLM 最小 pipeline 是什么

---

## Week 5：大模型训练输入输出关系

**目标**：把"训练数据如何变成 loss"彻底讲清楚。

**主资料**
- CS336：A1 对应内容
- modern_ai_for_beginners：Transformers based LLMs 分支

**任务**
- [ ] 理解 GPT vs BERT 训练差异（Decoder-only vs Encoder-only）
- [ ] 理解 `input_ids`、`labels`、shift one token 的关系
- [ ] 理解 vocab size 为什么对应分类空间大小
- [ ] 理解预训练数据 vs SFT 数据的差异（连续文本 vs QA 对）
- [ ] 画出"文本 → token → input_ids/labels → logits → cross entropy"完整流程

**本周输出**
- 一张训练输入输出流程图
- 一页总结：LLM 训练为什么本质上是分类问题

---

## Week 6：源码周——以 Qwen2 为锚点

**目标**：从概念切到代码，建立"读 forward 流程"的能力。

**主资料**
- HuggingFace transformers 源码：[Qwen2](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen2)
- CS336：A1 / architectures

**任务**
- [ ] 阅读 `modeling_qwen2.py`，追踪主干：输入 → embedding → attention → hidden states → logits → loss
- [ ] 阅读 `tokenization_qwen2.py`，理解 chat_template 格式与作用
- [ ] 记录 RoPE、KV Cache、GQA、RMSNorm 在哪里起作用
- [ ] 用 `transformers` 加载 Qwen2-0.5B，跑通推理（ModelScope 下载免梯子）

**本周输出**
- 一份源码阅读笔记
- 一张 Qwen2 forward 数据流图

---

## Week 7：SFT / LoRA 最小闭环

**目标**：跑通从数据到损失的监督微调全流程。

**主资料**
- `transformers` + `peft` 库
- CS336：A5 SFT 视角
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

**任务**
- [ ] 理解 `encode / padding / truncate / apply_chat_template`
- [ ] 手写最小 LoRA 微调脚本：数据处理 → 注入 LoRA adapter → Trainer → 保存
- [ ] 理解 label mask（只对 response 部分计算 loss）
- [ ] 用 Alpaca-zh 数据集微调 Qwen2-0.5B，对比微调前后输出
- [ ] 用 LLaMA-Factory 复现同样实验，与手写版对比

**本周输出**
- 一个最小 SFT 脚本
- 一页总结：SFT 的数据流和 loss 怎么接起来

---

## Week 8：Profiling、资源账本、并行基础

**目标**：第一次从系统角度看大模型，不再只盯着 loss。

**主资料**
- CS336：A2（GPU / kernels / profiling / parallelism）
- modern_ai_for_beginners：pytorch distributed 分支

**任务**
- [ ] 对模型做一次简单 profiling
- [ ] 统计：参数量、激活内存、optimizer state、KV cache 各占多少
- [ ] 理解 compute-bound vs memory-bound
- [ ] 理解 data parallel / tensor parallel / pipeline parallel 的区别

**本周输出**
- 一张资源账本表（参数 / 激活 / KV cache 估算）
- 一张并行方式对比图
- 一页总结：为什么多卡不一定线性提速

---

## Week 9：推理系统视角

**目标**：理解模型在 serving 时到底怎么工作。

**主资料**
- CS336：Inference 相关讲次

**任务**
- [ ] 理解 prefill 与 decode 两个阶段的区别
- [ ] 理解 KV cache 的作用与内存占用
- [ ] 理解 batching、latency、throughput 之间的 trade-off
- [ ] 了解 pass@k、PPL 等推理侧评估概念
- [ ] 画一张推理生命周期图

**本周输出**
- 一张 inference 流程图（prefill → decode → 输出）
- 一页总结：推理系统和训练系统关注点为什么不同

---

## Week 10：RL for LLM 总图

**目标**：先把后训练闭环画出来，再进算法细节。

**主资料**
- CS336：A5 / Alignment and Reasoning RL
- modern_ai_for_beginners：RL4LLM 分支
- [OpenAI Spinning Up - RL Intro](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)（只看概念部分）

**任务**
- [ ] 建立 state / action / reward 在 LLM 场景中的映射关系
- [ ] 画出：SFT → preference/reward → policy update 完整闭环
- [ ] 理解 RL 在 LLM 里是"后训练优化"，而非"从零学控制"
- [ ] 了解 RLHF 的基本流程，知道 reward model 的作用
- [ ] 不展开传统 value-based / model-based RL 全家桶

**本周输出**
- 一张 post-training 流程图
- 一页总结：为什么 LLM RL 和传统 RL 入门路径不一样

---

## Week 11：策略梯度主线——PPO / GRPO / REINFORCE

**目标**：只抓 LLM 里最常用的一条 RL 主线。

**主资料**
- modern_ai_for_beginners：RL4LLM
- CS336：Alignment / RL 相关讲次
- veRL 源码：[core_algos.py](https://github.com/volcengine/verl)
- 数学深究（选修）：[赵世钰《强化学习的数学原理》](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)

**任务**
- [ ] 理解策略梯度的直觉：为什么"奖励高的动作应该被加强"
- [ ] 理解 PPO 的目标函数和 clip 机制
- [ ] 理解 GRPO 的定位（组相对策略优化，DeepSeek 使用）
- [ ] 了解 TRPO 和 PPO 的关系（TRPO 是理论，PPO 是工程实现）
- [ ] 阅读 veRL 的 `core_algos.py`，对照算法理解实现

**本周输出**
- 一张 PPO / GRPO / REINFORCE 对比表
- 一页总结：为什么 LLM RL 更常讨论策略梯度而非价值函数

---

## Week 12：收束为一个最小项目

**目标**：把 11 周内容收成一条真正可复用的主线。

选一个方向即可：

**方向 A：偏训练**
- [ ] 做一个完整 SFT / LoRA 项目（数据 → template → labels → loss → 保存 → 推理）
- [ ] 写清楚每一步在做什么

**方向 B：偏系统**
- [ ] 做一个最小推理 demo，加上 profiling / KV cache 分析 / batching 观察
- [ ] 量化不同配置下的 latency 和 throughput

**方向 C：偏 RL**
- [ ] 做一个 toy post-training 示例（reward 打分 → rollout → policy update）
- [ ] 不求大，只求主流程跑通

**本周输出**
- 一个小仓库或项目目录 + README
- 一份 12 周总复盘：已打通哪条主线，还有哪几个断点

---

## 每周复盘模板

```
第 X 周复盘

1. 本周学了什么
2. 当前最卡的 3 个点
3. 已经看过的代码 / 公式 / 概念
4. 最想搞清楚的问题
5. 当前自己的理解（哪怕不完整）
```

---

## 12 周结束后的理想状态

| 层次 | 目标 |
|------|------|
| 模型层 | 能解释 Transformer、tokenizer、NTP、cross entropy、label shift |
| 源码层 | 能看懂主流 decoder-only LLM 的 forward，追踪 hidden states → logits → loss |
| 训练层 | 能独立跑一个 SFT / LoRA，理解 chat_template、padding、label mask |
| 系统层 | 能说清 prefill/decode、KV cache、batching、并行基础 |
| RL 层 | 能讲清 PPO / GRPO / REINFORCE 在后训练中的位置 |

---

## 里程碑检查点

| 时间 | 里程碑 |
|------|--------|
| 第 1 周末 | 手写 autograd，跑通最小训练闭环 |
| 第 4 周末 | 跑通 nanoGPT，理解 Transformer 每个模块 |
| 第 6 周末 | 能读懂 Qwen2 源码 forward 主干 |
| 第 7 周末 | 独立完成一次 LoRA 微调实验 |
| 第 9 周末 | 能画出推理生命周期图，理解 KV cache |
| 第 11 周末 | 能对比 PPO/GRPO，读懂 veRL core_algos.py |
| 第 12 周末 | 完成一个选定方向的完整小项目 |

---

## 执行原则

1. **每周只保留 1 个主目标**，不贪多
2. **每周必须有 1 个可运行结果**，哪怕只有几行代码
3. **看不懂时优先**：自己画图 → 问大模型 → 回看资料
4. **缺什么补什么**，不把支线展开成完整主线
5. **坚持最重要**：看不懂先放一放，之后回来往往豁然开朗
