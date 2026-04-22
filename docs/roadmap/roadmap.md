# LLM 学习路线图

> 原理与实践互为表里，不可分割。既要懂代码实现，也要懂数学计算原理。
>
> 策略：基础只补到够用 → 尽快进入源码 → RL 只学大模型真正常用的那一支

---

## PyTorch 与反向传播

- 理解计算图、链式法则、梯度更新
- 动手：手写最小 autograd（参考 Karpathy micrograd）
- 参考：[经典神经网络模型拓扑结构（PyTorch）](https://space.bilibili.com/59807853/channel/collectiondetail?sid=446911)

## 最小语言模型直觉

- 从 bigram / char-level LM 开始，理解 next-token prediction
- 理解 negative log likelihood / cross entropy
- 动手：Karpathy [makemore](https://github.com/karpathy/nn-zero-to-hero) Lecture 2

## 数学基础（按需补）

- 多元微积分（Jacobian、链式法则）
- 矩阵分析（矩阵乘法、SVD）
- 概率统计（先验、似然、后验）
- 参考：[深度学习的数学基础](https://space.bilibili.com/59807853/channel/collectiondetail?sid=462509)

---

## Transformer 架构

- 直接读论文：*Attention Is All You Need*（遇到不懂问大模型）
- 理解 self-attention、residual、MLP、layer norm
- 理解 BPE tokenizer：encode / decode 流程
- 动手：Karpathy Lecture 7–8（nanoGPT + BPE tokenizer）
- 课程：[Stanford CS336](https://cs336.stanford.edu/spring2025/index.html) tokenization / architectures 部分

## 语言模型核心概念

| 概念 | 要点 |
|------|------|
| GPT 训练方式 | NTP（下一个 token 预测），单向，本质是分类任务 |
| BERT 训练方式 | Mask 机制，双向编码，Encoder-only |
| 训练本质 | 类别数 = 词表大小，label = input_ids shift 一位 |
| 预训练 vs SFT | 数据形式不同：连续文本 vs QA 对 + chat_template |
| 分词方式 | BPE（字节对编码）为主流 |

## 大模型架构源码

- 锚点：HuggingFace transformers 中的 [Qwen2](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen2)
- 追踪主干：输入 → embedding → attention → hidden states → logits → loss
- 重点机制：RoPE、GQA、RMSNorm、KV Cache

---

## SFT（有监督微调）

- 动手用 `transformers` + `peft` 手写 LoRA 微调流程：
  - 数据处理：encode、padding、truncate、apply_chat_template
  - label mask：只对 response 部分计算 loss
  - 损失计算：交叉熵
- 工程框架：[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)、[Swift](https://github.com/modelscope/swift)
- 模型下载：[ModelScope](https://modelscope.cn)（国内免梯子）/ [hf-mirror](https://hf-mirror.com)
- 参考：CS336 A5 SFT 视角

## 系统视角（Infra）

- **Profiling**：统计参数量、激活内存、optimizer state、KV cache
- **并行基础**：data parallel / tensor parallel / pipeline parallel 的区别
- **推理系统**：prefill vs decode、KV cache 作用、batching / latency / throughput
- 参考：CS336 A2、modern_ai_for_beginners pytorch distributed

## 强化学习（RL）

- 先建立 post-training 总图：SFT → preference/reward → policy update
- 理解 RL 在 LLM 里是"后训练优化"，而非"从零学控制"
- 主线算法：策略梯度 → PPO（TRPO 的工程实现）→ GRPO → REINFORCE
- 不展开传统 value-based / model-based RL
- 推荐框架：[veRL](https://github.com/volcengine/verl)，重点看 `core_algos.py`
- 数学深究（选修）：[赵世钰《强化学习的数学原理》](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)
- 视频：[李宏毅 DRL](https://www.youtube.com/watch?v=z95ZYgPgXOY&list=PLJV_el3uVTsODxQFgzMzPLa16h6B8kWM_) / [RL4LLM B站](https://space.bilibili.com/59807853/channel/collectiondetail?sid=4048984)

---

## RAG（检索增强生成）

- 核心：分块策略 > embedding 模型选择
- 重点：分块方法、embedding 模型微调、提升召回率的 trick

## Agent

- 核心流程：ReAct（推理 + 行动 + 反馈循环）
- 两个关键问题：上下文如何注入（skills）、历史如何存储（memory）
- 参考实现：[nanobot](https://github.com/rashadphz/nanobot)
- 进阶：Agentic RL（小模型 Agent 效果提升）+ veRL 框架

## Text2SQL

- 上下文工程的典型应用，按需学习

---

## 参考资源汇总

| 类型 | 资源 |
|------|------|
| 动手系列 | [Andrej Karpathy - nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) |
| 课程 | [Stanford CS336 - Language Modeling from Scratch](https://cs336.stanford.edu/spring2025/index.html) |
| 书籍 | [动手学深度学习 d2l.ai](https://zh.d2l.ai/index.html) |
| 书籍 | [Natural Language Processing with Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098103231/) |
| 视频 | [Andrej Karpathy YouTube](https://www.youtube.com/@AndrejKarpathy) |
| 视频 | [李宏毅深度学习 / DRL](https://www.youtube.com/c/HungyiLeeNTU) |
| B站 | [modern_ai_for_beginners](https://space.bilibili.com/59807853) |
| 框架 | [HuggingFace Transformers](https://github.com/huggingface/transformers) |
| 框架 | [veRL](https://github.com/volcengine/verl) |
| 框架 | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) |
| 课程 | [fast.ai](https://course.fast.ai/) |

---

## 学习原则

1. **基础够用即可**：不追求面面俱到，缺什么补什么
2. **尽快进入源码**：概念理解后立刻找到对应代码
3. **每周一个可运行结果**：哪怕只有几行代码
4. **善用大模型**：遇到不懂的，让它结合公式 + 代码解释
5. **坚持**：看不懂先放一放，回过头来往往豁然开朗
