博客推荐：[A Survey on Multimodal Large Language Models-全文解读 - 知乎](https://zhuanlan.zhihu.com/p/641866192)

本文将最近的代表性 MLLM 分为四种主要类型：多模态指令调整 (MIT)、多模态上下文学习 (M-ICL)、多模态思维链 (M-CoT) 和 [LLM 辅助视觉推理](https://zhida.zhihu.com/search?content_id=230768050&content_type=Article&match_order=1&q=LLM+%E8%BE%85%E5%8A%A9%E8%A7%86%E8%A7%89%E6%8E%A8%E7%90%86&zhida_source=entity) (LAVR)。

# MIT

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202503112035847.png)

Instruction Tuning（指令调优）是一种涉及在指令格式数据集集合上微调预先训练的llm的技术。通过这种方式进行调整，LLM 可以通过遵循新指令泛化到看不见的任务，从而提高零样本性能。Instruction Tuning学习如何泛化到未知的任务，而不是像一对一的拟合特定任务。

M-IT中校准预训练的一种常见方法是保持预训练模块(例如visual encoder和llm)冻结，仅训练一个可学习的接口。