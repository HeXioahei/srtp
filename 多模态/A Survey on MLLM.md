博客推荐：[A Survey on Multimodal Large Language Models-全文解读 - 知乎](https://zhuanlan.zhihu.com/p/641866192)
[【综述论文阅读】A Survey on Multimodal Large Language Models 上_prompt-based ensemble expert language models with -CSDN博客](https://blog.csdn.net/weixin_46231495/article/details/145903655?ops_request_misc=&request_id=&biz_id=102&utm_term=a%20survey%20on%20multimodal%20large%20l&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-145903655.142^v102^pc_search_result_base9&spm=1018.2226.3001.4187)


# 架构
典型架构抽象为三个模块，即预训练的**多模态编码器、预训练的大语言模型**以及连接它们的**多模态接口**。



本文将最近的代表性 MLLM 分为四种主要类型：多模态指令调整 (MIT)、多模态上下文学习 (M-ICL)、多模态思维链 (M-CoT) 和 [LLM 辅助视觉推理](https://zhida.zhihu.com/search?content_id=230768050&content_type=Article&match_order=1&q=LLM+%E8%BE%85%E5%8A%A9%E8%A7%86%E8%A7%89%E6%8E%A8%E7%90%86&zhida_source=entity) (LAVR)。

# MIT（Multimodal Instruction Tuning）

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202503112035847.png)

Instruction Tuning（指令调优）是一种涉及在指令格式数据集集合上微调预先训练的llm的技术。通过这种方式进行调整，LLM 可以通过遵循新指令泛化到看不见的任务，从而提高零样本性能。Instruction Tuning学习如何泛化到未知的任务，而不是像一对一的拟合特定任务。

M-IT中校准预训练的一种常见方法是保持预训练模块(例如visual encoder和llm)冻结，仅训练一个可学习的接口。

现有VQA和标题数据集的答案通常很简洁，直接使用这些数据集进行指令调优可能会限制MLLM的输出长度。解决这个问题有两种常见的策略：第一个是修改instruction，直接告诉LLM我们要简短、单句。第二种方法是延长现有答案的长度。

Self-Instruction：尽管现有的基准数据集可以提供丰富的数据源，但它们通常不能很好地满足现实场景中的人类需求，例如多轮对话。为了解决这个问题，一些作品通过自我指导来收集样本[60]，这引导llm使用一些手工注释的样本来生成遵循文本指令的数据。

混合指令调优不比多模态数据上单独调优差。

Learnable Interface：当冻结预训练模型的参数时，可学习接口负责连接不同的模态。挑战在于如何有效地将视觉内容翻译成LLM可以理解的文本。一种常见且可行的解决方案是利用一组可学习的query token以基于查询的方式提取信息。此外，一些方法使用基于projection head的桥接方法来缩小模态差距。

Expert Model：借助专家模型来简化视觉信息更简单，但是灵活性很差，而且会存在信息丢失的情况。

# MICL（Multimodal In-Context Learning）
