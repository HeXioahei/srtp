> 相关博客：[MVoT：首个多模态思维可视化，多模态中的CoT](https://mp.weixin.qq.com/s/r0Bn2EUwefIe_T2aTPGhgA)
> 论文：[[2501.07542] Imagine while Reasoning in Space: Multimodal Visualization-of-Thought](https://arxiv.org/abs/2501.07542)


# 摘要

思维链（CoT）提示在复杂的**空间推理**任务中表现不佳。

人类认知不仅限于语言，还能够用文字和图像进行思考。

受此启发，本文提出了一种新的推理范式——**多模态思维可视化（MVoT）**，它通过**生成推理轨迹的图像可视化**，使 MLLMs 能够进行视觉思维。

为确保高质量的可视化，本文在自回归 MLLMs 中引入了**标记差异损失**，这显著提高了视觉连贯性和保真度。

# 背景

近期研究主要通过两种途径将 CoT 扩展到多模态模型：
* 采用两阶段策略，在推理前先通过**图像描述、场景图生成**或**边界框检测**等方法提取图像信息；
* 利用 ReAct 风格的管道，借助**外部工具**（如代码解释器或专用视觉模型）从环境中获取图像观察结果。

但这些方法依赖于单独的视觉模块或外部工具集，在适应高级复杂的空间推理任务时存在困难。
依赖纯文本表示推理路径，难以捕捉图像的**复杂视觉模式和空间布局**，且用户难以解释缺乏直观视觉说明的推理过程。
与此同时，**多模态生成能力的出现**为将语言推理扩展到**原生视觉思维**提供了新的可能性。

# 贡献

1. 提出了多模态思维可视化（MVoT），这是一种多模态原生推理范式，在推理轨迹中统一了文本和视觉。它是首个在推理过程中**自然生成视觉思维**的方法。它为**视觉思维能有效补充语言推理的复杂任务**开辟了新的可能性。
    
2. 在 Chameleon-7B 中实现了 MVoT，并在自回归MLLMs中引入了**标记差异损失**，以**弥合单独训练的标记器之间的差距**。
    
3. 利用新收集的数据集在三个空间推理任务中进行了全面的实验和消融研究，结果表明在复杂场景中，MVoT 相较于CoT展现出了**更优越的适应性和鲁棒性**。

# 技术方案

## 推理过程公式化

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202503141015348.png)

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202503141022607.png)

## 基于自回归 MLLM 训练

**多模态序列建模**：遵循 Chameleon 架构，利用统一的 Transformer 处理图像和文本标记。

该架构集成了基于 ERO21 的图像标记器和文本标记器，分别将图像和文本转换为离散标记序列，然后连接并由因果 Transformer 模型处理。

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202503141024900.png)

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202503141025294.png)

# 实验结果

MVoT 在可解释性方面优于直接提示和 GPT-4o。

在所有三个模拟任务中，直接提示在空间推理任务中存在过拟合问题，准确率约为 70%，GPT-4o 的表现更差。

相比之下，MVoT 有持续的改进，在 FROZENLAKE 上比直接提示高出 7%，在 MAZE 和 MINIBEHAVIOR 上的准确率超过 90%

并且提供了中间推理状态的语言和视觉思维，增强了可解释性。

MVoT 比 CoT 具有更好的鲁棒性和相当或更好的性能。

CoT 在 MAZE 和 MINIBEHAVIOR 上通过用文本坐标明确描述环境布局和代理位置，准确率超过 95%，但在 FROZENLAKE 上比直接提示基线表现更差。

MVoT 在 MAZE（92.95%）和 MINIBEHAVIOR（95.14%）上表现相当，在 FROZENLAKE 上的准确率（85.60%）高于直接提示和 CoT，表明其具有更好的稳定性和鲁棒性。

数据表明，纳入交错训练数据（即使不生成可视化）可提高推理性能，MVoT 通过在训练中纳入交错的多模态原理作为视觉标记的监督信号，在所有任务中都有更高且更一致的改进。

在 FROZENLAKE 任务中，与未使用标记差异损失的模型相比，使用标记差异损失的 MVoT 生成的可视化具有更高的准确性和更少的冗余，提高了可视化质量并有助于提升任务性能。

此外，将 MVoT 作为插件应用于其他专有模型（如 GPT-4o）可提高其在所有任务中的性能，准确率提高超过 15%。

# 结论

本文介绍了多模态思维可视化（MVoT），这是一种利用多模态原生生成模型通过多模态思维引出推理过程的新颖推理框架。

MVoT 在各种任务中优于文本推理基线，对状态复杂性具有更好的鲁棒性，并提供了增强的可解释性。

为确保生成高质量的可视化，本文提出了**标记差异损失**，解决了自回归 MLLM 中的嵌入不匹配问题，有助于减少冗余模式和不准确的视觉思维生成问题，从而提高 MVoT 的任务性能。

此外，MVoT 和思维链（CoT）的互补优势突出了混合多模态推理方法的前景，强调了整合多模态线索的价值，并为未来研究推进复杂任务的混合模态推理思想铺平了道路。