> 论文：2025.04.11_KIMI-VL TECHNICAL REPORT
> 论文地址：[2504.07491v1](https://arxiv.org/pdf/2504.07491v1)
> 官方代码：[MoonshotAI/Kimi-VL: Kimi-VL: Mixture-of-Experts Vision-Language Model for Multimodal Reasoning, Long-Context Understanding, and Strong Agent Capabilities](https://github.com/MoonshotAI/Kimi-VL)
> 立即体验:https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking

# 背景
随着AI技术进步，人类对AI助手的交互需求从单一语言转向多模态（文本、图像、视频等），以匹配现实世界的复杂性。商业模型（如GPT-4o、Gemini）已实现视觉与语言的无缝融合，但开源社区进展滞后。而开源VLM的瓶颈有：
* **架构局限**：主流开源VLMs（如Qwen2.5-VL）仍依赖密集架构，未能采用更高效的混合专家（**MoE**）结构，且缺乏长链式思维推理（**Long-CoT**）能力。
* **性能短板**：早期MoE-VLM尝试（如DeepSeek-VL2、Aria）存在视觉编码器**适应性差**（*我的理解是不好迁移*）、上下文**长度受限**（仅4K）（*参考信息范围有限*）、**细粒度**任务（如OCR）表现不足（*不够精确*）等问题。

# 主要方法
模型采用 **MoE** 语言模型、原始分辨率视觉编码器 （**MoonViT**） 和 **MLP** 投影器。

![](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202505051905337.png)

**MoonViT**: 一个原生分辨率**视觉编码器**，直接处理各种分辨率的图像，无需复杂的图像分割拼接。通过将图像分割成patch，展平并连接成1D序列，结合可学习的**固定大小绝对位置嵌入**和**2D旋转位置嵌入（RoPE）**，使得模型能够高效处理**不同分辨率**的图像。

**MLP投影器**: 使用一个**两层MLP**来连接视觉编码器和LLM，首先使用**pixel shuffle**操作**压缩**图像特征的空间维度，然后通过MLP将特征投影到LLM嵌入的维度。

**MoE语言模型**: 使用**Moonlight模型**，一个具有**2.8B激活参数（总参数16B）** 的MoE语言模型。

# 训练过程

Kimi-VL的预训练分为四个阶段，共消耗4.4T tokens：

**ViT Training**: 训练MoonViT，目标包括**SigLIP损失L_siglip**（对比损失变体）和**caption生成交叉熵损失L_caption**。总损失函数为**L = L_siglip + λL_caption**，其中λ = 2。MoonViT首先使用**2T** tokens进行CoCa-alike训练，之后使用**0.1T** tokens与MoE语言模型对齐。

**Joint Pre-training**: 使用纯文本数据和多模态数据（图像alt文本、合成字幕、grounding bboxes、OCR文本等）联合训练模型，总共消耗**1.4T** tokens，逐步增加多模态数据的比例。

**Joint Cooldown**: 使用高质量的语言和多模态数据集继续训练模型，加入合成数据，以提升数学推理、知识和代码生成能力。

**Joint Long-context Activation**: 将模型的上下文长度从8K扩展到128K，RoPE嵌入的逆频率从50,000重置为800,000。长数据占总数据的25%。
# Muon优化器

使用增强的[Muon优化器](https://zhuanlan.zhihu.com/p/30895340275)，加入了权重衰减，并调整了参数更新的scale。实现了ZeRO-1优化策略的分布式Muon，以提高内存效率并减少通信开销。

# Post-Training 阶段

包括在32K和128K上下文长度下的联合SFT，以及进一步的long-CoT SFT和RL阶段，以激活和增强长期思维能力。

**Joint Supervised Fine-tuning**: 使用**指令微调**来增强模型遵循指令和进行对话的能力，模型使用**ChatML**格式，优化了语言模型、MLP投影器和视觉编码器，使用纯文本和视觉语言SFT数据的混合，监督仅应用于答案和特殊tokens，系统和用户提示被mask掉。

**Long-CoT Supervised Fine-Tuning**: 采用**prompt工程**构建**高质量long-CoT热身数据集**，包含用于文本和图像输入的准确验证的推理路径，进行**轻量级SFT**，以使模型内化多模态推理策略。

**Reinforcement Learning**: 使用**强化学习**训练模型，使用[在线策略镜像下降](https://zhuanlan.zhihu.com/p/685632557)作为**RL算法**，最大化目标函数：

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/pklYfLiczQuHsAPxk9AyKib3UPTMBNicVlcRhnaZt7DyUWOBItLTVq8tlPOhMS3wWZWTpeC2vzIZt9v25hiccwzY7w/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1)

- r：[奖励模型](https://zhuanlan.zhihu.com/p/20157090301#:~:text=%E5%9C%A8%E5%A4%A7%E6%A8%A1%E5%9E%8B%E7%9A%84%E8%AE%AD%E7%BB%83%E4%B8%AD%EF%BC%8C%E5%B8%B8%E9%87%87%E7%94%A8%20RLHF%20%EF%BC%88Reinforcement%20Learning,from%20Human%20Feedback%EF%BC%89%E5%8D%B3%E5%9F%BA%E4%BA%8E%E4%BA%BA%E7%B1%BB%E5%8F%8D%E9%A6%88%E7%9A%84%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%EF%BC%8C%E6%98%AF%E4%B8%80%E7%A7%8D%E5%B0%86%E4%BA%BA%E7%B1%BB%E5%8F%8D%E9%A6%88%E8%9E%8D%E5%85%A5%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E8%BF%87%E7%A8%8B%E7%9A%84%E6%8A%80%E6%9C%AF%E6%96%B9%E6%B3%95%EF%BC%8C%E8%AF%A5%E6%8A%80%E6%9C%AF%E7%9A%84%E6%9C%80%E9%87%8D%E8%A6%81%E7%9A%84%E4%B8%80%E9%83%A8%E4%BB%BD%E5%B0%B1%E6%98%AF%E5%A5%96%E5%8A%B1%E6%A8%A1%E5%9E%8B%E3%80%82%20%E5%A5%96%E5%8A%B1%E6%A8%A1%E5%9E%8B%E4%B9%9F%E5%8F%AB%20%E6%89%93%E5%88%86%E6%A8%A1%E5%9E%8B%EF%BC%8C%E6%98%AF%E4%B8%80%E7%A7%8D%E9%80%9A%E8%BF%87%E9%87%8F%E5%8C%96%E6%96%B9%E5%BC%8F%E5%AF%B9%E6%A8%A1%E5%9E%8B%E8%BE%93%E5%87%BA%E7%BB%93%E6%9E%9C%E8%BF%9B%E8%A1%8C%E8%B4%A8%E9%87%8F%E8%AF%84%E4%BC%B0%E5%B9%B6%E6%89%93%E5%88%86%EF%BC%8C%E4%BB%A5%E5%BC%95%E6%A8%A1%E5%9E%8B%E5%AD%A6%E4%B9%A0%E4%BC%98%E5%8C%96%E6%88%96%E8%BE%93%E5%87%BA%E7%BB%99%E7%94%A8%E6%88%B7%E7%BB%93%E6%9E%9C%E5%89%8D%E5%81%9A%E8%B4%A8%E9%87%8F%E8%AF%84%E4%BC%B0%EF%BC%8C%E5%88%A4%E6%96%AD%E6%98%AF%E5%90%A6%E9%9C%80%E8%A6%81%E9%87%8D%E6%96%B0%E7%BB%99%E7%94%A8%E6%88%B7%E9%A2%84%E6%B5%8B%E3%80%82)
- τ > 0：控制正则化程度的参数

使用[**长度惩罚**](https://zhuanlan.zhihu.com/p/716596412)来防止生成过长的回复，以及使用课程采样和优先采样策略。

# 实验结果

Kimi-VL在多个benchmark上表现出色，尤其在长文本、长视频和高分辨率图像处理方面。Kimi-VL-Thinking通过长CoT激活和强化学习，在MMMU、MathVision、MathVista等推理benchmark上超越了许多更大规模的VLM。

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202505052019636.png)

# 结论

Kimi-VL是一个高效、多功能的VLM，具有强大的多模态推理、长上下文理解和代理能力。Kimi-VL-Thinking进一步提升了模型的推理能力，在复杂的图像和视频推理任务中表现出色。





