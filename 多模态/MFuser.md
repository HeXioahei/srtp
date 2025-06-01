> github地址：[devinxzhang/MFuser](https://github.com/devinxzhang/MFuser)
> 论文地址：[2504.03193](https://arxiv.org/pdf/2504.03193)
> 推文：[【CVPR2025】Mamba 作为桥梁：连接视觉基础模型与视觉语言模型以实现领域泛化语义分割](https://mp.weixin.qq.com/s/hfXt3fA-D2plcbs9xudNEg)

# 简介
**VFMs**（如 DINOv2）在捕捉**细粒度**特征方面表现优异，而 **VLMs**（如 CLIP）在**文本对齐**方面具有强大的**鲁棒性**，但在处理细粒度信息时则存在一定困难。尽管它们在能力上互为补充，但**利用注意力机制将 VFMs 与 VLMs 有效融合仍具有挑战性**，因为更大量的 patch token 会加剧长序列建模的复杂性。

为了解决这一问题，我们提出了 **MFuser**——一种基于 **Mamba** 的新型融合框架，能够高效整合 VFMs 与 VLMs 的优势，并在序列长度上保持**线性扩展性**。MFuser 包含两个关键模块：
- **MVFuser**：一个协同适配器（co-adapter），通过捕捉时序与空间动态，实现对两个模型的联合微调；
- **MTEnhancer**：一个融合注意力机制与 Mamba 的模块，通过引入图像先验来优化文本嵌入表示。

# 准备工作
* ##### 域泛化语义分割（Domain Generalized Semantic Segmentation，DGSS）
	由一个视觉编码器E和一个分割掩码器D组成的分割模型M，训练网络**泛化到未知的目标域**。

* ##### 使用文本查询进行语义分割（Semantic Segmentation with Text Queries）
	* 利用基于 query 的机制，其中**可学习的 object query 作为动态指针，将模型的焦点引导到相关区域**。在此基础上，最近的研究越来越多地利用视觉语言模型（VLMs）的图像-文本对齐功能来设计基于文本的查询。
	* vlm产生的文本嵌入具有**固有的领域不变性**，**捕获的语义信息在不同的上下文和视觉风格中保持一致**。这种领域不变性源于VLM训练过程，该过程将文本描述与不同的视觉输入相关联，有效地将语义内容与领域特定的特征分离开来。**文本嵌入的领域不变性是促进视觉特征领域泛化的基础**。
	* 在本文中，我们遵循类似的 pipeline，利用每个类的文本嵌入作为 Mask2Former 解码器中的 queries。形式上，VLM的视觉编码器 $E^{VFM}_{V}$ 作为分割模型的编码器，对齐的文本编码器 $E^{VFM}_{T}$ 生成类嵌入 $\lbrace{class_k}\rbrace ^C_{k=1}$ 的每个类嵌入 $q_t = [t^1, t^2, t^3, ..., t^C]$ 。$q_t$ 将用于设计解码器的查询或条件查询。

# 方法
使用用于DGSS的Mask2Former解码器将**任意VFM与CLIP-like VLM集成**。下图展示了MFuser的总体架构。

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202505312203372.png)


![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202505312202774.png)

* MFuser增强了**特征局部性**，同时利用文本嵌入提供的**领域不变语义知识**来有效地**约束视觉表示**。该框架的核心组件包括MVFuser和MTEnhancer。冻结VLM和VFM的参数，只有MVFuser和MTEnhancer的参数来微调模型。
	* **MVFuser**：以参数有效的方式共同微调两个模型的视觉编码器，融合它们的特征以最大限度地发挥协同作用。（我的思考：融合就体现在最后的相乘操作，逐元素相乘完之后再分开继续在各自的模型中进行进一步的特征提取。而为什么要将MVFuser的结果再加上进入MVFuser前的特征，因为这就是残差连接，这样可以有效地避免模型的学习偏离正确方向。）
	* **MTEnhancer**：通过整合视觉特征来丰富文本查询，增强语义对齐和特征鲁棒性。（我的思考：将文本信息与图像特征进行融合，丰富文本查询。）

## 细节
### 1. MVFuser

**优点**：
1. 保留了两个视觉编码器的原始特性和优势，同时减轻了它们的弱点。
2. 改进两个视觉编码器以生成更多特定于任务的特性。
3. 实现两个视觉编码器功能的有效交互。

**如何有效交互**：
* 问题：捕获令牌间关系的一个自然想法是使用 self-attention 机制。然而，拼接来自两个编码器的特征使得序列长度增加了一倍，随着令牌数的增加，计算复杂度呈二次增长，所以在 transformer 中应用 attention 机制进行自适应是低效的。虽然引入可学习 token 并在可学习 token 和 patch token 特征之间应用 cross-attention 可以减少这种计算成本，但它很难有效地捕获 token 之间的依赖关系。
* 解决方案：使用经过略微改进的 [manba](manba) 结构。



