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
* 问题：捕获 token 间关系的一个自然想法是使用 self-attention 机制。然而，拼接来自两个编码器的特征使得序列长度增加了一倍，随着令牌数的增加，计算复杂度呈二次增长，所以在 transformer 中应用 attention 机制进行自适应是低效的。虽然引入可学习 token 并在可学习 token 和 patch token 特征之间应用 cross-attention 可以减少这种计算成本，但它很难有效地捕获 token 之间的依赖关系。
* 解决方案：使用经过略微改进的 [mamba](mamba.md) 结构。

### 2.MTEnhancer
text embeddings 已经被用作语义分割中的 queries，将任务视为代表性 class queries 和 image patch features 之间的匹配问题，或者作为 Mask2Former 解码器的初始 object queries 。该方法利用嵌入在文本中的领域不变语义信息来增强模型准确识别和分割图像中相关区域的能力。通过提出的 MTEnhancer 结合融合的视觉先验来增强来自VLM的原始文本嵌入。

**架构**：
MTEnhancer是一个混合架构，结合了 self-attention block、conditional Mamba block 和 MLP，利用了各种模型架构的优势。
* **self-attention block**：对类间关系进行编码。
* **conditional Mamba block**： 将图像标记集成到文本嵌入中。虽然 mamba 块擅长处理长符号序列，但它在 cross-attention 机制中的应用在很大程度上仍未被探索。为了有效地利用 manba 固有的单向扫描顺序，我们建议在 image token 的两侧连接两个 text embeddings 副本，它们一起作为 manba 块的输入。MTEnhancer中的每个块都是用残差连接实现的。
	![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202506011426341.png)

## 目标函数
使用**预测级分割损失**和**特征级对齐损失**来训练框架。
* 对于分割损失，我们遵循标准的Mask2Former：
	![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202506011619958.png)
	* bce：binary cross-entropy loss，二元交叉熵损失，对于每个预测掩膜的损失
	* dice：Dice相似系数损失函数，对于每个预测掩膜的损失
	* cls：cross-entropy loss， 交叉熵损失，对于每个查询提议。
* 此外，我们使用**像素-文本对齐损失**来强制像素级视觉语言对齐，以确保文本语义精确映射到相应的图像区域。实验涉及三个vlm: CLIP， EVA02-CLIP和SIGLIP。我们对CLIP和EVA02-CLIP使用SoftMax损失，对SIGLIP使用Sigmoid损失，与每个VLM原始训练时使用的损失函数一致。
* 因此，整体训练损失为：
	![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202506011624958.png)

# 实验
## 1.配置
### 1.1.数据集
我们评估了 MFuser 在“合成到真实”、“晴天到恶劣天气”，以及“真实到真实”场景中的性能。
* 作为合成数据集，GTAV 包含用于训练、验证和测试的 12,403、6,382 和 6181 张图像，分辨率为 1914 × 1052。
* 作为真实世界数据集，
	* Cityscapes 包含 2,975 张用于训练的图像和 500 张用于验证的图像，分辨率为 2048 × 1024。
	* BDD100K 包含用于训练和验证的 7,000 和 1,000张图像，每张分辨率为 1280 × 1024。
	* Mapillary 包含 18,000 张训练和 2,000 张验证图像，整个数据集分辨率不一。
* 我们也在补充中加入了从晴天到恶劣天气的泛化。

### 1.2.网络架构
* **VFM**：DINOv2。
* **vlm**：CLIP、EVA02CLIP、SIGLIP。
* **分割解码器**：我们遵循tqdm，它通过用增强的类嵌入替换随机初始化的对象查询来修改标准Mask2Former解码器。因此，文本对象查询被设置为19以匹配类的数量。

### 1.3.实现细节
* 保持VFM和VLM的参数不变，只训练MVFuser、MTEnhancer和分割解码器。
* 对所有VLM备选方案和两种泛化都设置相同的训练配置。
* 为文本编码器应用提示调优。
* 所有实验的输入大小为512×512，批量大小为2，学习率为1e-4。使用AdamW优化器，在t_warm = 1.5k 迭代中进行线性预热，然后进行线性衰减。
* 应用分割任务的标准增强，包括随机缩放、随机裁剪、随机翻转和颜色抖动。
* 所有实验均在一台24GB RTX A5000上进行。

## 2.实验结果
- **合成到真实场景**：MFuser在G→B基准上以EVA02-CLIP为VLM时提升1.49 mIoU，平均超越现有方法2.15 mIoU，最高成绩达68.20 mIoU。
- **真实到真实场景**：在C→B和C→M任务中分别提升0.74和1.7 mIoU，整体提升1.43 mIoU，最高成绩达71.87 mIoU。
- **多模态组合效果**：不同VLM与DINOv2的组合均保持优异性能，显示框架的鲁棒性和适应性。

## 3.消融实验与分析
- **MVFuser模块数量**：增加MVFuser模块数量通常提升性能，验证了多层次特征融合的有效性。
- **注意力机制设计**：MTEnhancer模块通过结合图像先验优化文本嵌入，增强了跨模态一致性，消融实验证实其关键作用。

## 4.定性对比
* **未见目标域处理**：在G→M和G→B等迁移场景中，MFuser相比Rein和tqdm等方法能更准确识别细粒度差异（如车辆、道路标志），误分类更少。
	![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202506011655461.png)
	![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202506011655434.png)
* - **复杂环境适应性**：在雨、雾等恶劣条件下，MFuser展现出更强的特征泛化能力，尤其在雨领域提升显著。

## 5.计算效率
基于Mamba的线性复杂度设计，MFuser在保持高性能的同时未显著增加计算负担，验证了框架的高效性。

# 结论
在这项工作中，我们提出了MFuser，一种新的融合框架，旨在将VFMs和VLMs集成到DGSS中。通过利用vfm和vlm的互补优势，MFuser通过高效、可扩展的线性复杂性融合解决了补丁令牌增加的挑战。该框架包含两个关键组件：MVFuser，它联合微调vfm和vlm以增强特征交互；MTEnhancer，它使用图像先验来改进文本嵌入，以获得更好的对齐和鲁棒性。广泛的实验结果表明，MFuser实现了精确的特征定位和稳健的文本对齐，同时在各种基准测试中优于最先进的DGSS方法。该研究强调了结合VFMs和VLMs在语义分割任务中实现卓越泛化能力的潜力，并强调了MFuser在推进DGSS方面的有效性，通过改进对未见域的泛化而不增加显著的计算开销。

# 个人的一些些思考
本文有两点值得借鉴：
1. MVFuser：将vfm和vlm的功能进行结合，使得粗粒度和细粒度的特征都能得到较为有效的提取。解决了跨模态vlm编码器难以提取细粒度特征的问题。
2. MTEnhancer：将文本查询和视觉特征进行先验交互，得到更高效的文本查询，有利于增强语义，实现高效的图像文本对齐。

缺陷：虽然mamba的使用可以提高计算效率，但是还是不够轻量化。