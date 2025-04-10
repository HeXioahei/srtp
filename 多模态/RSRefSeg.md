> 其他博客推荐：[遥感论文 | Arxiv | RSRefSeg：基于SAM和CLIP的1.2B遥感语义分割基础模型，代码已开源！ - 知乎](https://zhuanlan.zhihu.com/p/25865061675)
> 
> Github地址：[KyanChen/RSRefSeg: This is the pytorch implement of the paper "RSRefSeg: Referring Remote Sensing Image Segmentation with Foundation Models"](https://github.com/KyanChen/RSRefSeg)
> 
> 论文地址：[2501.06809](https://arxiv.org/pdf/2501.06809)

# 摘要

RRSIS（Referring remote sensing image segmentation）的**自由格式文本输入**可以实现**细粒度**视觉理解（==简单来说，就是由自然语言描述来决定要分割图像中的什么对象==），增强场景和对象提取。传统MVLM（如CLIP）通常难以在细粒度语义概念之间建立稳健的对齐，导致文本和视觉信息之间的表示不一致。故提出RSRefSeg（referring remote sensing image
segmentation foundational model）：
* **编码器**：CLIP。
* **过滤器**：全局和局部的文本语义学。
* 在潜在空间中生成**与引用相关的视觉激活特征** -> 用**AtnPrompter**处理 -> 作为**SAM**的输入提示 -> SAM通过其强大的**视觉泛化**功能改进分割掩码。
在RRSIS-D数据集上训练。

# 引言

#### **问题：**
早期的CNN和RNNs通过双峰特征提取来融合基本信息进行分割，而注意力机制能够探索深层语义空间中的模态交互和对齐。但CLIP这种模式难以实现细粒度语义概念对齐和保持文本-视觉信息表达的一致性，尤其在捕获不同地面物体之间的相关性、表示多尺度遥感特征和处理小物体时。

#### **解决：**
本文提出的RSRefSeg将**CLIP的粗粒度文本视觉语义学**和**SAM的精细掩码表示**相结合，来提高**泛化**和**多功能**性。具体流程如摘要。解决了两个挑战：跨域传输期间的性能下降以及来自多个基础模型的一般知识的集成。为了减轻域适应挑战，在CLIP和SAM主干网中引入了**低秩参数效率的微调**。

#### 这项工作的主要贡献：
* 基础模型RSRefSeg，它在参考遥感图像分割任务中展示了卓越的泛化和多功能性。
* 研究了使用CLIP的文本视觉粗粒度对齐语义学的潜力，作为SAM生成精细分割掩码的提示，解决了跨域传输中性能下降的挑战以及跨多个基础模型集成和传输一般知识的困难。
* RSRefSeg在RRSIS-D数据集上优于当代方法，从而突出了基础模型在理解遥感多态模态任务方面的实用价值。

# 方法

### 概括
#### 三个组成部分
* **微调CLIP**：从自由形式的文本引用和低分辨率视觉特征中提取全局和局部语义嵌入。==（图像和文本输入及编码）==
* **AtnPrompter**：处理CLIP的文本和视觉特征，以提取与引用内容相对应的粗略视觉激活特征，随后将它们转换为与SAM兼容的提示嵌入。==（激活特征并生成提示）==
* **微调SAM**：使用这些提示嵌入处理原始图像，以生成相应的二进制分割掩码。==（利用提示对原始图像生成分割掩码）==

#### 公式表示：

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504102056362.png)
#### 模型总览

![d75ea7c993fd48e90e8383518d12b63.jpg](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504102132537.jpg)

H1<H2, W1<W2, 即 I1 为 I2 的下采样。

### CLIP微调

利用SAM的细粒度分割功能来弥补CLIP粗粒度的语义对齐。CLIP主要是为一般场景理解而设计的，当直接应用于遥感领域时，其性能会下降。为了应对这一挑战，该工作通过引入**额外的可训练参数**来引入**低秩微调**，如下式所示，

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504102138228.png)

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504102139318.png)：基本模型中的冻结参数矩阵

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504102139990.png)：新引入的低秩可训练参数矩阵（r << d）

这些低秩可训练参数被合并到图像和文本编码器中。

原始CLIP架构产生稀疏的图像和文本表示以进行预训练或分类，而该工作通过**删除池化层**来保留每个文本标记所对应的原始图像特征图和隐藏状态。

结束标记整合了整个句子的语义信息，其对应的嵌入被指定为**全局语义特征**。其他单词的特征嵌入被归类为**局部语义特征**，代表特定的类别、位置和其他属性。

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504102150850.png)
![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504102150597.png)

完整的特征表示为`t=[tlocal， tglobal]`。

### AttnPrompter

**SAM**缺乏自动化的实体感知能力，本工作将SAM作为一个独立的、由引用语义信息控制的**解耦**（==*将原本紧密关联的组件或功能分离成独立的部分以减少相互依赖*==）**掩码生成器**。为了将CLIP的引用语义信息作为提示集成到SAM中，该工作将AtnPrompter作为二者的桥梁。

AtnPrompter利用**文本语义学**作为**过滤器**来提取**与引用表达式相关的关键视觉特征**，并生成**提示嵌入**（即指示分割目标的点或框）。

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504102208767.png)

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504102213938.png)：表示CLIP提取的视觉特征

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504102213311.png)：计算与全局语义学相关的激活图

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504102214162.png)：平均池化

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504102214320.png)：将三个分量级联

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504102214498.png)：视觉参与特征

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504102220634.png)

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504102224479.png)：稀疏提示。M为提示点数。

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504102224473.png)：表示具有1×1内核大小的卷积，将通道维度从d1减少到d2。

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504102224692.png)：表示多个Conv-BN-GELU卷积块，内核大小为3×3，步幅为2，用于空间维数缩减到M。

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504102225973.png)：密集提示。包括用全局语义学过滤CLIP视觉特征获得的粗粒度响应掩码，然后进行上采样。

### 微调SAM

SAM处理**原始图像I2**和来自提示生成器的**稀疏/密集提示**，并产生最终的**引用分割掩码**。由于该任务只需要单个分割输出，因此从SAM的输出中选择**第一个生成的掩码**作为最终结果。为了解决域传输中的语义分布差异，该工作在SAM的编码器中引入了**参数微调**，遵循如上CLIP微调的方法。

# 实验

### 实验数据集和设置

数据集RRSIS-D：
* 包含**17,402个三元组**（一个图像、一个掩码和一个引用表达式）
* 被分成**12,181个训练样本、1,740个验证样本和3,481个测试样本**
* 包含**20个不同的语义类别**，包括飞机、高尔夫球场、高速公路服务区、棒球场和体育场
* 所有图像都标准化为**800×800像素**
* 空间**分辨率从0.5到30米**不等

### 评估协议和指标

* **联合上的广义交叉点gIoU**：所有图像中单个IoU分数的平均值。
* **联合上的累积交叉点cIoU**：为累积交叉点与累积联合的比率。

**主要依赖gIoU**进行分析，因为cIoU倾向于偏向更大的目标区域，并显示出更大的统计方差。

此外，该工作实施阈值为0.5至0.9的精度指标（Pr@X），以量化符合特定IoU标准的测试图像的百分比。

### 实现的细节

#### 架构细节
* 使用**SigLIP**（CLIP的增强版本，在训练期间使用Sigmoid损失函数）对SAM所需的提示进行编码。具体来说，使用的是**siglip-so400m-patch14-384**版本。
* 提示器通过**两个卷积块**处理来自CLIP的视觉参与特征图以进行空间下采样，然后将它们展平以作为SAM的提示。
* 低秩微调的维度设置为**r=16**。
* 使用SAM的基本版本和大型版本进行了实验，分别指定为**RSRefSeg-b**和**RSRefSeg-l**，消融研究仅限于基本版本。

#### 训练细节
* 采用**二进制交叉熵损失**进行训练。
* 以不同的分辨率处理输入图像：**SigLIP为384×384像素，SAM为1024×1024像素**。
* 训练过程不包含任何额外的数据增强。
* 特征图保持维度为**h1=w1=24和h2=w2=64**，稀疏提示中**M=36**。
* 在训练期间，**CLIP和SAM骨干都保持冻结状态**，只有额外的**低秩参数被更新**。SAM的**轻量级掩码解码器**也包含在训练过程中。 
* 为了优化，该工作采用**AdamW**，**初始学习率为1e−4**，**余弦退火**学习率调度器和**线性预热**策略。
* **epoch=200，batch_size=32**。
* 在**8个NVIDIA A100 GPU**上实施，利用**DeepSpeed ZERO 2**支持分布式计算。

### 与SOTA的比较

使用几种最先进的参考图像分割方法评估了RSRefSeg。LGCE、RMISN和FIANet是专门为遥感图像设计的，而其他方法主要针对自然图像。

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504102346431.png)

该模型在较高IoU阈值下的卓越精度表明其**能够生成高精度的参考掩模**，从而验证了基础模型在遥感参考图像分割中的有效性。

### 消融实验

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504102351266.png)

* **改变SAM编码器中可训练参数数量（r=8,16,32）**。结果：最佳数量的可训练参数对于有效性能至关重要。
	* 参数不足：阻碍跨域知识转移
	* 参数过多：促进了快速训练收敛，但会过度拟合。
* 在CLIP编码器的等级固定为16的情况下，尝试了**微调文本编码器的不同层**。 结果：微调文本编码器显著提高了性能，挑战了语言模型由于域不变性而不需要参数调整的传统假设。这一改善可以归因于两个因素：
	* **引用表达式**通常由**短语**组成，而不是CLIP原始训练中使用的完整句子；
	* 这项任务强调判别**语义学**而不是一般语义理解，这表明存在**特定领域**的变化。
* 在提示器中测试了**不同的空间下采样率（2、4和8）**。结果：最佳提示维度对于卓越的性能至关重要。过多或不足的提示偏离了SAM的原始训练范式，导致次优性能。
* 评估了**密集提示对整体性能的影响**。结果：密集提示为解码提供了优越的先验，它们与稀疏提示的组合产生了最佳结果。

# 结论

* 本文通过利用基础模型的知识来引用遥感图像分割，解决了当前方法在细粒度语义对齐和文本视觉一致性方面的局限性。
* 介绍了RSRefSeg，这是一种包含12亿参数的基础模型。该模型的AtnPrompter架构通过将粗粒度文本语义激活的视觉特征转换为SAM模型的提示输入，将CLIP和SAM基础模型桥接起来，从而能够生成精确的引用掩码。
* 该工作对RRSIS-D数据集的实验评估证明了RSRefSeg组件的有效性。该模型的整体性能超过了当代方法，获得了最先进的结果，并验证了基础模型在理解多模态遥感任务方面的有效性。

# 一些其他知识点
## bimodal feature extraction（双峰特征提取）
[双峰特征提取是什么-秘塔AI搜索](https://metaso.cn/search/8598646243021402112?q=%E5%8F%8C%E5%B3%B0%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96%E6%98%AF%E4%BB%80%E4%B9%88)



