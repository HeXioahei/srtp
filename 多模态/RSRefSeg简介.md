# 前言
- 研究背景：遥感图像引用分割 (Referring Remote Sensing Image Segmentation, RRSIS) 是实现**细粒度视觉理解**的关键技术。*简单来说就是要根据**自然语言**的提示来进行精细的分割。*
- 核心挑战：现有方法在**细粒度语义概念对齐**和**文本-视觉信息一致性**方面存在不足。特别是在处理很小的物体时。
- 主要贡献：提出RSRefSeg模型，有效提升遥感图像引用分割的泛化能力和鲁棒性，**探索CLIP的粗粒度语义信息作为SAM的提示**，解决跨域迁移性能下降问题，验证了基础模型在遥感多模态任务中的有效性。

# 主要模块和工作流程

![d75ea7c993fd48e90e8383518d12b63.jpg](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504102132537.jpg)

* CLIP编码：输入图像和文本描述，提取低分辨率视觉特征和文本特征。
* **AttnPrompter生成提示**：基于CLIP的特征，生成稀疏提示和稠密提示。
* SAM分割：利用原始图像和提示，生成最终的分割掩膜。

*打个比方，CLIP就像是一个全面的学者，知道很多知识；AttnPqompter就像是个接地气的讲解员，把CLIP的那些宏泛的知识通俗易懂地告诉SAM；而SAM就是个专精的技术人员，根据AttnPqompter的提示来进行对图像的精细切割。*
# 关键创新点

## Fine-tuned CLIP
CLIP可以进行图像和文本特征的提取与对齐，但是效果较为粗粒度，转移到遥感领域可能效果没那么精确。
### 低秩参数高效微调

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504102138228.png)

W：基础模型的冻结参数矩阵

A、B：新增的低秩可训练参数矩阵

*相当于让CLIP这个全能学者学习到一些更为专业的知识。*

### 区分全局与局部语义

原本的池化层会使得丢失部分空间信息，所以移除池化层，保留原始图像特征图和文本token的隐藏状态。

* 全局语义特征（t_global）：从文本序列的结束token (EOS) 中提取，表示整个句子的语义信息。（但是代码里写的好像不太一样）
* 局部语义特征（t_local）：从其他文本token的嵌入表示中提取，表示特定类别、位置等属性。

二者拼接形成完整特征

*vit中的cls_token与普通token有点像。*

## AttnPrompter
SAM虽然分割功能强大，但是只能根据形式固定的提示来进行分割，如：点、框、粗略掩膜等。所以需要AttnPrompter来给它生成这些提示。

*相当于SAM这个技术人员虽然技艺精湛，但是文化水平不高，需要AttnPrompter来讲解具体做法。*

### 核心思想
用文本语义作为过滤器，将与引用相关的图像视觉信息提取出来。

### 注意力机制

（代码里写的和如下公式不太一样，但逻辑功能应该是差不多的）

* 全局注意力机制：基于全局文本语义 t_global，计算全局激活图。
* 局部注意力机制：基于局部文本语义 t_lobal，计算局部激活图。

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504102208767.png)

v：CLIP提取的视觉特征

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504102220634.png)

输出：稀疏提示 p_sparse（点、框）和稠密提示 p_dense（粗略分割掩膜）

## Fine-tuned SAM

输入原始图像，用AttnPrompter的提示进行分割，选择第一分割结果，如clip同样进行低秩微调，提升其在遥感图像分割上的性能。

# 实验设置

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504241642485.png)

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504241643354.png)








