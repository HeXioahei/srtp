> 博客：[美团/浙大等联合提出MobileVLM | 骁龙888就可以实时运行的多模态大模型，边缘端多模态大模型之战打响了！！！ - 知乎](https://zhuanlan.zhihu.com/p/675392936)
> 
> 代码：[MobileVLM/mobilevlm/model/vision_projector.py at main · Meituan-AutoML/MobileVLM](https://github.com/Meituan-AutoML/MobileVLM/blob/main/mobilevlm/model/vision_projector.py#L8)
>论文：[2312.16886](https://arxiv.org/pdf/2312.16886)
>

参数量级：1-2B
# 主要贡献

1. 提出了**MobileVLM**：
	1. 针对**移动场景**的多模态视觉语言模型的全栈式重构。
	2. 详细、可复现且强大的视觉语言模型。
	3. 使用受控和开源数据集。
	4. 一组高性能的基础语言模型和多模态模型。  
    
2. 对**视觉编码器**的设计进行了广泛的调优研究，并系统地评估了VLM在各种训练范式、输入分辨率和模型大小上的性能敏感性。  
    
3. 设计了一个高效的视觉和文本特征之间的**投影器**，更好地对齐多模态特征，同时减少了推理预算。  
    
4. 专门针对**移动、低功耗设备**进行了优化。  
    
5. 尽管作者主要**关注边缘场景**，但在实际中可以应用于许多任务。

# 相关工作

## ViT
当前视觉感知的主流backbone。

## LLM
虽然大的模型可以提高性能，但是目前越来越追求更轻量化更小的模型。

## VLMs
![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202505242321831.png)

**架构选择**：通常采用ViT，而预训练策略很多样。CLIP式的自然语言监督预训练更受欢迎。

**以数据为中心**：几乎每个新的模型都自带一个新的数据集。构建方式越来越多样。

## Model Compression for LLMs
LLM太大了，需要模型压缩技术来解决资源消耗大和处理速度慢两大问题，同时又不降低性能。这些技术包括但不局限于模型剪枝，量化，知识蒸馏和低秩分解。此外，LLM部署工具也逐渐繁荣。

## VLM Benchmarks
* POPE提供了评估VLMs中幻觉的基准，将评估转换为二分类任务，要求VLMs回答目标是否存在。
* GQA主要关注VLMs在现实世界推理、场景理解和组合性问题回答方面的能力。
* TextVQA包括与图像中的文本相关的问题，评估模型的OCR和推理能力。
* ScienceQA包括覆盖科学主题的多模态多项选择题。
* MME测量VLMs的感知和认知能力，它包括总共14个从粗粒度到细粒度的子任务。MMBench是一个精心构建的多模态数据集，涵盖了20个细粒度的技能，并采用循环评估策略，其中包含Chat-GPT

## Embodied AI 
作者的工作与具身人工智能密切相关。作为人工通用智能的一个核心目标，具身人工智能旨在构建以自我为中心的智能系统，能够通过感知、推理和规划能力与周围环境进行交互。

# MobileVLM
## 总体架构
![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202505242357807.png)

以图像 $X_v \in \mathbb{R}^{H \times W \times C}$ 作为输入，视觉编码器 $F_{\text{enc}}$ 提取视觉嵌入特征 $f \in \mathbb{R}^{N_v \times D_v}$ 用于图像感知，其中 $N_v = HW/P^2$ 表示图像块的数量，$D_v$ 表示视觉嵌入的隐藏层维度。为缓解长序列图像令牌导致的效率问题，我们设计了一个轻量级投影器 $P$ 用于视觉特征压缩与视觉-文本模态对齐。其将 $f$ 转换到词嵌入空间，并适配后续语言模型的输入维度，具体形式如下：
$$H_v = P(f)，f = F_{enc}(X_v).\quad(1)$$
由此我们得到图像标记 $H_v \in \mathbb{R}^{(N_v/4)\times D_t}$ 和文本标记 $H_q \in \mathbb{R}^{N_t\times D_t}$，其中 $N_t$ 表示文本标记数量，$D_t$ 表示词嵌入空间的隐藏维度。鉴于当前多模态大模型（MLLMs）设计范式下大型语言模型（LLM）占据主要计算和内存消耗，我们定制了一系列适用于移动端高速推理的轻量化LLM。该模型以自回归方式基于多模态输入预测响应 $Y_a = \{y_i\}_{i=1}^L$，其中 $L$ 表示输出标记长度。该过程可表述为：
$$ p(Y_a|H_v,H_q) = \prod_{i=1}^{L} p(y_i|H_v,H_q, y_{<i}). \quad(2)$$
## 核心组件

### Visual Encoder
* CLIP ViT-L/14
* 输入分辨率为336×336

### MobileLLaMA
* LLaMA2的sentence piece分词器
* 其词汇表规模为32000，并从头开始训练嵌入表示
* 这种做法有利于后续进行知识蒸馏时无需额外调整。
* 由于资源有限，所有模型在预训练阶段使用的上下文长度均为2k。但是，上下文窗口可以进一步缩放到8k以进行推理。
* 下面列出了其他组件的详细设置：
	* 使用RoPE注入位置信息。
	* 预归一化，使用RMSNorm代替LayerNorm；使用MLP扩展比8/3代替4。
	* 使用SwiGLU激活函数代替GELU。
![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/20250525162915.png)

### LDP（轻量级下采样投影器） 
* Pointwise：相当于 1 * 1 * 3 的卷积核。
* Depthwise：每个卷积核仅作用于输入张量的单个通道。s代表步长。
* [Depthwise卷积与Pointwise卷积](深度学习中的一些概念#Depthwise卷积与Pointwise卷积)
* GeLU
* Layer Normalization
* Pixel-wise addition

**投影器现有两种范式**：
* Q-Former：
	* 显式控制每个查询的视觉标记的数量，以强制提取最相关的视觉信息。
	* 但不可避免地丢失了令牌的空间位置信息，收敛速度慢。此外，在边缘设备上的推理效率低下。
* MLP：
	* 保留了空间信息，
	* 但通常包含无用的标记，如背景。

**问题**：对于patch大小为P的$X_v \in \mathbb{R}^{H \times W \times C}$ 图像，需要在LLM模型中注入$N_v = HW/P^2$ 个视觉令牌，这大大降低了整体推理速度。
**解决方案**：
* 利用深度卷积（PEG的最简单形式）来增强位置信息，并鼓励视觉编码器的局部交互。
* 使用LayerNorm而不是BatchNorm来使训练稳定并且不受batch大小的影响。
**效果**：
* 可以减少大量的视觉token，只包含不到20M的参数，运行速度比视觉编码器快81倍左右。
* 既保留了空间信息，稳定了性能，又足够轻量。
**流程**：
![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202505251650564.png)


# 实验

**三个阶段**：
1. 在纯文本数据集 RedPajama v1 上预训练LLM基础模型。
2. 跟随 Vicuna 对来自第三方平台的人类与ChatGPT之间的多回合对话数据集执行监督微调（SFT）。
3. 使用多模态数据集训练视觉大模型。

## language model pre-training
**相关参数**：
* [global batch_size](全局batch_size)=5,242,880。
* peak learning rate=3e-4，按照余弦策略，降到 3e-5。
* 用 2000 次迭代（iterations）来热身（warm up）
* 使用AdamW优化器，其中β1 = 0.9和β2 = 0.95，权重衰减正则化值为0.1。
* 梯度裁剪阈值设置为1.0。

不严格遵循缩放定律中模型容量和令牌的有效组合。为了使工作可复现，所有模型仅在RedPajama v1数据集中训练1.3T Token。

采用Pytorch lightning框架，具有DeepSpeed后端。使用ZERO 1和梯度累积，以在配备8个NVIDIA Tesla A100 GPU的20个节点上实现1.4B模型的训练速度为每秒18800个 Token ，以及2.7B模型的训练速度为每秒8500个 Token 。

此外，作者还倾向于使用Flash Attention V2来缓解I/O瓶颈，以实现更快的训练。作者随机打乱数据，用固定的种子打乱顺序，因为训练过程可能会间歇性中断，需要重新开始。作者首先将原始数据Token为ID，并将其分块保存为许多桶文件。然后，作者使用**内存映射**来提供所需的I/O速度。

此外，作者将不同的句子打包在一起，其中插入一个EOSToken来设置不同的句子。由于资源有限，作者没有尝试InternLM的设计，这可能通过禁用这样的打包进一步改善模型性能。随着消耗的 Token 增加，整体训练损失减小。




