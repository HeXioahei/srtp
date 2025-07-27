在Lseg中，文本并没有用作监督信号，监督信号是人工标注好的分割好的图片，这样成本很高，只能得到较小规模的数据集。而GroupVit则直接用文本来做监督信号了。

## **总框架图**：

![4d497f1d-c3c6-420a-98a7-deeaf09b383a](file:///C:/Users/Lenovo/Pictures/Typedown/4d497f1d-c3c6-420a-98a7-deeaf09b383a.png)

有两个输入，一个是图像的patch embedding，即Image Token。一个是可学习的group token（由正态分布采样初始化）。

image: `[224, 224]`, patch_size: `[16, 16]`, 用的是vit small，所以特征维度为384。最开始为了有尽可能多的聚类中心，所以group token设置了64个，维度和image token对齐也为384。

我个人理解可以把group token看作是颜料，用来填充图像的不同部位。也可以理解为是cls token，传统的cls token只有一个，那是因为在分类任务中，只需要将图片提取为一个核心特征即可，但是现在是分割任务，一个图片有多个物种多个特征，每个物种都需要一个cls token，每个cls token都是一个聚类中心。

一共有12层transformer layers，下面6层，中间3层，上面3层。

最后因为对比学习能对于图像和文本的一个整体特征进行对比学习，所以作者对本来长度为8的特征序列又进行了一个ave pooling，将其平均成一个特征，然后就可以和文本进行对比学习了。目标函数和CLIP的目标函数是一样的。

## **Grouping Blocks**:

![3fac1da9-a310-481a-bc38-b96dd316dbd5](file:///C:/Users/Lenovo/Pictures/Typedown/3fac1da9-a310-481a-bc38-b96dd316dbd5.png)

这里其实和自注意力机制还是有点区别的，这里其实主要是一个聚类操作。先将group token(64, 384)与image token(196, 384)进行相乘，得到了一个相似度矩阵(64, 196)，再根据这个相似度矩阵对原本的image token(196, 384)进行聚类中心的分配，得到(64, 384)。但是聚类中心的分配这个过程是不可导的（因为硬聚类属于无监督学习，所以不可导），所以作者加了一个softmax来使其可导。

## **zero-shot推理**：

![228d6d4c-8686-4b60-b488-f0cc5c9d7882](file:///C:/Users/Lenovo/Pictures/Typedown/228d6d4c-8686-4b60-b488-f0cc5c9d7882.png)

图片先进入GroupViT得到8个group token（虽然图片中只写出了两个zI1和zI2），然后C个类别通过prompt形成C个句子后进入TextEncoder形成C个文本特征（zT1~zTC），将文本特征和group token进行相乘得到相似度矩阵，就可以知道每个group token属于哪个类别。再用上采样将group展开，就可以得到分割好的group块，每个块都对应着其已知的类别。

## 结果

**zero shot**：

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504051227741.png)

**消融实验**：

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504051228262.png)

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504051228112.png)

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504051228866.png)

## 局限

* 更像是一个图像编码器，没有用好dense prediction这一特性，无法获取更多尺度的上下文信息。
* 无法很好地分割背景类。该篇论文区分背景类的方法是设置一个阈值，只有当group和文本类别的相似度超过了这个阈值，这个group才算是一个前景类别，不然直接当作背景处理。而这个阈值不太好设定，作者是将其设置为了90%，但是在某些单张图片中类别很多的数据集上分割效果很差。
* GroupViT将分割做得很好了，但是对语义分割做得没有那么好，也就是说，它可以很准确地切割好不同类别的mask，但是却没有很好地分类出每个mask代表着哪一类。


# 模型代码

* AttnBlock
	* MLP
	* Attention
* CrossAttnBlock
	* Attention
* GroupingBlock
	* MLP
	* CrossAttnBlock
	* AssignAttention
* GroupingLayer
	* AttnBlock
	* GroupingBlock
* PatchEmbed
* GroupViT
	* PatchEmbed

* build_text_transform
	* WordAugTokenizeWrapper
	* Tokenize
