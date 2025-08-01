# 摘要
**问题一**：像素级解译是遥感图像应用的关键环节，但目前普遍存在需要大量**人工标注**的局限性。
**解决方案一**：将开放词汇语义分割（OVSS）引入遥感领域。

**问题二**：遥感图像对低分辨率特征较为敏感，预测掩码中会出现目标形状失真和边界拟合不佳的问题。
**解决方案二**：一种简单通用的**上采样器SimFeatUp**，以==无训练的方式==恢复深度特征中丢失的空间信息。此外，基于对CLIP中局部patch令牌对`[CLS]`令牌的异常响应观察，我们提出通过**直接的减法运算来缓解patch令牌中的全局偏差**。

# 1.引言
遥感图像涉及到更多的空间分辨率、时间维度和物体视角，所以为其他数据模型（如自然图像）设计的解决方案对于遥感图像可能不是最佳的。

像素级感知即分割的应用比实例级感知更为频繁，而对像素级标注的需求加剧了获取大规模标签的难度。

从经验上看，我们认为这些问题在很大程度上可以归因于**过低的特征分辨率**：在当前基于CLIP的OVSS范例中，来自CLIP的特征映射被下采样到原始图像的1/16 （vitb /16）。因此，在本文中，我们提出了一种简单而通用的**特征上采样器SimFeatUp**，其训练目标是在少量未标记的图像上重建内容不变的高分辨率（HR）特征，并且可以在训练后对任意遥感图像特征进行上采样。由于SimFeatUp的这一特性，它可以作为一个通用的外部单元用于无需培训的OVSS框架。此外，CLIP在图像级别进行训练，它使用`[CLS]`令牌作为整个图像的表示，并将全局属性附加到本地令牌上。然而，在OVSS中，这种**全局属性会使局部特征对patch级推理产生偏差**。我们发现，**对局部patch特征和全局特征进行简单的相减运算可以有效地减小全局偏置**。大量的定量和定性实验表明，我们的方法比以前的工作有更好的分割质量。

**贡献**：
* 我们提出了SimFeatUp，一种用于无训练OVSS的通用特征上采样器，它可以鲁棒地对低分辨率（LR）特征上采样，并保持与图像内容的语义一致性。
* 我们提出了一种非常简单直接的方法来缓解CLIP的全局偏差问题，即执行局部和全局令牌的减法操作。
* 我们最后提出的模型，名为SegEarth-OV，在17个遥感数据集上实现了最先进的性能，涵盖语义分割、建筑提取、道路提取和洪水检测任务。

# 2.相关工作
* **视觉语言模型**：CLIP
* **监督语义分割**：feature up构建了一个模型无关的上采样方案，该方案使用多视图一致性损失，与nerf有很深的相似之处。然而，它只是用标签来探索这种情况。受FeatUp的启发并在其基础上构建，本工作中提出的SimFeatUp能够在没有任何标签的情况下显著改善OVSS。
* **开放词汇语义分割**：我们将目前基于clip的OVSS方法分为两组：需要培训的和不需要培训的。
	* 前者允许以监督或弱监督的方式在一些基类上训练模型。
		* 一些研究尝试训练一个可以自然地进行密集预测的定位感知CLIP。
		* 而另一些研究则选择CLIP预训练参数的一个子集或将有限数量的可训练参数引入冻结的CLIP，即对CLIP进行微调以适应基类的密集预测。
	* 无需训练的OVSS方法强调利用CLIP固有的定位能力，对特征或结构进行有限的手术。
		* MaskCLIP率先在CLIP图像编码器的注意力池层去除查询和关键投影。随后的研究充分探索了自我注意（即qq， k-k或v-v自我注意），这些修改在一定程度上减轻了CLIP的噪声激活和空间不变感知。
		* 另一个流是两阶段方法，它首先生成与类别无关的掩码建议，然后对掩码进行分类。
		* 此外，还可以引入其他一些基础模型（如SAM、Stable Diffusion）来增强CLIP的定位能力。

# 3.预备
![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202507071543618.png)

(a)是SimFeatUp的培训过程。CLIP是冻结的，只有SimFeatUp在推理中有用。
(b)为SegEarth-OV的推理过程。来自CLIP的LR特征映射由SimFeatUp上采样，然后减去`[CLS]`令牌以减轻全局偏差。
## 3.1.CLIP
在基于vit的CLIP中，图像编码器由一系列Transformer块组成。设 $X = [x_{cls}, x_1，…, x_{h×w}]^T∈R^{(hw+1,d)}$ 表示最后一个块的输入，其中h和w表示特征图的高度和宽度，d表示标记的维度，$x_{cls}$ 是一个可学习的全局标记，其他是来自不同图像patch的局部标记。该区块的正向过程可表示为：

$$q = Emb_q(X), k = Emb_k(X), v = Emb_v(X)$$
$$y = X+SA (q, k, v)\tag{1}$$
$$ z = y+FFN(LN(y))$$

其中q， k， v分别表示Query， Key, Value。Emb由层归一化（LN）层和线性层组成，FFN表示前馈网络。SA表示标准自注意模块，即 $SA(q, k, v) = softmax(q·k^T/√d)·v$，则投影层将z映射到多模态嵌入空间：

$$O = Proj(z)\tag{2}$$

其中 $O=[o_{cls},o_1,…,o_{h×w}]， oh×w]^T∈R^{(hw+1,c)}$ 表示图像编码器的输出，c表示投影层后的令牌维度，c < d。在CLIP训练过程中，使用$o_{cls}$进行图像级学习；而在OVSS推理中，$O[1: hw+1]$用于补丁级预测。
## 3.2.FeatUp
feature up 旨在训练一个与模型无关的上采样器。通过可学习上采样器σ↑对冰冻骨干网的LR特征`O[1: hw+1]`进行上采样操作，然后使用可学习下采样器σ↓重建LR特征。它的关键见解可以用以下损失函数简单概括：

$$ L_{rec} = ∥O[1 : hw + 1] − σ↓(σ↑(O[1 : hw + 1]))∥_2^2 $$

FeatUp实例化σ↑作为堆叠参数化 JBU（Joint Bilateral Upsampling）操作符。通过加权LR特征的相邻元素来估计上采样的HR特征元素。对于权值的生成，JBU考虑两个因素，即制导特征中相邻元素与中心元素之间的相似度和距离，对应于 $k_{range}$ 和 $k_{spatail}$ 。为简洁起见，我们省略了多视图一致性约束。
# 4.方法
## 4.1.SimFeatUp
feature up 缺乏对无训练设置的一些考虑，导致它不是OVSS任务的最佳选择，特别是在遥感环境中。
* **图像内容保留**：如第3.2节所述，FeatUp的目标是最小化原始LR特征和上下采样后的LR特征（即`σ↓(σ↑(O[1: hw 1]))`）。由于σ↑和σ↓都是可学习的，在这种弱约束下，上下采样过程变成了一个黑盒，中间的HR特征在内容上不能保证完整和与原始图像一致。为了解决这个问题，我们引入了一个额外的图像重建损失来约束HR特征：
	$$L_{img} = ∥I − CRN(σ↑(O[1 : hw + 1])))∥^2$$
	
	其中I表示输入图像。CRN表示内容保留网，接收HR特征作为输入，重建原始图像。具体来说，CRN由两个具有激活的2D卷积层和一个Tanh激活层组成，其中Tanh层被设计为约束输出为`[- 1,1]`，参见VAEs。最后，训练SimFeatUp的损失由Lrec和Limg组成，权重为γ，即：
	$$L = L_{rec} + γL_{img}$$

* **要上采样哪个特征？**：feature up将CLIP的最终输出作为上采样器的输入。这在基于训练的设置中可以很好地工作，例如，线性探针。然而，在没有训练的OVSS中，如第2节所述，vanilla的self-attention会导致较差的表现。因此，目前的OVSS方法将其调制为self-self attention，这一规律也适用于遥感图像。在此前提下，Eq.(1)中的SA会被其他模块取代，直接上采样$O[1: hw+1]$会导致训练和推理不匹配。基于此，我们建议在更早的层上采样CLIP特征。具体来说，我们选择CLIP图像编码器的最后一个Transformer块的输入，即式(1)中的$X[1: hw+1]$。此外，X中令牌的高维导致了高成本的上采样器。因此，我们保留投影层。最终，需要上采样的特征O ' 可以表示为：
	
	$$O^′ = Proj(X[1 : hw + 1]).$$

* **更大的上采样核**：我们在FeatUp中遵循上采样操作符，即参数化的JBU。如第3.2节所述，JBU的上采样核$k_{range}$和$k_{space}$是从制导特征中一个窗口内的元素计算出来的。生成公式如下：
	
	$$K_{spatial} (p, q) = exp\Big(\frac{∥p−q∥^2_2}{2τ^2_{spatial}}\Big)\tag{7}$$
	$$k_{range}(p, q) = softmax_{(a,b)∈Ω}\bigg(\frac{1}{τ^2_{range}}MLP(G[i,j])· MLP(G[a,b])\bigg)\tag{8}$$
	
	式中（p, q）表示在核中的位置。Ω表示从HR RGB图像中提取的制导特征G中以（i, j）为中心的窗口。$τ_{spatial}$和$τ_{range}$是可学习的因子。在遥感图像中，与自然图像不同，目标的大小呈现对数尺度，从米尺度（如树木、花园）到公里尺度（如森林、牧场）。因此，我们设置更大的上采样核以获得更宽的接受域。这里，我们将窗口大小扩展到11 × 11，而在FeatUp中是7 × 7。一个可能的担忧是，更大的接受野可能会引入更多不相关的背景，但是对于$k_{spatial}$，距离越远的点贡献的权重越低，这使得使用更大的上采样核更加合理。

* **简化**：在结构方面，我们简化了FeatUp中的组件。在FeatUp中，参数化的JBU模块被堆叠4次，进行16倍上采样，并且每个JBU模块的参数是独立的。尽管我们将HR特征输入到CRN中以确保其内容的完整性，但每个JBU模块的行为是不确定的。因此，在SimFeatUp中，我们将“JBU Stack”更改为“JBU One”，即只有一个参数化的JBU用于上采样。如果需要16倍上采样，那么它只需要重复执行4次。此外，“JBU One”显著减少了上采样器中可训练参数的数量，并提供了对任意倍数上采样的可能性。

## 4.2.减轻全局偏见
如3.1节所述，在CLIP的训练阶段，包含整个图像全局信息的`[CLS]`令牌通过对比学习将文本嵌入到多模态空间中进行优化。然而，在OVSS的推理阶段，通常会丢弃`[CLS]`令牌，只使用patch令牌与提示词汇进行相似度计算。这意味着训练和推理之间存在差距。事实上，先前的研究也表明：CLIP中的每个局部视觉标记都关注于广泛的位置，并且注意力地图通常具有类似的模式。这表明全局属性被附加到CLIP中的补丁令牌上。这种性质在分类任务中通常不需要考虑，但在密集预测中会严重影响性能。

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202507072316958.png)

图5中的可视化演示了上述阐述。我们使用CLIP提取图5(b)中RGB图像的`[CLS]`标记，并计算其与候选文本嵌入的相似度。图像被识别为建筑物，这是合理的，因为建筑物覆盖了图像中的最大范围。然后，我们计算`[CLS]`令牌与补丁令牌的相似度，如图5(a)所示。高响应区域不仅是有建筑物的区域，一些道路和人行道也被激活，这表明全局偏差污染了局部补丁令牌。基于这一观察结果，我们建议从patch令牌中“减去”一些全局偏差。这个解决方案非常直接和简单，它可以表述为：

$$\hat{O} = O[1: hw+1]−λO[0]\tag{9}$$

其中λ表示强度因子。`O[0]`重复了w次，达到与`O[1: hw+1]`相同的维度。

# 5.实验
## 5.1.数据集
* **语义分割**：我们在OpenEarthMap、LoveDA、iSAID、Potsdam、Vaihingen3、UAVid、UDD5和VDD等8个遥感语义分割数据集上对SegEarth-OV进行了评估。其中，前5个数据集以卫星图像为主，后3个数据集以无人机图像为主。它们包含自定义前景类和背景类。这些数据集的详细描述可在附录7.1中找到。
* **单类提取**：我们选择4建筑提取的数据集，4个道路提取数据集和1个洪水检测数据集用于单类提取的评估。这些数据集包含1个前景类（建筑、道路或洪水）和1个背景类。详细说明见附录7.2-7.4。
* **SimFeatUp的训练数据集**：SimFeatUp只需要图像数据进行训练，为了避免不公平的比较，我们使用了一个公共的遥感图像分类数据集——Million-AID，该数据集主要收集来自谷歌地球的图像。我们随机选择这些图像中的16k来训练SimFeatUp。

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202507081500029.png)

开放词汇语义分割遥感数据的定量比较。评估指标：mIoU。最佳和第二好的表演被突出显示。

## 5.2.设置
* **实施**：
	* 我们的实现基于MMSegmentation6工具包。如果没有指定，我们使用OpenAI提供的CLIP原始预训练权值（ViT-B/16）。
	* 对于文本部分，我们使用OpenAI ImageNet模板作为文本编码器的输入，例如，“一个{类名}的照片”。此外，由于某些类的定义在某些数据集中可能有所不同，因此我们对所有方法都使用了轻微的类重命名技巧。例如，我们将“杂波”重命名为“背景”，将“建筑”重命名为{“建筑”，“房子”，“屋顶”}，并且{}中概率最高的子类将是该类的概率。所有数据集的详细提示类名称列在附录表7中。
	* 对于图像部分，我们将输入图像的长边调整为448，并使用224 × 224窗口和112步幅进行滑动推理。
	* 对于SimFeatUp训练，我们在原始图像上随机裁剪224 × 224个图像补丁。
	* 我们使用两个4090 gpu来训练一个epoch，批处理大小设置为8。我们保留了多视图一致性约束，并应用了随机翻转、平移和缩放。对于上述超参数，所有数据集的γ值设置为0.1，λ设置为0.3。
* **评估**：
	* 我们使用平均交联（mIoU）度量来评估语义分割。对于单类提取，使用前景类的IoU。
* **Baseline**：
	* 我们从自然图像OVSS中吸取了一些教训，这也适用于遥感场景：我们删除了最后一个Transformer块的FFN和残余连接。另外，将最后一个自我注意替换为我们的调制注意，即q-q、k-k、v-v作为v的权重之和：
		
		$$M—SA (q, k,v) = \sum_{i∈\{q,k,v\}}softmax(\frac{i·i^T}{√d})·v\tag {10}$$
		
