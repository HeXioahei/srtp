# 借助深度与语言实现开放词汇域泛化语义分割
## 摘要

开放词汇语义分割（OVSS）与语义分割域泛化（DGSS）存在微妙的互补性，这一特性推动了开放词汇域泛化语义分割（OV-DGSS）的研究。开放词汇域泛化语义分割旨在为未见过的类别生成像素级掩码，同时确保模型在未见过的域中保持鲁棒性，这一能力对于恶劣条件下的自动驾驶等现实场景至关重要。本文提出一种全新的单阶段开放词汇域泛化语义分割框架 Vireo，首次整合了开放词汇语义分割与语义分割域泛化的优势。Vireo 以冻结的视觉基础模型（VFMs）为基础，通过深度视觉基础模型（Depth VFMs）融入场景几何信息，提取域不变结构特征。为解决域偏移下视觉与文本模态间的鸿沟，本文提出三个核心组件：（1）几何 - 文本提示（GeoText Prompts），将几何特征与语言线索对齐，并逐步优化视觉基础模型编码器的表示；（2）粗掩码先验嵌入（CMPE），增强梯度流动以实现更快收敛，并强化文本对模型的影响；（3）域 - 开放词汇向量嵌入头（DOV-VEH），融合优化后的结构特征与语义特征，实现鲁棒预测。对这些组件的全面评估验证了设计的有效性。所提 Vireo 框架在域泛化和开放词汇识别任务中均实现了当前最优性能，大幅超越现有方法，为多样化动态环境下的鲁棒视觉理解提供了统一且可扩展的解决方案。代码已开源：[anonymouse-9c53tp182bvz/Vireo](https://github.com/anonymouse-9c53tp182bvz/Vireo)

## 1 引言

开放词汇域泛化语义分割（OV-DGSS）是开放词汇语义分割（OVSS）[1,2,3,4] 与语义分割域泛化（DGSS）[5,6,7,8] 任务的结合。该任务要求训练一个模型：在无法获取目标域样本及新类别标注的情况下，既能为未见过的类别生成像素级分割结果，又能在从未接触过的域（如不同城市、光照环境或气候条件）中保持高性能。例如，当自动驾驶车辆接收到 “我能把车停在那个路障旁边吗？” 这类查询时，其感知系统必须理解语言输入，并在像素级别准确分割出查询所指的物体 —— 即便在光线昏暗、镜头有雨痕或物体具有区域特定视觉外观等恶劣条件下。

  

开放词汇语义分割与语义分割域泛化存在显著共性，二者均可采用多阶段或单阶段策略实现。在多阶段开放词汇语义分割方法中 [1]，先生成候选区域或粗掩码，再通过文本方法对其进行分类；而多阶段语义分割域泛化方法 [9] 则先通过对抗对齐或风格迁移实现域对齐，再基于对齐后的特征训练分割模型。

  

在单阶段开放词汇语义分割方法中 [10,11]，分割头以文本提示为动态条件，直接为每个类别生成掩码。与之不同，单阶段语义分割域泛化方法 [12,13] 在骨干网络或分割头中融入域不变模块，使模型能在统一的前向传播过程中同时学习分割任务与泛化能力，无需分阶段处理。二者的核心差异在于关注重点：开放词汇语义分割需将视觉特征与文本语义融合，以准确识别未见过的类别；而语义分割域泛化则强调对域偏移的鲁棒性。

  

因此，将新类别的开放词汇识别与域鲁棒性整合到统一框架中，面临两大核心挑战：（1）文本 - 视觉对齐模块在源域之外常出现性能退化，即便对于已见过的类别，也会导致显著的性能下降；（2）域不变策略可能会抑制细粒度语义线索，阻碍模型对详细文本查询做出精准响应。

  

近年来，单阶段语义分割域泛化研究越来越多地采用微调冻结视觉基础模型（VFM）各层可学习令牌的策略 [14,15,16,17,18]，以调整其特征表示。而在开放词汇语义分割中，视觉基础模型编码器通常被完全冻结，研究重点集中在设计解码器，使模型具备开放词汇识别能力。这揭示了两种范式间微妙的互补性：语义分割域泛化注重编码器，利用视觉基础模型强大的特征泛化能力学习跨域表示；而开放词汇语义分割则冻结编码器，将重点放在解码器上以实现开放词汇识别。

  

此外，在跨域场景中，深度和几何线索对光照与纹理变化具有较强的不敏感性 [19,20]。它们能提供可靠的空间约束，缓解 RGB 特征的分布偏移，并优化边界定位。DepthForge [21] 等近期研究表明，向冻结的视觉基础模型中注入深度提示可提升域泛化能力。受此启发，本文采用 DepthAnything V2 作为深度视觉基础模型：其多样化的预训练使其能在不同域中实现稳定的深度估计，且完全冻结该模型既能将训练成本降至最低，又能保持实时推理速度。

  

本文提出一种基于视觉基础模型的单阶段开放词汇域泛化语义分割框架，命名为 Vireo。具体而言，在编码器中，视觉基础模型与 DepthAnything 模块均保持冻结状态。视觉基础模型用于稳健捕捉跨域视觉特征，而 DepthAnything 则提取场景的固有几何结构。在此基础上，本文引入几何 - 文本提示（GeoText Prompts），将提取的结构特征与人工提供的文本线索融合，逐步优化冻结视觉基础模型各层之间的特征图。为缓解稀疏梯度通过冻结编码器传播导致的收敛缓慢问题，本文在解码器开头引入粗掩码先验嵌入（CMPE），以注入更密集的梯度信号。该设计不仅加快了掩码监督的收敛速度，还进一步强化了文本先验的影响。随后，本文设计域 - 开放词汇向量嵌入头（DOV-VEH），加强结构模态与文本模态的协同融合，确保几何 - 文本提示所学习的跨域结构特征与开放词汇线索均能在最终预测中得到充分利用。

  

![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27400%27%20height=%27256%27/%3e)![图1：Vireo框架概述及有效性。（Ⅰ）在ACDC和DELIVER数据集的各类恶劣条件下的性能对比，显示Vireo持续优于现有方法；（Ⅱ）极端夜景下分割预测（左）与注意力图（右）的定性可视化，体现Vireo的鲁棒性及与语义线索的精准对齐；（Ⅲ）架构对比：（a）传统开放词汇语义分割和（b）语义分割域泛化流水线分别冻结或微调视觉基础模型，未进行跨模态融合；（c）本文提出的Vireo引入几何-文本提示与深度视觉基础模型融合，同时提升语义对齐与域鲁棒性](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjI1NiIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==)![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27400%27%20height=%27256%27/%3e)![image](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjI1NiIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==)

  

![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27400%27%20height=%27256%27/%3e)![图2：所提Vireo开放词汇域泛化语义分割框架概述。（a）框架总览：几何-文本提示贯穿整个模型——注入可调节Vireo模块以对齐域先验，在粗掩码先验嵌入中复用以引导多尺度特征融合，在域-开放词汇嵌入头中作为开放词汇分割查询，形成统一的端到端流程；（b）可调节Vireo模块：几何-文本提示用于在多层注入结构-文本先验，在冻结中间层之间应用深度感知融合与注意力优化，逐步优化特征表示；（c）粗掩码先验嵌入（CMPE）：视觉基础模型的多尺度特征与几何-文本提示融合，为下游模块提供密集监督与梯度信号；（d）域-开放词汇嵌入头（DOV-VEH）：多级特征经像素解码器与Transformer解码器处理，在开放词汇文本查询引导下生成最终预测](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjI1NiIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==)![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27400%27%20height=%27256%27/%3e)![image](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjI1NiIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==)

  

针对挑战 1：研究发现，几何 - 文本提示不仅能捕捉冻结视觉基础模型编码器内的结构与语义线索，还能引导其特征表示的逐步优化。针对挑战 2：粗掩码先验嵌入增强了梯度向编码器的反向传播，而重新设计的域 - 开放词汇嵌入头则深化了视觉与文本先验的融合。这些组件共同构成统一框架，同时实现域鲁棒性与强大的开放词汇泛化能力。本文的主要贡献总结如下： （1）提出 Vireo—— 一种基于视觉基础模型的全新单阶段开放词汇域泛化语义分割框架； （2）引入几何 - 文本提示，通过注入 DepthAnything 的几何线索并将其与文本语义对齐，逐步优化冻结视觉基础模型的特征，实现编码器各层间的结构 - 语义融合； （3）设计两个互补模块 —— 用于增强梯度流动的粗掩码先验嵌入（CMPE）与用于融合视觉 - 文本先验的域 - 开放词汇嵌入头（DOV-VEH），共同提升模型在域偏移与未见过类别下的分割性能。

## 2 相关工作

### 开放词汇语义分割

开放词汇语义分割（OVSS）旨在基于任意文本描述对物体进行分割，突破固定预定义类别的限制。视觉语言模型（VLMs），尤其是 CLIP [16]（其实现了视觉与文本表示的对齐），是开放词汇语义分割的关键支撑技术 [22,23,1,24]。

### 语义分割域泛化

语义分割域泛化（DGSS）致力于解决模型因域偏移（数据分布差异，如光照、天气）遇到未见过的目标域时性能下降的问题。数据增强 [25,26] 与学习域不变表示 [27,28] 是提升模型鲁棒性的两大核心策略。附录中对这些研究领域的详细讨论，进一步明确了本文贡献在相关研究大背景下的定位。

## 3 方法 ——Vireo

### 3.1 框架总览

如图 2 所示，本文框架包含三个专为开放词汇域泛化语义分割设计的核心模块：

  

- 带几何 - 文本提示的可调节 Vireo 模块：引入几何 - 文本提示，在冻结视觉基础模型的各层注入并优化几何与文本信息；
- 粗掩码先验嵌入（CMPE）：生成粗先验掩码，引导分割过程，并强化从解码器到冻结编码器的梯度流动；
- 域 - 开放词汇向量嵌入头（DOV-VEH）：整合视觉、几何与语义特征，生成最终的开放词汇域泛化语义分割预测结果。

  

输入 RGB 图像\(I \in \mathbb{R}^{H \times W \times 3}\)被复制后，分别输入两个冻结编码器 —— 视觉基础模型编码器\(F^V\)与 DepthAnything 编码器\(F^D\)。这两个骨干网络分别提取一系列多尺度特征\(f_l^V\)与\(f_l^D\)，其中l表示编码器层索引，取值范围为\([1, L]\)。

  

同时，一组类别标签\(C=\{c_1, \dots, c_K\}\)被转换为自然语言提示\(T=\{prompt_1, \dots, prompt_K\}\)，并通过冻结的 CLIP 文本编码器\(\overline{F^T}\)编码，得到文本嵌入\(t_k \in \mathbb{R}^d\)。这些特征作为整个框架的共享语义先验，在初始化时预先计算一次。在可调节 Vireo 模块中，每一层接收元组\((f_l^V, f_l^D, t_k)\)，并应用几何 - 文本提示\(P_l\)对齐与优化视觉特征。具体通过计算跨模态注意力图\(A_l=Attn(P_l, f_l^V, f_l^D, t_k)\)，随后进行特征融合与投影实现。优化后的输出\(f^{\hat{V}_t}\)被反馈回视觉基础模型，以更新其层激活值。

  

本文从视觉基础模型编码器的第 8、12、16、24 层（记为\(l_1=8, l_2=12, l_3=16, l_4=24\)）中选取优化后的视觉特征\(\{f_{l_i}^V\}_{i=1}^4\)，用于下游解码。在粗掩码先验嵌入（CMPE）模块中，这些特征首先被上采样至统一的空间分辨率，再通过通道 - 空间注意力门控函数\(G(\cdot)\)处理。门控输出经融合后形成全局粗特征\(f^M \in \mathbb{R}^d\)。将\(f^M\)投影至与文本嵌入\(t_k \in \mathbb{R}^d\)相同的维度，并进行匹配，生成粗类别概率图\(M \in \mathbb{R}^{H \times W \times K}\)。该概率图既作为弱监督信号，又作为先验用于构建查询嵌入。查询先验与几何 - 文本提示相加后，传递至最终的分割头。在域 - 开放词汇向量嵌入头（DOV-VEH）中，多尺度特征\(\{\hat{f}_{l_i}^V\}_{i=1}^4\)先经过像素解码器\(D_p(\cdot)\)以增强空间表示，再输入 Transformer 解码器\(D_T(\cdot)\)（该解码器利用位置嵌入）。几何 - 文本提示作为可学习查询，与解码特征及文本嵌入交互，生成像素级掩码嵌入\(E_{mask}(x,y) \in \mathbb{R}^d\)与分类嵌入\(E_{cls}(k) \in \mathbb{R}^d\)。最终预测结果\(\hat{M}(x,y,k) \in \mathbb{R}^{H \times W \times K}\)提供像素级语义预测，兼具细粒度细节与开放词汇泛化能力。

### 3.2 带几何 - 文本提示的可调节 Vireo 模块

为提升效率，本文首先利用冻结的 CLIP 文本编码器\(\dot{F}^T\)预计算并共享文本提示嵌入。具体而言，将一组类别标签\(C=\{c_1, \dots, c_K\}\)转换为语言提示\(T=\{prompt_1, \dots, prompt_K\}\)，并编码为\(t_k=F^T(prompt_k)\)（其中\(t_k \in \mathbb{R}^d\)）。这些嵌入在所有几何 - 文本提示层、粗掩码先验嵌入模块与域 - 开放词汇向量嵌入头模块中复用，以避免冗余计算。

  

推理过程中，输入图像\(I \in \mathbb{R}^{H \times W \times 3}\)同时由冻结视觉编码器\(F^V\)与冻结深度编码器\(F^D\)（如 DepthAnything）处理。对于每个选定的层\(l \in \{1, \dots, L\}\)，可得到视觉特征图\(f_l^V=F_l^V(I)\)与深度特征图\(f_l^D=F_l^D(I)\)。可调节 Vireo 模块的每一层接收元组\((f_l^V, f_l^D, \{t_k\})\)以及层特定的几何 - 文本提示\(P_l \in \mathbb{R}^{N \times d}\)（其中N为可学习提示的数量）。

  

提示\(P_l\)首先通过融合块与文本嵌入交互，再通过跨注意力机制关注\(f_l^V\)与\(f_l^D\)： \(\mathcal{A}_l=CrossAttn\left(P_l, f_l^V, f_l^D,\left\{t_k\right\}\right) \quad(1)\)

  

注意力输出通过加权求和融合，再经过 MLP 投影层处理，并与\(P_l\)进行逐元素相乘。随后通过残差连接，将该结果与原始特征图\(f_l^V\)相加，得到优化后的视觉表示\(\hat{f}_l^V\)。最后，另一 MLP 将\(\hat{f}_l^V\)转换为视觉基础模型下一层的输入，同时将更新后的几何 - 文本提示\(P_{l+1}\)向前传递。这种逐步优化在所有选定层中持续进行，使模型能在多尺度下注入并对齐几何与语义信息，从而提升跨域鲁棒性与开放词汇泛化能力。

### 3.3 粗掩码先验嵌入（CMPE）

本文从视觉基础模型编码器的第 8、12、16、24 层（分别记为\(l_1=8, l_2=12, l_3=16, l_4=24\)）中选取优化后的视觉特征\(\{\hat{f}_{l_i}^V\}_{i=1}^4\)。通过双线性插值将每个特征图上采样至统一的空间分辨率\((H \times W)\)，再输入自适应注意力门（AAG）\(G(\cdot)\)—— 该门控函数可增强含信息通道与空间区域。具体而言，自适应注意力门先应用两个 1×1 卷积（后接 ReLU 与 Sigmoid 激活函数）实现通道注意力，再通过 3×3 卷积（后接 Sigmoid 激活函数）实现空间注意力。

  

将经注意力处理后的特征沿通道维度拼接，再通过 1×1 卷积融合，恢复嵌入维度d，得到融合特征表示：\(f^M=Fuse(G(\hat{f}_{l_i}^V))\)。将\(f^M\)与最后一层输出\(\hat{f}_{l_4}^V\)进行残差相加，得到更新后的掩码特征：\(f^M=f^M+\hat{f}_{l_4}^V\)。将该融合特征\(f^M(x,y) \in \mathbb{R}^d\)投影至与文本嵌入\(t_k \in \mathbb{R}^d\)相同的维度，并通过爱因斯坦求和进行比较，生成粗语义概率图\(\tilde{M}(x,y,k)=<f^M(x,y), t_k>\)（其中\(M \in \mathbb{R}^{B \times K \times H \times W}\)）。该粗掩码通过分割损失监督，以增强梯度在冻结编码器中的流动。

  

为向下游分割头生成查询先验，首先在空间域对M进行归一化，得到注意力权重： \(\alpha_k(x,y)=\frac{exp(\mathcal{M}(x,y,k))}{\sum_{x',y'} exp\left(\mathcal{M}\left(x',y',k\right)\right)}\) \(f_k^{class}=\sum_{x,y} \alpha_k(x,y) \cdot f^M(x,y)\) 随后，通过对类别M的空间加权，计算类别特定的聚合特征\(f^M\)。

  

将每个\(f_k^{class}\)投影至嵌入空间，得到\(e_k^{class} \in \mathbb{R}^d\)，并与一组可学习查询向量\(\{q_j\}_{j=1}^{N_q}\)融合，生成查询先验： \(q_j^{prior}=\sum_{k=1}^K Softmax\left(<q_j, e_k^{class}>\right) \cdot e_k^{class} \quad(4)\)

  

最终的查询先验\(\{q_j^{prior}\}_{j=1}^{N_q}\)与对应的几何 - 文本提示相加后，传递至域 - 开放词汇向量嵌入头（DOV-VEH）。

### 3.4 域 - 开放词汇向量嵌入头（DOV-VEH）

域 - 开放词汇向量嵌入头模块接收来自视觉基础模型编码器第 8、12、16、24 层的优化后多尺度特征\(\{\hat{f}_{l_i}^V\}_{i=1}^4\)，以及更新后的几何 - 文本提示\(\{P_l\}\)。这些特征首先经像素解码器\(D_p(\cdot)\)处理 —— 该解码器利用多尺度跨注意力提取丰富的空间上下文，得到\(f^{pix}=D_p(\hat{f}_{l_i}^V)\)。融合特征\(f^{pix} \in \mathbb{R}^{H \times W \times d}\)通过 1×1 卷积压缩维度，并融入正弦位置编码以保留空间结构。增强后的特征输入 Transformer 解码器\(D_T(\cdot)\)，其中几何 - 文本提示作为可学习查询。通过与\(f^{pix}\)的自注意力和跨注意力堆叠层，模型捕捉细粒度的视觉 - 语义对齐，在每个空间位置生成一组高分辨率掩码特征\(E_{mask}(x,y) \in \mathbb{R}^d\)。

  

同时，几何 - 文本提示\(\{P\}\)经两层 MLP 处理后，与文本嵌入\(\{t_k\}\)（由 CLIP 文本编码器预计算）交互，生成分类级表示\(E_{cls}(k) \in \mathbb{R}^d\)。最终分割预测\(\hat{M} \in \mathbb{R}^{H \times W \times K}\)通过对两种嵌入进行爱因斯坦求和生成： \(\hat{\mathcal{M}}(x,y,k)=\sum_{d=1}^D \mathcal{E}_{mask}(x,y,d) \cdot \mathcal{E}_{cls}(k,d)\) 其中D为特征嵌入维度。该设计使域 - 开放词汇向量嵌入头能够生成像素级分割掩码，既保证空间准确性，又能与开放词汇文本查询实现语义对齐。

## 4 实验

### 4.1 数据集与评估协议

本文在 6 个真实世界数据集（Cityscapes [37]、BDD100K [38]、Mapillary [39]、ACDC [40]、ADE150 [41]、ADE847 [41]）与 2 个合成数据集（GTA5 [42]、DELIVER [43]）上对 Vireo 进行评估。

  

- Cityscapes（简称 City.）：自动驾驶数据集，包含 2975 张训练图像与 500 张验证图像，分辨率均为 2048×1024；
- BDD100K（简称 BDD.）与 Mapillary（简称 Map.）：分别提供 1000 张和 2000 张验证图像，分辨率分别为 1280×720 和 1920×1080；
- ACDC：包含 406 张极端条件（夜晚、下雪、大雾、下雨）下拍摄的验证图像，分辨率均为 1920×1080；
- GTA5：合成数据集，包含 24966 张来自视频游戏的带标注图像；
- DELIVER：多模态合成数据集，涵盖 5 种天气条件（阴天、大雾、夜晚、下雨、晴天），包含 3983 张训练图像、2005 张验证图像与 1897 张测试图像，每张图像分辨率为 1042×1042，包含 25 个类别；
- ADE150 与 ADE847：均为 ADE20K 数据集 [41] 的子集，各包含 2000 张分辨率可变的验证图像（源自 SUN、Places 等多样化场景），分别涵盖 150 个和 847 个语义类别。

  

遵循现有语义分割域泛化评估协议，本文以一个数据集作为源域进行训练，并在多个未见过的目标域上验证。采用三种标准评估设置：（1）Cityscapes→ACDC；（2）GTA5→Cityscapes、BDD100K、Mapillary；（3）Cityscapes→BDD100K、Mapillary、GTA5。为评估所提开放词汇域泛化语义分割方法，并对比其开放词汇能力与开放词汇语义分割方法，额外增加两种配置：（4）Cityscapes→DELIVER、ADE150、ADE847；（5）GTA5→DELIVER、ADE150、ADE847。评估指标采用平均交并比（mIoU）。

### 4.2 部署细节与参数设置

本文基于 MMSegmentation [44] 代码库实现模型，采用 AdamW 优化器，初始学习率为 1e-4，权重衰减为 0.05，epsilon 设为 1e-8，beta 参数为 (0.9, 0.999)。总训练迭代次数为 40000 次（与 REIN 方法一致），采用多项式学习率衰减策略 —— 在 40000 次迭代内将学习率降至 0，衰减幂为 0.9，不进行基于 epoch 的预热。数据增强策略包括多尺度调整大小、随机裁剪（固定裁剪尺寸与类别比例约束）、随机水平翻转与光度失真。所有实验在 NVIDIA RTX A6000 GPU 上进行，批大小为 8，训练耗时约 14 小时，GPU 内存峰值占用约 45GB。

### 4.3 性能对比

#### 域泛化能力

表 1 总结了现有主流开放词汇语义分割（OVSS）与语义分割域泛化（DGSS）方法在两种跨域设置（Cityscapes→ACDC、BDD100K、Mapillary、GTA5 与 GTA5→Cityscapes、BDD100K、Mapillary）下的评估结果。结果显示，本文方法在所有目标数据集上均实现卓越性能，显著优于其他开放词汇语义分割 / 语义分割域泛化方法。此外，图 3 的可视化结果表明，Vireo 在极端天气条件及行人和车辆密集场景中均能生成理想的预测结果。

  

表 1：Vireo 与现有开放词汇语义分割（OVSS）和语义分割域泛化（DGSS）方法在 Cityscapes→ACDC+BDD100K+Mapillary 及 GTA5→Cityscapes+BDD100K+Mapillary 泛化设置下的性能对比（前三名结果分别标注为最佳、第二、第三，单位：%）

  

|方法|会议及年份|基于 Cityscapes 训练||||||基于 GTA5 训练||||
|---|---|---|---|---|---|---|---|---|---|---|---|
|||夜晚 - ACDC|大雾 - ACDC|下雨 - ACDC|下雪 - ACDC|BDD100K|Mapillary|GTA5|Cityscapes|BDD100K|Mapillary|
|开放词汇语义分割方法||||||||||||
|FC-CLIP [29]|NeurIPS 2023|40.8|64.4|63.2|61.5|55.92|66.12|47.12|53.54|51.41|58.60|
|EBSeg [30]|CVPR 2024|27.7|56.5|51.8|50.1|48.91|63.40|42.61|44.80|40.59|56.28|
|CAT-Seg [10]|CVPR 2024|37.2|58.3|45.6|49.0|48.26|54.74|45.18|43.52|44.28|50.88|
|SED [3]|CVPR 2024|38.7|69.0|56.4|60.2|53.30|64.32|48.93|47.45|48.16|57.38|
|语义分割域泛化方法（基于 ResNet）||||||||||||
|IBN [31]|ECCV 2018|21.2|63.8|50.4|49.6|48.56|57.04|45.06|-|-|-|
|RobustNet [9]|CVPR 2021|24.3|64.3|56.0|49.8|50.73|58.64|45.00|36.58|35.20|40.33|
|WildNet [32]|CVPR 2022|12.7|41.2|34.2|28.4|50.94|58.79|47.01|44.62|38.42|46.09|
|语义分割域泛化方法（基于 Transformer）||||||||||||
|HGFormer [33]|CVPR 2023|52.7|69.9|72.0|68.6|53.40|66.90|51.30|-|-|-|
|CMFormer [34]|AAAI 2024|33.7|77.8|67.6|64.3|59.27|71.10|58.11|55.31|49.91|60.09|
|语义分割域泛化方法（基于视觉基础模型）||||||||||||
|REIN [35]|CVPR 2024|55.9|79.5|72.5|70.6|63.54|74.03|62.41|66.40|60.40|66.10|
|FADA [36]|NeurIPS 2024|57.4|80.2|75.0|73.5|65.12|75.86|63.78|68.23|61.94|68.09|
|开放词汇域泛化语义分割方法||||||||||||
|Vireo（本文方法）|-|60.6|82.3|76.3|76.2|66.73|75.99|67.86|70.69|62.91|69.63|

#### 开放词汇能力

表 2 对比了 Vireo 与其他开放词汇语义分割（OVSS）方法在 Cityscapes→DELIVER（晴天、下雨、夜晚、阴天、大雾）、ADE150、ADE847 配置下的性能。结果显示，传统开放词汇语义分割方法在极端场景（如夜晚）中性能大幅下降，而本文模型通过深度几何特征增强，保持了稳健性能，相较于性能最强的开放词汇语义分割基准模型，提升幅度至少达 5%。此外，如图 4 所示，DELIVER 数据集中新开放类别与极端天气条件的共存，导致开放词汇语义分割方法出现大量假阳性与假阴性标注，而 Vireo 的表现明显更优。

  

表 2：Vireo 与现有开放词汇语义分割（OVSS）方法在 Cityscapes→DELIVER+ADE150+ADE847 泛化设置下的性能对比（前三名结果分别标注为最佳、第二、第三，单位：%）

  

|方法|会议及年份|晴天 - DELIVER|夜晚 - DELIVER|阴天 - DELIVER|下雨 - DELIVER|大雾 - DELIVER|ADE150|ADE847|
|---|---|---|---|---|---|---|---|---|
|基于 Cityscapes 训练|||||||||
|开放词汇语义分割方法：|||||||||
|FC-CLIP [29]|NeurIPS 2023|16.93|14.93|17.50|16.59|17.26|16.12|6.29|
|EBSeg [30]|CVPR 2024|26.41|15.50|22.62|20.35|22.00|12.75|3.75|
|CAT-Seg [10]|CVPR 2024|28.21|20.56|26.22|26.53|24.80|20.19|6.95|
|SED [3]|CVPR 2024|27.14|22.79|24.40|25.18|25.25|18.86|5.45|
|开放词汇域泛化语义分割方法：|||||||||
|Vireo（本文方法）|–|35.73|27.51|32.34|31.80|32.72|21.37|7.31|
|基于 GTA5 训练|||||||||
|开放词汇语义分割方法：|||||||||
|FC-CLIP [29]|NeurIPS 2023|22.24|18.58|18.50|16.59|19.12|15.47|5.73|
|EBSeg [30]|CVPR 2024|32.32|20.05|26.19|26.19|28.69|11.87|4.19|
|CAT-Seg [10]|CVPR 2024|28.59|23.49|27.31|27.94|27.66|20.45|7.18|
|SED [3]|CVPR 2024|26.56|21.18|24.95|24.58|26.17|19.57|6.80|
|开放词汇域泛化语义分割方法：|||||||||
|Vireo（本文方法）|–|38.49|29.89|33.89|33.46|35.80|21.23|7.68|

  

![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27400%27%20height=%27256%27/%3e)![图3：现有语义分割域泛化（DGSS）方法与Vireo在Cityscapes→ACDC未见过目标域（夜晚、下雪、下雨、大雾条件）下的关键分割示例](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjI1NiIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==)![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27400%27%20height=%27256%27/%3e)![image](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjI1NiIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==)

  

![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27400%27%20height=%27256%27/%3e)![图4：现有开放词汇域泛化语义分割（OV-DGSS）方法与Vireo在Cityscapes→DELIVER未见过类别及跨域场景下的关键分割示例对比](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjI1NiIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==)![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27400%27%20height=%27256%27/%3e)![image](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjI1NiIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==)

  

表 3 对比了基于 Cityscapes 训练的 Vireo 与其他开放词汇语义分割方法在 DELIVER 数据集 5 种天气条件下 “已见类别” 与 “未见类别” 的性能。Vireo 在两类类别中均实现最高 mIoU：在已见类别上，相较于第二名方法提升约 7-10 个百分点；在未见类别上，提升约 2-3 个百分点。尽管所有方法在未见类别上的 mIoU 均较低（凸显开放词汇分割的难度），但 Vireo 显著缩小了这一差距，证明其深度几何引导与跨域对齐能力有助于提升新类别的识别效果。总体而言，Vireo 在不同天气场景与类别类型中均保持优势，展现出更强的域外传泛化能力。

### 4.4 消融实验

#### 稳健的性能提升

表 4 对比了在 GTA5→Cityscapes+BDD100K+Mapillary 迁移任务中，轻量级微调方法在 EVA02-Large 与 DINOv2-Large 模型上的参数开销与 mIoU。Vireo 仅需约 380 万可训练参数，就在两个模型上均实现领先性能：在 EVA02 上达到 66.0%（比 FADA 高 1.1%），在 DINOv2 上达到 67.7%（比 FADA 高 1.6%）。其他方案（REIN、LoRA、VPT、AdaptFormer）虽能改进冻结骨干网络的性能，但在准确率 - 效率平衡上均不及 Vireo。与全微调相比，Vireo 大幅降低了训练成本，同时进一步提升了 mIoU，表明其深度几何提示能在不同视觉基础模型间实现泛化。

  

表 4：GTA5→Cityscapes+BDD100K+Mapillary 设置下，Vireo 与其他语义分割域泛化（DGSS）方法的性能对比（EVA02（Large）[45,46]、DINOv2（Large）[53]）

  

|微调方法|可训练参数（百万）|mIoU|||平均值|
|---|---|---|---|---|---|
|||Cityscapes|BDD100K|Mapillary||
|（基于 EVA02（Large））||||||
|全微调|304.24|62.1|56.2|64.6|60.9|
|+AdvStyle[47]|304.24|63.1|56.4|64.0|61.2|
|+PASTA[48]|304.24|61.8|57.1|63.6|60.8|
|+GTR-LTR[49]|304.24|59.8|57.4|63.2|60.1|
|冻结|0.00|56.5|53.6|58.6|56.2|
|+AdvStyle[47]|0.00|51.4|51.6|56.5|53.2|
|+PASTA[48]|0.00|57.8|52.3|58.5|56.2|
|+GTR-LTR[49]|0.00|52.5|52.8|57.1|54.1|
|+LoRA[50]|1.18|55.5|52.7|58.3|55.5|
|+AdaptFormer[51]|3.17|63.7|59.9|64.2|62.6|
|+VPT[52]|3.69|62.2|57.7|62.5|60.8|
|+REIN[35]|2.99|65.3|60.5|64.9|63.6|
|+FADA[36]|11.65|66.7|61.9|66.1|64.9|
|+Vireo（本文方法）|3.78|68.5|62.1|67.4|66.0|
|（基于 DINOv2（Large））||||||
|全微调|304.20|63.7|57.4|64.2|61.7|
|+AdvStyle[47]|304.20|60.8|58.0|62.5|60.4|
|+PASTA[48]|304.20|62.5|57.2|64.7|61.5|
|+GTR-LTR[49]|304.20|62.7|57.4|64.5|61.6|
|冻结|0.00|63.3|56.1|63.9|61.1|
|+AdvStyle[47]|0.00|61.5|55.1|63.9|60.1|
|+PASTA[48]|0.00|62.1|57.2|64.5|61.3|
|+GTR-LTR[49]|0.00|60.2|57.7|62.2|60.0|
|+LoRA[50]|0.79|65.2|58.3|64.6|62.7|
|+AdaptFormer[51]|3.17|64.9|59.0|64.2|62.7|
|+VPT[52]|3.69|65.2|59.4|65.5|63.3|
|+REIN[35]|2.99|66.4|60.4|66.1|64.3|
|+FADA[36]|11.65|68.2|62.0|68.1|66.1|
|+Vireo（本文方法）|3.78|70.7|62.9|69.6|67.7|

  

表 6 报告了 Vireo 与几种轻量级微调方案在四种骨干网络（CLIP-L、SAM-H、EVA02-L、DINOv2-L）上的参数开销与 mIoU。与基于提示的 REIN 基准模型相比，Vireo 仅增加约 79 万额外参数，却在所有骨干网络上均实现最高平均 mIoU：在参数受限的 CLIP-L 上，性能提升最为显著；即便在较大的 EVA02-L 与 DINOv2-L 模型上，Vireo 仍比 FADA 等更复杂的适配器高出 1-2 个 mIoU，凸显其参数效率与在不同骨干网络上的可扩展性。

  

表 5 显示，单独使用 Depth Anything V2 在所有六种场景中可实现约 0.6% 的 mIoU 提升，而结合深度增强与注意力优化（DA+AO）可进一步提升约 1%。域 - 开放词汇向量嵌入头（DOV-VEH）与粗掩码先验嵌入（CMPE）模块各带来 0.5%-0.8% 的提升，验证了掩码向量与密集梯度嵌入的有效性。单独使用几何 - 文本提示（GeoText Prompts）可实现约 4.4% 的显著提升，凸显融合语义与几何线索的互补优势。

#### 显著的注意力聚焦

图 5 进一步验证，几何 - 文本提示能引导模型关注几何敏感区域，而粗掩码先验嵌入则增强前景掩码的梯度；与基准模型相比，Vireo 对场景结构与语义边界的聚焦更精准，这也解释了其在跨域与开放词汇场景中持续保持优势的原因。

  

![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27400%27%20height=%27256%27/%3e)![图5：不同场景下注意力图与亲和图的可视化（CMPE表示粗掩码先验嵌入，GTP表示几何-文本提示）](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjI1NiIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==)![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27400%27%20height=%27256%27/%3e)![image](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjI1NiIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==)

  

![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27400%27%20height=%27256%27/%3e)![图6：特征空间的t-SNE嵌入可视化（左：原始源域数据集；右：Vireo在Cityscapes→ACDC+BDD100K+Mapillary适应后的结果；每个点按语义类别着色）](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjI1NiIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==)![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27400%27%20height=%27256%27/%3e)![image](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjI1NiIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==)

  

![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27400%27%20height=%27256%27/%3e)![图7：基准模型与含粗掩码先验嵌入（CMPE）模型的训练损失对比（a）分类损失；（b）掩码损失](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjI1NiIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==)![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27400%27%20height=%27256%27/%3e)![image](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjI1NiIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==)

  

表 5：Cityscapes→ACDC（下雪、夜晚、大雾、下雨）+BDD100K+Mapillary 泛化设置下，Vireo 各组件配置的消融实验（前三名结果分别标注为最佳、第二、第三，单位：%）

  

|配置|下雪 - ACDC|夜晚 - ACDC|大雾 - ACDC|下雨 - ACDC|BDD100K|Mapillary|
|---|---|---|---|---|---|---|
|REIN [35]|70.6|55.9|79.5|72.5|63.5|74.0|
|+ 拼接深度特征|70.8|56.1|79.4|72.8|63.6|74.2|
|+Depth Anything 提示|70.4|55.5|79.7|72.0|63.8|73.9|
|+Depth Anything V2|71.5|56.7|80.5|73.3|64.4|74.5|
|+DA+AO（深度增强 + 注意力优化）|72.2|57.4|80.9|74.2|65.1|75.0|
|+DOV-VEH（域 - 开放词汇向量嵌入头）|70.9|56.2|79.8|72.8|63.7|74.2|
|+CMPE（粗掩码先验嵌入）|71.6|56.9|80.2|73.4|64.1|74.6|
|+GeoText Prompts（几何 - 文本提示）|74.0|58.4|81.1|74.8|65.3|75.3|
|Vireo（本文方法）|76.2|60.6|82.3|76.3|66.7|76.0|

#### 语义分割域泛化（DGSS）特征的 t-SNE 可视化

图 6 可视化了原始数据集与本文方法的特征分布，结果显示本文方法学习到的特征在形成语义分离聚类方面具有优势，证明所提域泛化视觉 - 文本对齐方法在构建开放词汇语义空间中的有效性。

## 5 结论

本文提出 Vireo—— 首个整合开放词汇识别与域泛化语义分割的单阶段框架。通过整合冻结视觉基础模型、深度感知几何信息及三个核心模块（几何 - 文本提示、粗掩码先验嵌入、域 - 开放词汇向量嵌入头），Vireo 收敛速度更快，能聚焦于场景结构，且在多个基准测试中超越现有主流方法，证明了文本线索与几何先验融合对实现稳健像素级感知的重要作用。

### 局限性

Vireo 假设存在可靠的 RGB 相机，但在少数情况下（如遮挡、眩光、硬件故障），RGB 图像流可能丢失，从而影响感知性能。未来研究将探索多源感知设置 —— 当 RGB 相机失效时，自动切换至激光雷达、雷达或事件相机，以确保在所有条件下均能保持稳健的分割性能。

## 参考文献

（注：参考文献作者、年份及会议 / 期刊名称保留原文格式，标题按原文含义翻译，确保学术严谨性） [1] F. Liang, B. Wu, X. Dai, K. Li, Y. Zhao, H. Zhang, P. Zhang, P. Vajda, and D. Marculescu. 2023. 基于掩码自适应 CLIP 的开放词汇语义分割. 《IEEE/CVF 计算机视觉与模式识别会议论文集》，第 7061-7070 页. [2] J. Qin, J. Wu, P. Yan, M. Li, R. Yuxi, X. Xiao, Y. Wang, R. Wang, S. Wen, X. Pan 等. 2023. FreeSeg：统一、通用的开放词汇图像分割方法. 《IEEE/CVF 计算机视觉与模式识别会议论文集》，第 19446-19455 页. [3] B. Xie, J. Cao, J. Xie, F. S. Khan, and Y. Pang. 2024. SED：面向开放词汇语义分割的简单编码器 - 解码器. 《IEEE/CVF 计算机视觉与模式识别会议论文集》，第 3426-3436 页. [4] C. Han, Y. Zhong, D. Li, K. Han, and L. Ma. 2023. 基于解耦单通道网络的开放词汇语义分割. 《IEEE/CVF 国际计算机视觉会议论文集》，第 1086-1096 页. [5] Y. Benigmim, S. Roy, S. Essid, V. Kalogeiton, and S. Lathuilière. 2024. 基于基础模型协作的域泛化语义分割. 《IEEE/CVF 计算机视觉与模式识别会议论文集》，第 3108-3119 页. [6] Z. Wei, L. Chen, Y. Jin, X. Ma, T. Liu, P. Ling, B. Wang, H. Chen, and J. Zheng. 2024. 更强、更精简、更卓越：利用视觉基础模型实现域泛化语义分割. 《IEEE/CVF 计算机视觉与模式识别会议论文集》，第 28619-28630 页. [7] J. Niemeijer, M. Schwonberg, J.-A. Termöhlen, N. M. Schmidt, and T. Fingscheidt. 2024. 基于适应的泛化：面向域泛化语义分割的扩散式域扩展方法. 《IEEE/CVF 冬季计算机视觉应用会议论文集》，第 2830-2840 页. [8] Q. Bi, J. Yi, H. Zheng, H. Zhan, Y. Huang, W. Ji, Y. Li, and Y. Zheng. 2024. 面向域泛化语义分割的频率自适应视觉基础模型学习. 《神经信息处理系统进展》，第 37 卷：第 94047-94072 页. [9] S. Choi, S. Jung, H. Yun, J. T. Kim, S. Kim, and J. Choo. 2021. RobustNet：通过实例选择性白化提升城市场景分割的域泛化能力. 《计算机视觉与模式识别会议论文集》，第 11580-11590 页. [10] S. Cho, H. Shin, S. Hong, A. Arnab, P. H. Seo, and S. Kim. 2024. CAT-Seg：面向开放词汇语义分割的代价聚合方法. 《IEEE/CVF 计算机视觉与模式识别会议论文集》，第 4113-4123 页. [11] H. Luo, J. Bao, Y. Wu, X. He, and T. Li. 2023. SegCLIP：基于可学习中心的补丁聚合开放词汇语义分割. 《国际机器学习大会论文集》，PMLR 出版社，第 23033-23044 页. [12] Z. Wu, X. Wu, X. Zhang, L. Ju, and S. Wang. 2022. Siamdoge：基于孪生网络的域泛化语义分割. 《欧洲计算机视觉会议论文集》，施普林格出版社，第 603-620 页. [13] D. Peng, Y. Lei, M. Hayat, Y. Guo, and W. Li. 2022. 语义感知的域泛化分割. 《IEEE/CVF 计算机视觉与模式识别会议论文集》，第 2594-2605 页. [14] M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, and A. Joulin. 2021. 自监督视觉 Transformer 的涌现特性. 《IEEE/CVF 国际计算机视觉会议论文集》，第 9650-9660 页. [15] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby 等. 2023. DINOv2：无监督学习稳健视觉特征。预印本 arXiv:2304.07193. [16] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark 等. 2021. 基于自然语言监督的可迁移视觉模型学习. 《国际机器学习大会论文集》，PMLR 出版社，第 8748-8763 页. [17] L. Yang, B. Kang, Z. Huang, X. Xu, J. Feng, and H. Zhao. 2024. Depth Anything：释放大规模无标签数据的潜力. 《IEEE/CVF 计算机视觉与模式识别会议论文集》，第 10371-10381 页. [18] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao. 2024. Depth Anything V2. 《神经信息处理系统进展》，第 37 卷：第 21875-21911 页. [19] S. Chen, T. Han, C. Zhang, J. Su, R. Wang, Y. Chen, Z. Wang, and G. Cai. 2025. HSPFormer：面向语义分割的层次化空间感知 Transformer. 《IEEE 智能交通系统汇刊》. [20] T. Han, S. Chen, C. Li, Z. Wang, J. Su, M. Huang, and G. Cai. 2024. Epurate-net：面向交通环境城市道路检测的高效渐进式不确定性优化分析. 《IEEE 智能交通系统汇刊》. [21] S. Chen, T. Han, C. Zhang, X. Luo, M. Wu, G. Cai, and J. Su. 2025. 更强、更稳定、更卓越：深度视觉基础模型中的几何一致性实现域泛化语义分割。预印本 arXiv:2504.12753. [22] B. Li, K. Q. Weinberger, S. Belongie, V. Koltun, and R. Ranftl. 2022. 语言驱动的语义分割。预印本 arXiv:2201.03546. [23] C. Zhou, C. C. Loy, and B. Dai. 2022. 从 CLIP 中提取免费密集标签. 《欧洲计算机视觉会议论文集》，施普林格出版社，第 696-712 页. [24] J. Xu, S. Liu, A. Vahdat, W. Byeon, X. Wang, and S. De Mello. 2023. 基于文本 - 图像扩散模型的开放词汇全景分割. 《IEEE/CVF 计算机视觉与模式识别会议论文集》，第 2955-2966 页. [25] J. Hoffman, E. Tzeng, T. Park, J.-Y. Zhu, P. Isola, K. Saenko, A. Efros, and T. Darrell. 2018. CyCADA：循环一致对抗域适应. 《国际机器学习大会论文集》，PMLR 出版社，第 1989-1998 页. [26] R. Volpi, H. Namkoong, O. Sener, J. C. Duchi, V. Murino, and S. Savarese. 2018. 基于对抗数据增强的域外泛化. 《神经信息处理系统进展》，第 31 卷. [27] Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, M. March, and V. Lempitsky. 2016. 神经网络的域对抗训练. 《机器学习研究期刊》，第 17 卷（第 59 期）：第 1-35 页. [28] S. Choi, S. Jung, H. Yun, J. T. Kim, S. Kim, and J. Choo. 2021. RobustNet：通过实例选择性白化提升城市场景分割的域泛化能力. 《计算机视觉与模式识别会议论文集》，第 11580-11590 页. [29] Q. Yu, J. He, X. Deng, X. Shen, and L.-C. Chen. 2023. 卷积永不落幕：基于单冻结卷积 CLIP 的开放词汇分割. 《神经信息处理系统进展》，第 36 卷：第 32215-32234 页. [30] X. Shan, D. Wu, G. Zhu, Y. Shao, N. Sang, and C. Gao. 2024. 基于图像嵌入平衡的开放词汇语义分割. 《IEEE/CVF 计算机视觉与模式识别会议论文集》，第 28412-28421 页. [31] X. Pan, P. Luo, J. Shi, and X. Tang. 2018. 一举两得：通过 IBN-Net 增强学习与泛化能力. 《欧洲计算机视觉会议论文集》，第 464-479 页. [32] S. Lee, H. Seong, S. Lee, and E. Kim. 2022. WildNet：从野外场景学习域泛化语义分割. 《计算机视觉与模式识别会议论文集》，第 9936-9946 页. [33] J. Ding, N. Xue, G.-S. Xia, B. Schiele, and D. Dai. 2023. HGFormer：面向域泛化语义分割的层次化分组 Transformer. 《计算机视觉与模式识别会议论文集》，第 15413-15423 页. [34] Q. Bi, S. You, and T. Gevers. 2024. 学习内容增强掩码 Transformer 实现域泛化城市场景分割. 《美国人工智能协会会议论文集》，第 38 卷（第 2 期）：第 819-827 页. [35] Z. Wei, L. Chen, Y. Jin, X. Ma, T. Liu, P. Ling, B. Wang, H. Chen, and J. Zheng. 2024. 更强、更精简、更卓越：利用视觉基础模型实现域泛化语义分割. 《计算机视觉与模式识别会议论文集》，第 28619-28630 页. [36] Q. Bi, J. Yi, H. Zheng, H. Zhan, Y. Huang, W. Ji, Y. Li, and Y. Zheng. 2024. 面向域泛化语义分割的频率自适应视觉基础模型学习. 《神经信息处理系统进展》，第 37 卷：第 94047-94072 页. [37] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele. 2016. Cityscapes 数据集：面向城市场景语义理解. 《计算机视觉与模式识别会议论文集》，第 3213-3223 页. [38] F. Yu, H. Chen, X. Wang, W. Xian, Y. Chen, F. Liu, V. Madhavan, and T. Darrell. 2020. BDD100K：面向异构多任务学习的多样化驾驶数据集. 《计算机视觉与模式识别会议论文集》，第 2636-2645 页. [39] G. Neuhold, T. Ollmann, S. Rota Bulo, and P. Kontschieder. 2017. Mapillary Vistas 数据集：面向街道场景语义理解. 《国际计算机视觉会议论文集》，第 4990-4999 页. [40] C. Sakaridis, D. Dai, and L. Van Gool. 2021. ACDC：面向语义驾驶场景理解的恶劣条件对应数据集. 《国际计算机视觉会议论文集》，第 10745-10755 页. [41] B. Zhou, H. Zhao, X. Puig, T. Xiao, S. Fidler, A. Barriuso, and A. Torralba. 2019. 基于 ADE20K 数据集的场景语义理解. 《国际计算机视觉期刊》，第 127 卷（第 3 期）：第 302-321 页. [42] S. R. Richter, V. Vineet, S. Roth, and V. Koltun. 2016. 为数据而 “玩”：从电子游戏中获取真值. 《欧洲计算机视觉会议论文集》，施普林格出版社，第 102-118 页. [43] J. Zhang, R. Liu, H. Shi, K. Yang, S. Reiß, K. Peng, H. Fu, K. Wang, and R. Stiefelhagen. 2023. 实现任意模态语义分割. 《计算机视觉与模式识别会议论文集》. [44] M. Contributors. 2020. MMSegmentation：OpenMMLab 语义分割工具包与基准测试. [https://github.com/open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation). [45] Y. Fang, W. Wang, B. Xie, Q. Sun, L. Wu, X. Wang, T. Huang, X. Wang, and Y. Cao. 2023. EVA：探索大规模掩码视觉表示学习的极限. 《计算机视觉与模式识别会议论文集》，第 19358-19369 页. [46] Y. Fang, Q. Sun, X. Wang, T. Huang, X. Wang, and Y. Cao. 2023. EVA-02：面向新一代视觉表示。预印本 arXiv:2303.11331. [47] Z. Zhong, Y. Zhao, G. H. Lee, and N. Sebe. 2022. 面向域泛化城市场景分割的对抗风格增强. 《神经信息处理系统进展》，第 35 卷：第 338-350 页. [48] P. Chattopadhyay, K. Sarangmath, V. Vijaykumar, and J. Hoffman. 2023. PASTA：面向合成到真实域泛化的比例振幅谱训练增强. 《国际计算机视觉会议论文集》，第 19288-19300 页. [49] D. Peng, Y. Lei, L. Liu, P. Zhang, and J. Liu. 2021. 面向合成到真实语义分割的全局与局部纹理随机化. 《IEEE 图像处理汇刊》，第 30 卷：第 6594-6608 页. [50] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. 2021. LoRA：大型语言模型的低秩适应。预印本 arXiv:2106.09685. [51] S. Chen, C. Ge, Z. Tong, J. Wang, Y. Song, J. Wang, and P. Luo. 2022. AdaptFormer：适配视觉 Transformer 实现可扩展视觉识别. 《神经信息处理系统进展》，第 35 卷：第 16664-16678 页. [52] M. Jia, L. Tang, B.-C. Chen, C. Cardie, S. Belongie, B. Hariharan, and S.-N. Lim. 2022. 视觉提示微调. 《欧洲计算机视觉会议论文集》，施普林格出版社，第 709-727 页. [53] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby 等. 2023. DINOv2：无监督学习稳健视觉特征。预印本 arXiv:2304.07193. [54] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark 等. 2021. 基于自然语言监督的可迁移视觉模型学习. 《国际机器学习大会论文集》，PMLR 出版社，第 8748-8763 页. [55] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo 等. 2023. Segment Anything. 《国际计算机视觉会议论文集》，第 4015-4026 页. [56] M. Xu, Z. Zhang, F. Wei, Y. Lin, Y. Cao, H. Hu, and X. Bai. 2022. 基于预训练视觉语言模型的开放词汇语义分割简单基准. 《欧洲计算机视觉会议论文集》，施普林格出版社，第 736-753 页. [57] G. Ghiasi, X. Gu, Y. Cui, and T.-Y. Lin. 2022. 基于图像级标签的开放词汇图像分割扩展. 《欧洲计算机视觉会议论文集》，施普林格出版社，第 540-557 页. [58] X. Zou, Z.-Y. Dou, J. Yang, Z. Gan, L. Li, C. Li, X. Dai, H. Behl, J. Wang, L. Yuan 等. 2023. 面向像素、图像与语言的通用解码. 《IEEE/CVF 计算机视觉与模式识别会议论文集》，第 15116-15127 页. [59] J. Xu, S. De Mello, S. Liu, W. Byeon, T. Breuel, J. Kautz, and X. Wang. 2022. GroupViT：语义分割从文本监督中涌现. 《IEEE/CVF 计算机视觉与模式识别会议论文集》，第 18134-18144 页. [60] H. Zhang, F. Li, X. Zou, S. Liu, C. Li, J. Yang, and L. Zhang. 2023. 面向开放词汇分割与检测的简单框架. 《IEEE/CVF 国际计算机视觉会议论文集》，第 1020-1031 页. [61] Q. Huang, H. Hu, and J. Jiao. 2025. 重新审视开放词汇分割的 “开放” 本质. 《第十三届学习表示国际会议论文集》. [62] D. Li, J. Yang, K. Kreis, A. Torralba, and S. Fidler. 2021. 基于生成模型的语义分割：半监督学习与强域外泛化. 《IEEE/CVF 计算机视觉与模式识别会议论文集》，第 8300-8311 页. [63] W.-J. Ahn, G.-Y. Yang, H.-D. Choi, and M.-T. Lim. 2024. 基于协方差对齐与语义一致性对比学习的风格无关域泛化语义分割. 《IEEE/CVF 计算机视觉与模式识别会议论文集》，第 3616-3626 页. [64] H. Niu, L. Xie, J. Lin, and S. Zhang. 2025. 探索语义一致性与风格多样性实现域泛化语义分割. 《美国人工智能协会会议论文集》，第 39 卷（第 6 期）：第 6245-6253