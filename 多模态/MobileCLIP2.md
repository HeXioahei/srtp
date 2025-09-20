# MobileCLIP2：改进多模态强化训练

法尔塔什・法赫里 *、帕万・库马尔・阿纳索萨卢・瓦苏 * 、 杰姆・科奇、瓦伊沙尔・尚卡尔（曾任职于苹果公司）、亚历山大・托舍夫、翁塞尔・图泽尔、哈迪・普兰萨里（苹果公司）电子邮箱：fartash@apple.com、panasosaluvasu@apple.com、cem_koc@apple.com、toshev@apple.com、otuzel@apple.com、mpouransari@apple.com

OpenReview 评审链接：[https://openreview.net/forum?forum?id=WeF9zolng8](https://openreview.net/forum?forum?id=WeF9zolng8)

## 摘要

CLIP 等具备零样本能力的图文基础模型可支持各类应用场景。MobileCLIP 是近期推出的图文模型家族，==延迟仅 3-15 毫秒、参数规模 5000 万 - 1.5 亿==，且零样本准确率达到当前最优水平。MobileCLIP 的核心优势在于低延迟轻量架构，以及创新的多模态强化训练方法 —— 该方法使来自多个描述生成器和 CLIP 教师模型的知识蒸馏更高效、可扩展且可复现。本文通过以下方式改进 MobileCLIP 的多模态强化训练：
- 1）采用在 DFN 数据集上训练的更优 CLIP 教师集成模型；
- 2）使用在 DFN 数据集上预训练、并在精选高质量图像描述数据集上微调的改进型描述生成教师模型。
通过消融实验，我们得出多项新见解：对比知识蒸馏中==温度调优的重要性==、==描述生成器微调==对提升描述多样性的有效性，以及==融合多模型生成的合成描述==可带来额外性能提升。基于此，我们训练出全新模型家族**MobileCLIP2**，在低延迟条件下实现 ImageNet-1k 零样本准确率的当前最优。具体而言，MobileCLIP2-B 相较于 MobileCLIP-B 架构，ImageNet-1k 准确率提升 2.2%；值得注意的是，MobileCLIP2-S4 的 ImageNet-1k 零样本准确率与 SigLIP-SO400M/14 持平，但参数规模仅为后者的 1/2；同时，其性能优于 DFN ViT-L/14，延迟却降低 2.5 倍。我们已开源预训练模型 ¹ 和数据生成代码 ²，其中数据生成代码支持通过分布式可扩展处理，基于任意教师模型轻松构建新的强化数据集。

* 贡献均等。¹[https://github.com/apple/ml-mobileclip](https://github.com/apple/ml-mobileclip)²[https://github.com/apple/ml-mobileclip-dr](https://github.com/apple/ml-mobileclip-dr)

## 1. 引言

CLIP（Radford 等人，2021）是一种图文模型，可将图像和文本输入映射到共享嵌入空间：描述某图像的文本（即描述）会与该图像的嵌入向量相近，而与无关图像的嵌入向量远离。在大量相关研究（Frome 等人，2013；Socher 等人，2014；Karpathy 与 Fei-Fei，2015；Kiros 等人，2014；Faghri 等人，2018）的基础上，CLIP 大幅提升了训练数据与模型的规模。随之而来的是，除图文检索性能提升外，模型还具备了全新的零样本分类能力 —— 无需借助分类标签进行显式监督训练，仅通过线性探测即可在分类任务上实现可观准确率。此外，通过固定图像编码器进行线性探测，或对编码器进行全微调，可使其适配新任务，进而在各类任务中实现当前最优性能（Wortsman 等人，2022）。凭借多样化的能力与应用场景，CLIP 成为首批被称为 “基础模型” 的模型之一（Bommasani 等人，2021）。

CLIP 的成功推动了模型与数据集规模的持续扩大，性能也随之逐步提升（Fang 等人，2024b；Zhai 等人，2023；Gadre 等人，2023；Fang 等人，2024a）。近年来，这一趋势逐渐转向**小型低延迟模型**，以适配移动设备应用场景。其中，TinyCLIP（Wu 等人，2023）与 MobileCLIP（Vasu 等人，2024c）推出的模型参数总量最低仅 5000 万（图像与文本编码器参数之和）。例如，MobileCLIP-S0 的总延迟（图像与文本编码器延迟之和）仅 3 毫秒，平均性能与原始 OpenAI ViT-B/16 CLIP 相当，但参数规模缩小 3 倍、速度提升 5 倍，且性能优于 SigLIP（Zhai 等人，2023）等此前的大尺寸最优模型。

本文通过消融实验深入分析多模态强化训练，并提出改进的训练方案，最终训练出全新模型家族 MobileCLIP2。该家族在一系列延迟区间内实现 ImageNet-1k 准确率的当前最优，性能可与 SigLIP（Zhai 等人，2023）、DFN（Fang 等人，2024a）等大尺寸模型持平，同时参数规模最高缩小 4 倍（如 MobileCLIP2-S2 相较于 SigLIP2-B/32），速度最高提升 2.5 倍（如 MobileCLIP2-S4 相较于 DFN ViT-L/14）。此外，我们开源了高效分布式代码，支持基于任意教师模型生成强化数据集。

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202509202117630.png)

## 2. 改进的训练方法

MobileCLIP 推出了一系列低延迟图文模型，包括 S0、S1、S2、B、B-LT 变体，图文总延迟覆盖 3.8-13.7 毫秒。这些低延迟得益于基于 FastViT（Vasu 等人，2023b）的专用架构，以及名为 “多模态强化训练” 的改进型训练方法。本文将进一步探索并优化多模态强化训练的每个环节，同时扩展架构家族以覆盖更广泛的延迟区间。

强化训练是一种通过引入额外来源（如预训练模型）的信息改进基础数据集，从而提升模型性能的方法（Faghri 等人，2023）。Vasu 等人（2024c）提出的多模态强化训练，会从预训练图文模型与预训练合成描述生成器中提取信息，添加到图文数据集中。具体而言，他们为 DataComp-1B 数据集补充了以下信息：
- 1）两个 CLIP 教师模型对每张图像的 10 次随机增强版本生成的图像嵌入；
- 2）两个 CLIP 教师模型对原始文本及 CoCa 描述生成器生成的 5 个合成描述生成的文本嵌入。
对于强化数据集，他们通过引入基于教师模型嵌入的知识蒸馏损失（Hinton 等人，2015），修改训练损失函数。为确保教师与学生模型的一致性，通过存储的增强参数复现相同的图像增强（Beyer 等人，2022；Faghri 等人，2023）。他们还通过消融实验确定了最优的 CLIP 教师模型、描述生成器与图像增强方案，以在 ImageNet 及 DataComp 的 38 项评估任务（Gadre 等人，2023）中实现最大性能提升。

本文沿用与 MobileCLIP 类似的多模态强化训练框架，但对各环节均进行优化，最终形成的模型家族命名为 MobileCLIP2。表 1 总结了各项主要改进带来的性能提升：简言之，与 MobileCLIP 相比，我们==采用了更优的训练数据、CLIP 教师模型，以及更优且更多样的合成描述生成器==。所有消融实验中，我们均在含 1280 万张图像的数据集上，对 MobileCLIP-B 训练 3 万次迭代（约 20 轮）。本文使用的数据集汇总详见表 15。

**表 1：MobileCLIP2 训练改进汇总**CoCa 模型先在大型数据集上预训练（见过 130 亿样本），再进行 1200 万样本的微调（以 “￫” 表示）。表中所有 CLIP 教师模型的架构均为 ViT-L/14。若有数据，均报告 5 次实验的均值与标准差。

|名称|数据集|CLIP 教师模型训练数据集|CoCa 模型训练数据集|ImageNet 验证集（IN-val）准确率（%）|Flickr30k 检索率（%）|38 项任务平均性能（%）|
|---|---|---|---|---|---|---|
|MobileCLIP（Vasu 等人，2024c）|DataComp-1B12M|OpenAI + DataComp-XL|LAION-2B ￫ MSCOCO-123k|61.6|72.8|53.5|
|表 2 结果|DFN-5B12M|OpenAI + DataComp-XL|LAION-2B ￫ MSCOCO-123k|63.1±0.2|73.3±0.6|54.1±0.4|
|表 4 结果|DFN-5B12M|DFN-2B + DFN-2B-s39B|LAION-2B ￫ MSCOCO-123k|65.4±0.4|75.8±0.3|56.2±0.6|
|MobileCLIP2（表 6）|DFN-5B12M|DFN-2B + DFN-2B-s39B|DFN-2B ￫ MSCOCO-38k|65.9±0.3|75.4±0.2|56.5±0.3|
|表 6 结果|DFN-5B12M|DFN-2B + DFN-2B-s39B|DFN-2B ￫ 10 个合成描述|66.0±0.1|75.1±0.6|56.5±0.3|

图 2 对比了训练过程中，DFN（Fang 等人，2024a）、DataComp（Gadre 等人，2023）与 DataCompDR（Vasu 等人，2024c）数据集的效率差异：在 DFNDR-2B12M 上训练 3000 万样本，效率是 DataComp-1B12M 的 5 倍 —— 即仅需 DFNDR-2B12M 的 600 万样本，即可达到 DataComp-1B12M 训练 3000 万样本的 ImageNet-1k 零样本准确率。类似地，DataCompDR-12M 的训练效率是 DFN-2B12M 的 3.3 倍、DataCompDR-12M 的 1.3 倍。在见过 130 亿样本的训练中，DFNDR-2B 的训练速度比 DataCompDR-1B 快 1.6 倍。与 DataCompDR 类似，在 DFNDR 数据集上训练无任何 wall-clock 时间开销 —— 即 DataComp、DFN、DataCompDR、DFNDR 的每次训练步骤耗时相同。这意味着，样本数量与训练迭代次数上的效率提升，可直接转化为 wall-clock 时间效率的提升。*（Wall‑clock 时间指从真实世界的时钟（墙上挂的钟）上测量的实际经过时间，即从某一时刻到另一时刻的真实流逝时间。在计算机系统中，它用于衡量程序、任务或整个系统的总耗时，包括 CPU 计算、I/O 等待、调度延迟等所有因素。与仅计 CPU 核心占用的 CPU 时间不同，wall‑clock 时间反映了用户感知到的整体运行时长。）*

### 2.1 多模态强化训练

数据集强化（DR）（Faghri 等人，2023）是一种在最小化训练代码修改与计算开销的前提下，通过改进数据集提升准确率的方法。DR 最初用于图像分类器训练：Faghri 等人（2023）通过存储强分类器集成模型的分类概率，改进 ImageNet 数据集。==基于存储的概率，训练本质上是知识蒸馏（Hinton 等人，2015），且无需额外计算教师模型预测，成本效率极高，因此可通过延长训练实现更大性能提升（Beyer 等人，2022）==。Vasu 等人（2024c）将 DR 应用于 CLIP 图文模型训练，通过存储 CLIP 强集成模型的知识与图像描述生成器的合成描述，实现了较非强化 CLIP 训练高达 1000 倍的学习效率提升。

给定包含b个图文对的批次，目标学生模型的嵌入表示为 $\Phi_{img}, \Phi_{txt} \in \mathbb{R}^{b \times d}$ （d为共享嵌入空间维度）。我们使用两类教师模型：
- 1）图文教师集成模型，可将图像与文本映射到类似 CLIP 的共享空间（Radford 等人，2021）；
- 2）基于 CoCa 的描述生成器，采用编码器 - 解码器架构，可根据图像生成描述（Yu 等人，2022）。
设 $\Psi_{img}^{(k)}, \Psi_{txt}^{(k)} \in \mathbb{R}^{b \times d_k}$ 为第k个 CLIP 类教师模型的图文嵌入（$d_k$ 为该教师模型共享空间维度），蒸馏损失定义为：

$$\mathcal{L}_{Distill} = \frac{1}{2bK} \sum_{k=1}^K \underbrace{KL\left(\mathcal{S}_{\tau_k}(\Psi_{img}^{(k)}, \Psi_{txt}^{(k)}) \parallel \mathcal{S}_{\hat{\tau}}(\Phi_{img}, \Phi_{txt})\right)}_{图像到文本} + \underbrace{KL\left(\mathcal{S}_{\tau_k}(\Psi_{txt}^{(k)}, \Psi_{img}^{(k)}) \parallel \mathcal{S}_{\hat{\tau}}(\Phi_{txt}, \Phi_{img})\right)}_{文本到图像} \tag{1}$$

其中，KL表示 KL 散度（Kullback-Leibler divergence），$\mathcal{S}_{\tau}(U, V)$ 表示对 $UV^\top/\tau$ 执行行级 Softmax 运算（$\tau$ 为温度参数）。总损失为 $\mathcal{L}_{Total} = (1-\lambda)\mathcal{L}_{CLIP} + \lambda\mathcal{L}_{Distill}$ ，即标准 CLIP 损失与蒸馏损失的加权和，权重分别为 $1-\lambda$ 与 $\lambda$ 。

### 2.2 更优的基础数据集：DFN

多模态强化训练始于包含真实图文对的基础数据集，这类数据集通常来自网页。DataComp（Gadre 等人，2023）表明，通过基于图文兼容性等分数的过滤，可显著提升大规模图文数据集的质量：其在 120 亿样本池上应用 BestPool 过滤，得到 DataComp-1B 数据集，并作为 MobileCLIP 的基础数据集；同时，DataComp 还公开了原始 120 亿样本，作为数据集筛选方法的基准。DFN（Fang 等人，2024a）提出使用在高质量数据上训练的过滤网络筛选数据：将该模型应用于 DataComp-12B 样本池，得到 DFN-2B 数据集；此外，他们还从网页收集了与 DataComp-12B 无重叠的 30 亿图像，筛选后与 DFN-2B 合并，形成 DFN-5B 数据集。

本文研究用 DFN-5B 替代 MobileCLIP 中基础数据集的效果。消融实验使用 Vasu 等人（2024c）提出的 DataComp-1B 的 1200 万均匀采样子集（DataComp-1B12M，用于快速实验），并类似地从 DFN-5B 中采样 1200 万样本，得到 DFN-5B12M。表 2 对比了有无蒸馏 / 合成描述时的训练性能：结果显示，结合蒸馏与合成描述后，DFN-5B12M 较 DataComp-1B12M 最多提升 1.4%；尽管这一提升小于无蒸馏 / 合成描述时的 6%，但仍超过标准差范围，具有统计显著性。

**表 2：无论有无蒸馏 / 合成描述，基于 DFN 的训练均优于 DataComp**CLIP 教师模型与合成描述生成器均与 MobileCLIP 一致（OpenAI+DataCompXL CLIP-ViT-L/14、CoCa-ViT-L/14）。蒸馏时，系数\(\lambda\)设为 1.0（无 CLIP 损失），并使用强图像增强。

|数据集|蒸馏（Distill.）|合成描述（Syn. Caps.）|IN-val 准确率（%）|Flickr30k 检索率（%）|38 项任务平均性能（%）|
|---|---|---|---|---|---|
|DataComp-1B12M|否|否|44.6|42.4|40.1|
|DFN-5B12M|否|否|49.9|48.5|43.5|
|DataComp-1B12M|否|是|51.9|71.8|47.8|
|DFN-5B12M|否|是|54.9|70.7|49.6|
|DataComp-1B12M|是|否|56.3|57.8|48.7|
|DFN-5B12M|是|否|59.5|60.4|50.0|
|DataComp-1B12M|是|是|61.6|72.8|53.7|
|DFN-5B12M|是|是|63.0|74.1|54.6|

### 2.3 DFN CLIP 教师模型

多模态强化训练的强化信息来源之一，是作为 CLIP 蒸馏目标的 CLIP 教师模型嵌入。Vasu 等人（2024c）在其发表时对当时所有强 CLIP 教师模型进行了全面研究，发现 ViT-L-14-openai 与 ViT-L-14-datacomp_xl_s13b_b90k 的集成可使学生模型性能最优。本文探究在 DFN 上预训练的模型作为教师的有效性：基于 DFN 预训练的 CLIP 模型（ViT-L-14 与 ViT-H-14 架构）在 DataComp 的 38 项评估任务（Fang 等人，2024a）中表现最优，性能超过 SigLIP（Zhai 等人，2023）等主流模型。

由于描述生成器与 CLIP 教师模型的选择可能相互影响，为简化分析，我们先固定描述生成器为未微调的 CoCa 模型（见 2.4 节），仅分析 CLIP 教师模型的影响；描述生成器微调对合成描述多样性的影响将在 2.5 节探讨。

#### Logit 缩放

CLIP 模型训练时会调优 logit 缩放（范围 0-100），MobileCLIP 将 logit 缩放与 KD 损失中的温度缩放设为相同值。本文发现，DFN 与 DataComp 模型的 logit 缩放并非 KD 的最优值，需进一步调优。表 3 展示了训练 MobileCLIP-B 时各教师模型的最优 logit 缩放：结果表明，logit 缩放并非敏感超参数 —— 在 5 个单位范围内取值，性能均相近。

**表 3：不同教师模型的最优 logit 缩放存在差异**数据集为 DFN-5B12M，合成描述由 2.4 节的 CoCa-DFN-2B 生成。损失系数\(\lambda\)设为 1.0，训练使用强图像增强。

|教师模型|Logit 缩放|IN-val 准确率（%）|Flickr30k 检索率（%）|38 项任务平均性能（%）|
|---|---|---|---|---|
|datacomp_xl_s13b_b90k-CLIP-ViT-L-14|50|62.6|65.6|53.3|
|DFN2B-CLIP-ViT-L-14|70|65.5|68.0|56.5|
|DFN5B-CLIP-ViT-H-14|90|64.0|65.9|54.7|
|DFN5B-CLIP-ViT-H-14-384|55|64.6|67.6|54.4|
|DFN2B-CLIP-ViT-L-14-s39b|60|65.2|67.5|54.8|

#### 教师集成

我们使用 DataComp 与 DFN 教师模型构建规模为 2 的集成模型。表 4 展示了基于不同集成嵌入训练 MobileCLIP-B 的性能：结果表明，相较于 MobileCLIP 使用的教师模型，性能显著提升 ——IN-val 与 Flickr30k 最多提升 3%。综合性能与成本效率（相较于更大尺寸或更高分辨率的集成），MobileCLIP2 选择 DFN2B-CLIP-ViT-L-14-s39b 与 DFN2B-CLIP-ViT-L-14 的集成作为教师模型。每个集成成员的最优 logit 缩放均独立确定；尽管集成时联合调优可能进一步优化，但本文未开展此项工作。

**表 4：DFN CLIP 教师集成使 ImageNet-1k 验证集准确率提升 2.8%**数据集为 DFN-5B12M，合成描述由 2.4 节的 CoCa-DFN-2B 生成。损失系数\(\lambda\)设为 1.0，训练使用强图像增强。各模型的最优 logit 缩放基于表 3 独立设置。

|教师模型 1|教师模型 2|IN-val 准确率（%）|Flickr30k 检索率（%）|38 项任务平均性能（%）|
|---|---|---|---|---|
|ViT-L-14-openai|ViT-L-14-datacomp_xl_s13b_b90k|63.1|64.7|55.2|
|ViT-L-14-datacomp_xl_s13b_b90k|DFN5B-CLIP-ViT-H-14-384|64.5|67.8|54.5|
|ViT-L-14-datacomp_xl_s13b_b90k|DFN5B-CLIP-ViT-H-14|64.4|67.3|55.3|
|ViT-L-14-datacomp_xl_s13b_b90k|DFN2B-CLIP-ViT-L-14|65.3|68.1|56.2|
|DFN5B-CLIP-ViT-H-14-384|DFN5B-CLIP-ViT-H-14|64.7|66.9|54.9|
|DFN5B-CLIP-ViT-H-14-384|DFN2B-CLIP-ViT-L-14|65.8|68.6|56.2|
|DFN5B-CLIP-ViT-H-14|DFN2B-CLIP-ViT-L-14|65.2|68.0|55.8|
|DFN2B-CLIP-ViT-L-14-s39b|datacomp_xl_s13b_b90k|65.1|67.6|55.7|
|DFN2B-CLIP-ViT-L-14-s39b|DFN5B-CLIP-ViT-H-14-384|65.7|67.3|55.1|
|DFN2B-CLIP-ViT-L-14-s39b|DFN5B-CLIP-ViT-H-14|65.7|68.2|55.7|
|DFN2B-CLIP-ViT-L-14-s39b|DFN2B-CLIP-ViT-L-14|65.9|68.7|55.9|

### 2.4 DFN 描述生成器

MobileCLIP2 训练的另一强化信息来源，是图像描述生成器生成的合成描述。MobileCLIP 使用单一 CoCa 描述生成器 —— 该模型采用双塔图文架构，搭配文本解码器（Yu 等人，2022）。与最新视觉语言模型（VLM）相比，其文本解码器更轻量，因此整体速度快于多数最新 VLM（Liu 等人，2024b；Vasu 等人，2024a）。由于 MobileCLIP 需为数十亿图像生成大量合成描述，CoCa 的效率成为关键选择因素。MobileCLIP 未分析描述生成器的选择依据，但观察到使用合成描述较不使用时性能显著提升（3 万次迭代提升 7.4%）；尽管每张图像生成 5 个合成描述，但性能提升主要来自前 1-2 个。

本文探索在 DFN 数据集上训练新的 CoCa 模型，以提升合成描述质量。我们采用与 MobileCLIP 相同的 CoCa 架构（基于 ViT-L/14 图像编码器）：MobileCLIP 的 CoCa 模型在 LAION-2B 数据集上预训练，再在 MSCOCO-128k 数据集上微调；而本文的 CoCa 模型通过 OpenCLIP（Ilharco 等人，2021）在 DFN-2B 上预训练 130 亿样本。

**表 5：在 DFN-2B 上预训练 CoCa（不微调），IN-1k 性能相近但鲁棒性与检索性能下降**数据集为 DFN-5B12M，CLIP 教师模型与 MobileCLIP 一致（OpenAI+DataComp-XL CLIP-ViT-L/14），CoCa 架构为 CoCa-ViT-L/14。蒸馏时，系数\(\lambda\)设为 1.0（无 CLIP 损失），训练使用强图像增强。每组中与最优值相差不超过 1 个标准差的结果已高亮标注。

|蒸馏（Distill.）|强增强（High Aug.）|CoCa 模型训练数据集|IN-val 准确率（%）|Flickr30k 检索率（%）|38 项任务平均性能（%）|
|---|---|---|---|---|---|
|否|否|-|49.9|48.5|43.5|
|否|否|LAION-2B ￫ MSCOCO-128k|54.9|70.6|49.6|
|否|是|LAION-2B ￫ MSCOCO-128k|51.1|65.7|45.3|
|否|是|DFN-2B|54.6|55.1|46.2|
|否|是|LAION-2B ￫ MSCOCO-128k + DFN-2B|56.8|67.2|48.4|
|是|是|-|59.5|60.3|50.0|
|是|是|LAION-2B ￫ MSCOCO-128k|63.0|74.1|54.6|
|是|是|DFN-2B|63.1|64.7|55.2|
|是|是|LAION-2B ￫ MSCOCO-128k + DFN-2B|63.4|72.0|55.1|

表 5 展示了有无蒸馏时，DFN-CoCa 合成描述对性能的影响：使用 DFN-CoCa 合成描述可提升 IN-val 与 38 项任务平均性能，但检索性能下降。如 2.5 节所示，在 MSCOCO 等高质量数据集上微调后，检索性能可恢复。此外，将原始 CoCa 与 DFN-CoCa 的合成描述结合，可带来额外小幅性能提升（尤其在蒸馏场景下）。

### 2.5 描述生成器微调

2.4 节表明，在 DFN-2B 上预训练 CoCa 模型，可提升多模态强化训练的 IN-val 与 38 项任务平均性能，但检索性能下降 —— 这源于缺乏高质量数据集微调。MobileCLIP 的 CoCa 模型在 MSCOCO（Chen 等人，2015）上微调：MSCOCO-2017 包含 12.3 万张图像，其描述质量高于 DataComp 与 DFN 数据集中的平均图文对。

本文研究在不同高质量数据集上微调的影响：除 MSCOCO 的 12.3 万样本（MSCOCO-123k）外，还使用 3.8 万具有宽松许可（CC Attribution 2.0、CC Attribution-ShareAlike 2.0、CC Attribution-NoDerivs 2.0）的子集（MSCOCO-38k），以及 GBC-1M/10M（Hsieh 等人，2024）、DOCCI-9kshort/extended/complete（Onoe 等人，2025）、DCI-8k（Urbanek 等人，2024）、ReCap-COCO-30k（Li 等人，2024）。所有 DFN-CoCa 微调均使用与 CoCa 预训练相同的损失，训练 1200 万样本。

**表 6：DFN-5B12M 数据集上的微调实验**CLIP 教师模型为本文选择的 DFN 模型（DFN2B-CLIP-ViT-L-14-s39b 与 DFN2B-CLIP-ViT-L-14），CoCa 架构为 CoCa-ViT-L/14。蒸馏时，系数设为 1.0（无 CLIP 损失），训练使用强图像增强。

|CoCa 基础训练数据集|微调数据集（FT Dataset）|上下文长度（Context len.）|IN-val 准确率（%）|Flickr30k 检索率（%）|38 项任务平均性能（%）|
|---|---|---|---|---|---|
|LAION-2B|MSCOCO-123k|77|65.4±0.4|75.8±0.3|56.2±0.6|
|DFN-2B|-|77|65.9|68.7|55.9|
|DFN-2B|MSCOCO-123k|77|65.9|76.0|56.2|
|DFN-2B|MSCOCO-38k|77|65.9±0.3|75.4±0.2|56.5±0.3|
|DFN-2B|GBC1M-short|77|65.8|75.0|56.6|
|DFN-2B|DOCCI|77|66.3|72.6|57.3|
|DFN-2B|DCI-short|77|65.9|74.0|56.3|
|DFN-2B|DCI-extended|77|65.7|73.5|56.1|
|DFN-2B|DCI-complete|77|65.8|73.8|56.2|
|DFN-2B|Recap-COCO-30K|77|65.1|73.5|55.5|
|DFN-2B|GBC-1M-long|255|64.7|72.4|55.1|
|DFN-2B|GBC-10M-short-relation|255|65.2|73.8|55.4|
|DFN-2B|GBC-10M-long|255|64.6|71.9|54.6|
|DFN-2B|DOCCI|255|66.1|74.0|57.2|
|DFN-2B|DCI-extended|255|65.7|75.1|55.9|
|DFN-2B|DCI-complete|255|65.6|74.0|56.8|
|DFN-2B|5 个模型各生成 2 个描述|77|65.9±0.2|74.7±0.4|56.3±0.2|
|DFN-2B|10 个模型各生成 1 个描述|77|66.0±0.1|75.1±0.6|56.5±0.3|

#### MSCOCO-38k 与 MSCOCO-128k 微调

结果表明，仅在 MSCOCO 的宽松许可样本上微调，不会对性能产生负面影响。

#### 合成描述数量与束搜索消融

Vasu 等人（2024c）发现，尽管 CoCa 模型可生成多个合成描述，但分类任务中 2 个描述即可使性能饱和。本文使用单一 CoCa 模型，尝试不同采样策略（top-p、top-k、束搜索），发现束搜索生成的描述在定性上更具多样性，但用于强化训练时，下游性能无明显提升。

#### 在 GBC1M、GBC12M、DOCCI、DCI、ReCap-COCO30k 上的微调

多数微调数据集的性能低于或与 MSCOCO 微调持平（相差不超过 1 个标准差）；例外的是，在 DOCCI 上微调后，38 项任务平均性能提升 0.8%，显著高于 MSCOCO-38k（超过 1 个标准差）。

#### 上下文长度的影响

CLIP 与 CoCa 模型的上下文长度通常设为 77。本文尝试将训练与生成的上下文长度设为 255，以生成更长描述，多数结果与上下文长度 77 时相差不超过 1 个标准差。近期研究通过改进损失函数与训练策略，已提升 CLIP 模型对长描述的支持（Zhang 等人，2024；Zheng 等人，2024；Najdenkoska 等人，2024），将这些改进扩展到 CoCa 模型的工作留待未来开展。

#### 合成描述多样性的影响

本文进一步探索使用多个在不同数据集上微调的 CoCa 模型生成多样化描述：其核心假设是，微调数据集的多样性可提升合成描述的多样性，进而增强额外合成描述的有效性。结果表明，使用多达 10 个不同 CoCa 模型，性能仍与最优结果相差不超过 1 个标准差。

#### DFN 强化数据集

最终的小型强化数据集 DFNDR-5B12M 与 DFNDR-2B12M 包含：5 个经 MSCOCO-38k 微调生成的合成描述、2 个 DFN2B-ViT-L/14 教师模型（见 2.3 节）对 30 次图像增强及真实 / 合成描述生成的嵌入。我们对比了仅使用 DFN-2B 子集与使用包含 30 亿新增样本的 DFN-5B 全集的训练效果：表 7 显示，两者在 38 项任务平均性能上相差不超过标准差，但 DFN-5B 的 1200 万样本在 ImageNet-1k 验证集上表现更优；然而，大规模训练时这一优势消失，因此最终方案采用 DFN-2B 数据集。

**表 7：DFNDR-5B12M 与 DFNDR-2B12M 在 38 项任务平均性能上相近**

|数据集|IN-val 准确率（%）|Flickr30k 检索率（%）|38 项任务平均性能（%）|
|---|---|---|---|
|DFNDR-5B12M|65.9±0.3|75.4±0.2|56.5±0.3|
|DFNDR-2B12M|65.5|74.8|56.4|

## 3. 架构设计

MobileCLIP2 的架构既包含与 MobileCLIP 相似的变体，也新增了两种变体：具体而言，我们训练了 MobileCLIP2-S0、MobileCLIP2-S2、MobileCLIP2-B（其中 MobileCLIP2-S0 使用标准 “Base” 文本编码器，移除 S1 变体）；此外，新增 MobileCLIP2-S3 与 MobileCLIP2-S4 两种变体。这些变体的文本编码器为纯 Transformer 架构，图像编码器基于 FastViT（Vasu 等人，2023b）—— 该架构采用 Vasu 等人（2023a）提出的训练时过参数化块。

较小的变体（MCi0、MCi1、MCi2）为混合视觉 Transformer，包含 4 个不同计算阶段；新增的 MCi3 与 MCi4 则在输入张量 4 倍下采样后，额外增加一个 Transformer 阶段（图 3a）。五阶段设计在规模扩展时具有两大优势：1）参数可分布在五个阶段，且最大层仅需处理 1/4 数量的令牌；2）更易适配高分辨率输入。

我们通过实验验证了不同图像分辨率下的设计有效性：图 3b 中，将 MCi2 缩放至与 MCi3 参数规模相同（1.25 亿参数），并在四种输入分辨率下测试性能。结果显示，五阶段设计的 MCi3 较缩放后的 MCi2，性能权衡更优：在低分辨率（256×256）下，MCi3 速度是 MCi2 的 1.9 倍；在高分辨率（1024×1024）下，速度提升至 7.1 倍。高分辨率下的响应速度对图像编码器微调至关重要 —— 例如图像分割等密集预测任务，输入分辨率通常为 512×512。

![](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%27400%27%20height=%27256%27/%3e)![image](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjI1NiIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjwvc3ZnPg==)

## 4. 实验

本节将训练全新高效 CLIP 模型家族 MobileCLIP2，并在各类任务上评估性能。基于 2 节的发现，我们构建强化数据集 DFNDR-2B：包含 5 个由 CoCa-ViT-L/14 模型（DFN-2B 预训练、MSCOCO-38K 微调）生成的合成描述，以及 CLIP 集成模型（DFN2B-CLIP-ViT-L-14-s39b 与 DFN2B-CLIP-ViT-L-14）对所有图像、真实描述与合成描述生成的图文嵌入。MobileCLIP2 的架构家族较 MobileCLIP 更丰富，我们在 38 项零样本分类任务（Gadre 等人，2023）上评估其性能，尤其引入了基于 DFNDR-2B 训练的 MobileCLIP2-S3 与 MobileCLIP2-S4，以及基于 DataCompDR-1B 训练的 MobileCLIP-S3 与 MobileCLIP-S4（标注为 “MobileCLIP-S3/S4”）。表 8 对比了 MobileCLIP2 与相近延迟模型的性能，训练细节与超参数详见附录 A。

我们将 MobileCLIP2 与此前的小型 CLIP 架构（TinyCLIP（Wu 等人，2023，基于 LAION 训练（Schuhmann 等人，2022；2021））、ACED（Udandarao 等人，2024）），以及大尺寸模型（OpenAI CLIP（Radford 等人，2021）、DataComp（Gadre 等人，2023）、VeCLIP（Lai 等人，2023）、EVA（Sun 等人，2023）、DFN（Fang 等人，2024a）、SigLIP（Zhai 等人，2023）、SigLIP2（Tschannen 等人，2025））进行对比。所有评估均通过 OpenCLIP（Ilharco 等人，2021）与 DataComp（Gadre 等人，2023）完成；部分模型（如 SigLIP2）的评估结果与原文存在正负偏差。

MobileCLIP2 在不同延迟区间均实现 ImageNet-1k 零样本验证准确率的当前最优：值得注意的是，MobileCLIP2-S4 的 ImageNet 验证集零样本准确率与 SigLIP-SO400M/14 持平，但参数规模仅为后者的 1/2；其性能优于 DFN ViT-L/14，延迟却降低 2.5 倍。此外，相较于 ACED 模型，MobileCLIP2 在相近延迟下提升了 ImageNet-1k 性能 ——ACED 模型虽优化了推理计算量，但其 ACED-F1 与 ACED-F2 的延迟与 MobileCLIP2-S2 相近，参数规模与延迟仍更高。SigLIP-B/16 与 SigLIP2-B/16 的规模与延迟，更接近 MobileCLIP2 的新大尺寸变体；尤其需注意，SigLIP2 的文本编码器规模显著大于 SigLIP。

需说明的是，基于 DFNDR-2B 预训练的模型并非在所有检索任务中均实现当前最优 —— 这源于 DFNDR-2B 数据集对零样本分类（尤其 ImageNet-1k）的偏向性。实验发现，基于 DataComp、WebLI 及其衍生数据集训练的模型，检索性能可能高于 DFN 及其衍生数据集，但 38 项任务平均性能更低。因此，我们也在 DataCompDR-1B 上训练新架构（MobileCLIP-S3/S4），两种架构家族的结合可为更广泛应用场景提供灵活性。

### 4.1 视觉语言模型（VLM）评估

我们在 LLaVA-1.5 框架（Liu 等人，2024a）中评估 MobileCLIP2 预训练模型的视觉语言性能：所有实验均固定视觉骨干网络，使用 Qwen2-7B 替代 Vicuna-7B，其余训练细节与原始 LLaVA-1.5 一致（详见附录）。我们评估了在 130 亿样本上预训练的 ViT-B/16 模型（分别基于 DataComp、DFN、DataCompDR、DFNDR）：表 9 显示，基于 DFNDR 训练的模型平均准确率，较 DFN 预训练模型高 3.5%，较 DataComp 预训练模型高 1.6%，较 DataCompDR 预训练模型高 0.6%。

**表 9：LLaVA-1.5 框架下的 VLM 评估**基于 DFNDR 预训练的 ViT-B/16 模型，较 DFN 预训练模型准确率高 3.5%，较 DataComp 预训练模型高 1.6%，较 DataCompDR 预训练模型高 0.6%。

|数据集|GQA 准确率（%）|SQA 准确率（%）|TextVQA 准确率（%）|POPE 准确率（%）|MMMU 准确率（%）|MMB 准确率（%）|VizWiz 准确率（%）|VQAv2 准确率（%）|平均准确率（%）|
|---|---|---|---|---|---|---|---|---|---|
|DataComp-1B|59.6|71.5|50.5|81.8|42.6|59.1|51.8|70.7|61.0|
|DFN-2B|56.9|71.3|46.0|81.4|41.9|52.2|60.2|56.1|59.1|
|DataCompDR-1B|60.3|73.1|50.4|81.7|43.6|45.2|54.9|72.1|62.0|
|DFNDR-2B|60.4|72.9|49.9|83.3|61.9|54.5|72.4|62.6|66.9|

### 4.2 密集预测任务

我们通过微调图像编码器，评估其在密集预测任务（目标检测、语义分割、深度估计）上的视觉表征质量：

- **目标检测与实例分割**：表 10 报告了 ViT-B/16 模型搭配 MaskRCNN（He 等人，2017）头，在 MS-COCO（Chen 等人，2015）数据集上的实例分割性能。所有模型均使用 MMDetection 库（Chen 等人，2019），按 Wei 等人（2023）的描述采用 1× 调度与单尺度测试，微调设置详见 Wei 等人（2023）（附录含更多细节）。
    
- **语义分割**：表 11 报告了 ViT-B/16 模型搭配 UperNet（Xiao 等人，2018）头，在 ADE20k（Zhou 等人，2017）数据集上的性能。训练设置遵循 Liu 等人（2024c）的描述。
    
- **深度估计**：表 12 报告了模型在 NYUv2 数据集（Nathan Silberman & Fergus，2012）上的均方根误差（RMSE）。实验设置与 Vasu 等人（2024b）一致（附录含更多细节）。
    

此外，我们还评估了 MobileCLIP2 小尺寸变体在密集预测任务上的性能：MAE（He 等人，2022）等主流预训练方法无法直接应用于 MCi 等分层卷积与混合架构，因此我们将 MobileCLIP2 预训练与相同架构的监督预训练进行对比。表 13 与表 14 显示，MobileCLIP2 预训练显著优于监督预训练，可作为分层架构的优质预训练方案。

**表 10：MS-COCO 数据集上基于 Mask-RCNN 头的目标检测与实例分割结果（1× 调度训练）**所有模型均为 ViT-B/16 架构。

|方法|数据集|边界框 mAP（mAP box）|掩码 mAP（mAP mask）|
|---|---|---|---|
|CatLIP（Mehta 等人，2024）|DataComp|45.7|40.6|
|MAE（He 等人，2022）|IN-1K|46.5|40.9|
|MAE（Singh 等人，2023）|IG-3B|46.4|42.1|
|MAWS（Singh 等人，2023）|IG-3B|48.0|43.4|
|FD-CLIP（Wei 等人，2023）|OpenAI-WIT + IN-1K|48.2|42.5|
|CLIP（Radford 等人，2021）|OpenAI-WIT|45.0|39.8|
|MobileCLIP2|DFNDR-2B|47.0|41.8|

**表 11：ADE20k 数据集上基于 UperNet 解码器的语义分割结果**所有模型均为 ViT-B/16 架构。

|方法|数据集|平均交并比（mIoU）|平均准确率（mAcc）|
|---|---|---|---|
|MAE（He 等人，2022）|IN-1K|48.1|58.9|
|dBOT（Liu 等人，2024c）|IN-1K|49.5|60.7|
|MAWS（Singh 等人，2023）|IG-3B|50.4|61.5|
|CatLIP（Mehta 等人，2024）|DataComp|50.6|61.8|
|FD-CLIP（Wei 等人，2023）|OpenAI-WIT + IN-1K|51.7|-|
|CLIP（Radford 等人，2021）|OpenAI-WIT|49.5|-|
|MobileCLIP2|DFNDR-2B|52.8|64.0|

**表 12：NYUv2 数据集上的深度估计结果（遵循 Wei 等人，2023 的设置）**所有结果均为 ViT-B/16 模型。

|方法|数据集|RMSE（越低越好）|
|---|---|---|
|CatLIP（Mehta 等人，2024）|DataComp|0.394|
|MAE（He 等人，2022）|IN-1K|0.383|
|MAWS（Singh 等人，2023）|IG-3B|0.371|
|FD-CLIP（Wei 等人，2023）|OpenAI-WIT + IN-1K|0.352|
|MAE（Singh 等人，2023）|IG-3B|0.348|
|CLIP（Radford 等人，2021）|OpenAI-WIT|0.416|
|MobileCLIP2|DFNDR-2B|0.356|

**表 13：ADE-20k 语义分割预训练方法对比**灰色标注为近期语义分割当前最优模型。

|编码器|解码器|预训练方式|分辨率|参数规模（百万）|mIoU|
|---|---|---|---|---|---|
|InternImage-B（Wang 等人，2023）|UperNet（Xiao 等人，2018）|监督 IN-1K|512×512|128.0|50.8|
|ViT-Adapter-B（Chen 等人，2023）|SemanticFPN（Kirillov 等人，2019）|监督 IN-22K|512×512|104.6|50.7|
|ViT-Adapter-B（Chen 等人，2023）|UperNet（Xiao 等人，2018）|监督 IN-22K|512×512|133.9|51.9|
|Swin-L（Liu 等人，2021）|UperNet（Xiao 等人，2018）|监督 IN-22K|640×640|234.1|52.1|
|MCi0|SemanticFPN（Kirillov 等人，2019）|监督 IN-1K|512×512|14.5|44.8|
|MCi2|SemanticFPN（Kirillov 等人，2019）|监督 IN-1K|512×512|38.5|48.9|
|MCi0|SemanticFPN（Kirillov 等人，2019）|MobileCLIP2|512×512|14.5|47.0（+2.2）|
|MCi2|SemanticFPN（Kirillov 等人，2019）|MobileCLIP2|512×512|38.5|51.6（+2.7）|

**表 14：MS-COCO 目标检测任务预训练方法对比（基于 MaskRCNN（He 等人，2017）检测头，1× 调度训练）**灰色标注为近期目标检测当前最优模型。

|模型|预训练方式|参数规模（百万）|边界框 mAP|掩码 mAP|
|---|---|---|---|---|
|ViT-Adapter-B（Chen 等人，2023）|监督 IN-1K|284|47.0|41.8|
|InternImage-B（Wang 等人，2023）|监督 IN-1K|115|48.8|44.0|
|ViT-Adapter-L（Chen 等人，2023）|监督 IN-22K|347.9|48.7|43.3|
|MCi0|监督 IN-1K|31.0|41.8|38.0|
|MCi2|监督 IN-1K|55.0|46.6|41.7|
|MCi0|MobileCLIP2|31.0|44.4（+2.6）|39.6（+1.6）|
|MCi2|MobileCLIP2|55.0|49.1（+2.5）|43.2（+1.5）|

## 5. 相关工作

多模态模型训练的改进主要集中在三方面：数据、目标函数与架构。MobileCLIP2 基于 MobileCLIP 开发，在这三方面均实现改进。

### 数据层面

数据改进方法分为两类：数据集过滤与信息增强。基础过滤方法先收集大规模候选图文对，再基于 URL 或图文统计特征通过启发式规则筛选（Radford 等人，2021；Schuhmann 等人，2021；2022；Xu 等人，2024）；更先进的过滤方法则使用在高质量数据上训练的过滤模型移除低质量图文对，这类方法可能采用预训练 CLIP 模型（Gadre 等人，2023）或更专用的过滤模型（Fang 等人，2024a）。数据方法的挑战在于，启发式规则或预训练模型可能引入偏差 —— 例如，DataComp 等多数公开数据集仅筛选英文数据，限制模型在非英文任务上的性能（Carlsson 等人，2022；Nguyen 等人，2024；Pouget 等人，2024）。此外，预训练模型还可用于基于样本难度的主动数据选择（Evans 等人，2024a；b）；研究还发现，重复使用高质量数据可提升数据利用率（Goyal 等人，2024）。

更广泛地说，预训练模型的输出可作为新增强数据集的一部分 —— 例如，多项研究利用图像描述模型为数据集中的图像生成合成描述（Yang 等人，2023a；Nguyen 等人，2023；Lai 等人，2023；Liu 等人，2024d；Li 等人，2024）；大型语言模型可重写真实描述（Fan 等人，2023），也可与文本到图像模型结合生成全合成数据集（Hammoud 等人，2024）。MobileCLIP 提出多模态数据集强化方法，通过图像描述模型生成合成描述，并利用 CLIP 大模型集成生成多图像增强与合成描述的 CLIP 嵌入，高效存储这些信息（Vasu 等人，2024c）。本文采用类似方法，但通过更优的 DFN 模型（Fang 等人，2024a）改进描述生成器与 CLIP 嵌入生成器。

### 目标函数层面

多模态训练目标函数的改进：原始 CLIP 采用对比损失，使数据集中配对的图文表征相近，而与批次内其他图文表征远离（Radford 等人，2021）；SigLIP 提出基于 Sigmoid（而非 Softmax）的损失变体，在大批次训练中提升效率（Zhai 等人，2023；Tschannen 等人，2025）；其他方法则采用基于图像掩码（Yang 等人，2023b；Fang 等人，2023；Sun 等人，2023；Li 等人，2023b）、单模态自监督（Mu 等人，2022；Li 等人，2021）的目标函数，以及多分辨率训练（Li 等人，2023a）以降低训练成本。多模态蒸馏可带来更显著的性能提升，尤其对小尺寸架构变体（Wang 等人，2022b；Kuang 等人，2023；Wang 等人，2022a；Wu 等人，2023）。值得注意的是，MobileCLIP（Vasu 等人，2024c）通过离线知识蒸馏方法（Shen & Xing，2022；Yun 等人，2021；Faghri 等人，2023）实现了高训练效率。本文采用与 MobileCLIP 类似的目标函数，包含图文对嵌入蒸馏与合成描述蒸馏。

### 架构层面

架构改进旨在在给定参数、计算量或延迟预算下，提升推理效率与性能。CLIP 架构通常借鉴单模态图像与文本模型 —— 原始 CLIP 及后续多数工作采用标准 ViT 架构，搭配改进的 BERT 文本编码器（Dosovitskiy 等人，2020；Devlin 等人，2019；Radford 等人，2021）。CLIP 的高效架构包括：通过剪枝 ViT 得到的 TinyCLIP（Wu 等人，2023）、通过减少令牌数量的 Cao 等人（2023）、通过减少参数降低计算量的 Evans 等人（2024b）。MobileCLIP 为 CLIP 设计了专用高效架构，提出低延迟卷积 - Transformer 混合架构，同时适配图像与文本编码器。本文进一步改进架构，新增两种变体以填补常见 B 与 L 架构间的延迟空白。

## 6. 结论

本文提出全新低延迟图文模型家族 MobileCLIP2，实现 ImageNet-1k 零样本验证准确率的当前最优。我们通过采用更强的 CLIP 教师模型与新训练的图像描述模型，改进多模态强化训练；尤其对 CLIP 教师模型的调优与集成、高效图像描述模型的训练与微调进行了全面研究。值得注意的是，MobileCLIP2-S4 的 ImageNet-1k 零样本准确率与 SigLIP-SO400M/14 持平，但参数规模仅为后者的 1/2；其性能优于 DFN ViT-L/14，延迟却降低 2.5 倍。我们已开源模型检查点与数据生成代码，为大规模数据集生成提供支持。

## 更广泛影响说明

本文提出的基础模型家族专为移动与边缘设备部署优化，有助于扩大基础模型的应用范围，为更广泛用户群体开发应用提供支持。MobileCLIP2 可用于图像分类等各类场景，但其输出性能会受训练数据集与教师模型固有偏差的影响。

## 致谢

感谢 Albin Madappally Jose、Barry Theobald、Chen Huang、Rick Chang 及苹果公司机器学习研究团队，在项目期间提供的帮助与讨论。

## 参考文献（注：保留原文格式，关键会议 / 期刊名已译为中文）

Lucas Beyer, Xiaohua Zhai, Amélie Royer, Larisa Markeeva, Rohan Anil, and Alexander Kolesnikov. 知识蒸馏：优秀的教师需兼具耐心与一致性. In _IEEE/CVF 计算机视觉与模式识别会议论文集_, 第 10925-10934 页，2022.Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. 基础模型的机遇与风险. arXiv 预印本 arXiv:2108.07258, 2021.Qingqing Cao, Bhargavi Paranjape, and Hannaneh Hajishirzi. PuMer：通过剪枝与合并令牌优化视觉语言模型效率. In _第 61 届计算语言学协会年会论文集（第 1 卷：长论文）_, 2023.Fredrik Carlsson, Philipp Eisen, Faton Rekathati, and Magnus Sahlgren. 跨语言与多语言 CLIP. In _第 13 届语言资源与评估会议论文集_, 第 6848-6854 页，2022.Kai Chen, Jiaqi Wang, Jiangmiao Pang, Yuhang Cao, Yu Xiong, Xiaoxiao Li, Shuyang Sun, Wansen Feng, Ziwei Liu, Jiarui Xu, Zheng Zhang, Dazhi Cheng, Chenchen Zhu, Tianheng Cheng, Qijie Zhao, Buyu Li, Xin Lu, Rui Zhu, Yue Wu, Jifeng Dai, Jingdong Wang, Jianping Shi, Wanli Ouyang, Chen Change Loy, and Dahua Lin. MMDetection：开源 MMLab 检测工具包与基准. arXiv 预印本 arXiv:1906.07155, 2019.Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollár, and C Lawrence Zitnick. Microsoft COCO 描述：数据收集与评估服务. arXiv 预印本 arXiv:1504.00325, 2015.Zhe Chen, Yuchen Duan, Wenhai Wang, Junjun He, Tong Lu, Jifeng Dai, and Yu Qiao. 用于密集预测的视觉 Transformer 适配器. In ICLR, 2023.MMSegmentation 贡献者. MMSegmentation：开源 MMLab 语义分割工具包与基准. [https://github.com/open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation), 2020.Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT：用于语言理解的深度双向 Transformer 预训练. In _2019 年北美计算语言学协会年会论文集：人类语言技术（第 1 卷：长论文与短论文）_, 第 4171-4186 页，2019.Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. 一张图像相当于 16×16 个单词：用于大规模图像识别的 Transformer. arXiv 预印本 arXiv:2010.11929, 2020.（后续参考文献均遵循此格式，保留作者名、年份及会议 / 期刊信息，关键术语已译为中文，技术工具名与模型名保留原文。）