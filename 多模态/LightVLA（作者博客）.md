- 项目主页：[https://liauto-research.github.io/LightVLA/](https://link.zhihu.com/?target=https%3A//liauto-research.github.io/LightVLA/)
- 论文链接：[https://arxiv.org/abs/2509.12594](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2509.12594)  

**核心创新**

LightVLA 是一个旨在提升 VLA 推理效率且同时提升性能的视觉 token 剪枝框架。当前 VLA 模型在具身智能领域仍面临推理代价大而无法大规模部署的问题，然而大多数免训练剪枝框架依赖于中间注意力输出，并且会面临性能与效率的权衡问题。为应对这些挑战，LightVLA 引入了两大核心创新：

- **无参数可微分 token 剪枝框架**：创新的应用==无参数查询初始化==和 ==Gumbel softmax 技术==实现训练时可微分 VLA 模型能够根据多样的文本任务输入自适应地选择对任务完成==最有贡献的关键视觉 token==，验证了性能和效率可以做到协同优化。  
    
- **基于可学习查询的 token 剪枝框架**：相比于无参数的 LightVLA，LightVLA * 初始化一系列的可学习查询（==Learnable Query==），可分别作用于视觉编码器或 LLM 浅层解码器层，借助额外参数引导 VLA 模型学习关键视觉 Token 选取，同样实现了较好的性能提升。

我们研究了 VLA 模型中广泛存在的视觉 token 冗余，设计了一种在微调中实现可微分视觉 token 剪枝的加速框架，创新实现 Gumbel-softmax 引导的无参数 token 选择过程，强化 LightVLA 对关键视觉 token 的选择能力，为 VLA 模型的推理加速提供新的范式。

在 LIBERO 上的实验证明，LightVLA 不仅取得了当前最佳性能（SOTA），超越了 Pi-0 、Openvla-OFT 等经典 VLA 模型，并且实现了高效的推理加速。并且通过可学习的参数初始化 query 选择方法 LightVLA * 验证微调实现推理加速的可行性。消融实验充分验证了 LightVLA 自适应 token 选择的有效性，并证实效率和性能并非是此消彼长的零和博弈，为构建更高效的具身智能大模型提供了新的思路。

**研究动机与核心挑战 (Motivation)**

**让 VLA 学会更聪明地选择关键视觉 token**

当前视觉 - 语言 - 动作（VLA）模型在机器人任务中展现了令人瞩目的认知推理和操作等能力，但庞大的计算开销和较高的推理延迟限制了其广泛部署，如家用机器人。

我们认为计算瓶颈的核心大部分来源于**视觉 token 的固有冗余**，而传统基于 Token 剪枝的加速方法往往面临 “效率 vs 性能” 的权衡困境，现有工作为了提升效率而剪枝视觉 token，不可避免地造成模型性能下降。我们认为对于 VLA 的视觉输入，==冗余的视觉 token 不仅会带来额外的计算开销，而且稀释了模型对于关键区域的注意力，同时造成性能和效率的下降==。

因此，我们认为效率和性能并非天生矛盾，但需要引入更聪明的剪枝方法，而非以固定比例或固定个数限制保留 token 的数量，让模型学会**主动、任务自适应地 “关注” 信息最丰富的视觉区域**，并忽略无关的冗余信息。基于此，我们实现了两种 token 选择机制：

- **`LightVLA`**: 它不再依赖任何启发式的超参数预设剪枝比例，基于==无参数查询==的方式，进一步引入 ==Gumbel-softmax== 实现在微调过程实现 ==token 的可微分选择==，使模型自发学会保留对提升任务性能更重要的 “关键 token”，从而实现性能和效率的双重提升。  
    
- **`LightVLA*`**: 为了验证在微调过程中==剪枝的可行性以及剪枝位置的影响==，我们进一步探索了基于==可学习查询==的剪枝框架，在引入可训练参数后，LightVLA * 仍然可实现性能和效率的较好提升。

![](https://pic4.zhimg.com/v2-70cd0146687bef3748d1727cbd90363f_1440w.jpg)

上图展示了 LightVLA 在 LIBERO 中和主流 VLA 模型、剪枝类方法在视觉 token 数量和任务性能上的对比。从图中可以看出，LightVLA 实现了在保留 token 最少情况下最好的性能，不仅说明了视觉 token 的大量冗余，同时说明通过微调可以实现性能和效率优化的最优解。

**方法详解**

**总体架构示意图**

![](https://picx.zhimg.com/v2-0d3d3c8a7bdcf7b45597446db328c8d3_1440w.jpg)

**可微分的 Token 剪枝**  

我们提出了可微分的 Token 剪枝算法，以实现自适应剪枝。我们使用一系列 Query 来实现 Token 筛选，具体来说，LightVLA 构建了与 Visual Token 数量相同的 Query，并由每个 Query 独立选择一个最重要的 Visual Token。没有被 Query 选中的 Token 被剪除，而所有被 Query 选中的 Visual Token 组成剪枝后的 Token 集。可微分的 Token 剪枝算法具体流程如下：  

- **Query 生成**  
    
	LightVLA 使用一组 Query Token 来识别有用和无用 Token。一个 Visual Token 是否有用，由携带的视觉信息和 VLA 输入的文字指令共同决定。因此，LightVLA 取 Visual Token 对 Language Token 的 Cross Attention，在视觉表征中融合任务信息，作为 Query。
	
	![](https://picx.zhimg.com/v2-d1a7c1a94a057560d4cae5bde35908b7_1440w.jpg)

- **Token评分**  
    
	计算Query Token与Visual Token之间的内积，作为每一个Query Token对每一个Visual Token的重要性评分。
	
	![](https://pic3.zhimg.com/v2-53b34ca7e2b6a4276911cf3edcbbd848_1440w.jpg)

- **Token 筛选**  
    
	每个 Query 独立地选择重要性评分最高的 Visual Token，所有被选中的 Visual Token 保留下来，而没有被选中的 Visual Token 被剪除。
	
	![](https://pic1.zhimg.com/v2-f47f862d3f07f072e4fdbc8c2efd554c_1440w.jpg)
	
	注意到 argmax 是不可导运算，在这里，我们==使用 Gumbel-softmax 技巧将 argmax 变为可导运算==，以实现训练时梯度的反向传播。首先，为了提高训练过程中 Token 筛选的多样性，我们给重要性评分注入采样噪声：
	
	![](https://pic3.zhimg.com/v2-0f0412035821885a8847901d28e00020_1440w.jpg)
	
	最后，筛选后的Token集可以通过以下公式得到：
	
	![](https://pic3.zhimg.com/v2-aa2e0541352ee3e7276945a8af54cc5e_1440w.jpg)
	
	![](https://pic3.zhimg.com/v2-8e90fca8db7e180bac1369f8d54bd6e6_1440w.jpg)
	
	为了在训练前期鼓励模型探索 Token 筛选的多样性，而在训练后期使 Token 筛选的策略收敛，我们对采样噪声的方差进行线性衰减，使噪声方差从 1 逐渐衰减至 0。

  

**实验结果**

![](https://pic3.zhimg.com/v2-29168d272fd05f4a989ad6a1c372dfd4_1440w.jpg)

![](https://pic3.zhimg.com/v2-8afd12dd0c71744df505e9237e80343a_1440w.jpg)

- **LightVLA 在指标上显著超越现有基线**：在 LIBERO 基准上的所有任务中，LightVLA 均取得了最佳表现，平均成功率达到 97.4%，全面超越包括 OpenVLA-OFT 在内的各类强基线模型。这表明 LightVLA 在兼顾效率的同时仍能保持领先的任务执行能力。  
    
- **Token 稀疏性揭示冗余视觉信息**：与消耗 512 个视觉 token 的 OpenVLA-OFT 相比，LightVLA 仅保留平均 78 个视觉 token，却仍实现更优性能。这一结果表明，大量视觉 token 并未贡献有效信息，LightVLA 成功捕捉关键语义 token，证明了视觉模态的高度稀疏性。  
    
- **唯一兼顾性能与效率的加速方案**：在与其他加速方法的对比中，LightVLA 不仅将 FLOPs 与延迟分别减少 59.1% 与 38.2%，同时还提升平均成功率 2.6%。值得注意的是，LightVLA 是现有所有加速方法中唯一一个在加速的同时还能提升性能的方案，验证了消除视觉冗余能够同时优化性能与效率。

  

**剪枝过程可视化**

![](https://picx.zhimg.com/v2-a0dc8040fa114f2223f4631e28ece7cb_1440w.jpg)

为了说明 LightVLA 剪枝过程的可解释性，我们随机选择了任务 “把咖啡壶放在炉子上”，并展示任务执行过程中的 token 选择可视化结果，每帧图片的左右两列分别代表第三人称相机输入和腕部相机输入，第二行点亮的区域代表被选择的视觉 token。关键帧代表操作任务的重要阶段（物体交互，任务完成等），可以看出被保留的视觉 token 更多地关注咖啡壶、炉子、和机械臂本体等任务相关物体，并没有关注无用的背景信息。进一步验证了 LightVLA 在自适应 token 选择上的能力。

**消融实验**

- 噪声衰减的有效性：
	
	![](https://pic4.zhimg.com/v2-8502167af18d756181c7496d1d66963f_1440w.jpg)
	
	**引入噪声提升模型探索能力**：不引入噪声的变体保留最少的视觉 token，实现了次优的性能，说明==噪声的引入对训练过程中模型主动探索任务相关的视觉 token 至关重要==，否则会导致对语义密集场景视觉 token 的 “无感”。
	
	**噪声衰减让模型变得更 “聪明”**：固定噪声的引入使得模型保留最多的视觉 token，但模型对关键 token 的筛选能力不足，==噪声衰减让模型学会对视觉 token 的有效取舍从而提升到最优性能==。

  

- Token 选择有效性：
	![](https://pica.zhimg.com/v2-9168a91f73a0117f9f47b69522e353e0_1440w.jpg)
	
	**保留无用 token 导致性能下降**： 当在 LightVLA 已保留的 k 个 token 之外再补充 k 个随机 token 时，整体性能反而下降，说明 LightVLA 已经捕捉到所有关键信息，额外的随机 token 只会引入噪声与干扰。
	
	**丢弃有用 token 导致性能下降**：当从 LightVLA 已筛选的 k 个 token 中随机丢弃 10% 时，性能同样下降。充分验证 LightVLA 学会了选择对任务成功率更相关的视觉 token，并没有保留无用信息。
	  

**结论**

我们研究了视觉 - 语言 - 动作（VLA）模型中固有的视觉冗余问题，并提出了一种无参数的可微分视觉 token 剪枝框架 LightVLA。通过基于无参数查询的 token 剪枝过程，该方法能够自适应地选择最具信息量的视觉 token。在 LIBERO 基准上，LightVLA 在显著降低计算开销的同时取得了当前最优的性能。我们还提出了另一种变体 LightVLA*，相较于 LightVLA，其引入了可学习查询作为额外的可训练参数，同样在性能上优于同类方法。本工作为解决 VLA 模型中的视觉冗余挑战提供了新的范式，在实现更低计算开销与时延的前提下取得了更优性能，为未来 VLA 模型轻量化与部署提供了新颖的解决方案。