# Overview
* 专门为遥感（RS）场景量身定制。
* 擅长处理**高分辨率**RS图像，采用**区域级推理**进行全面的**场景解释**。
* 利用新创建的**RS多模态数据集**。
* 使用**LLaVA-1.5架构**进行了微调。
* 可以在各种RS任务中实现**稳健的零样本**性能，包括
	* 图像和区域字幕
	* 视觉问题回答
	* 场景分类
	* 基于视觉的对话
	* 参考对象检测

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504041654842.png)

# Train
使用geochat_directive Dataset：**318k vicuna生成的多模态指令跟随数据**进行**视觉指令调优**，在**LlaVA-v1.5**的预训练权重上进行**微调**。  

在**3个拥有40GB内存的A100 gpu**上训练GeoChat。

## Hyperparameters
| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| -------------- | ----------------: | ------------: | -----: | ---------: | -----------: |
| GeoChat-7B     |               144 |          2e-5 |      1 |       2048 |            0 |

## Pretrain(Feature alignment)
使用LLaVAv1.5的预训练投影器，其在带有BLIP caption的LAION-CC-SBU数据集的558K子集上进行训练。LLaVA-v1.5-7B大约需要3.5小时。

- `--mm_projector_type mlp2x_gelu`: the two-layer MLP vision-language connector.
- `--vision_tower openai/clip-vit-large-patch14-336`: CLIP ViT-L/14 336px.

# Evaluation
在7个不同的基准上评估GeoChat。为了保证模型的再现性，使用贪婪解码对模型进行评估。不评估使用beam search([十分钟读懂Beam Search 1：基础 - 知乎](https://zhuanlan.zhihu.com/p/114669778))使推理过程与实时输出的聊天演示一致。

# Contributions
* **RS多模态指令跟随数据集**。利用现有的对象检测数据集创建图像的简短描述，然后使用Vicuna-v1.5单独生成的文本创建对话。此外，添加了视觉问答和场景分类能力，使用他们的相应数据集。使得RS总共有**318k个指令对**。
* **GeoChat**。作者利用创建的数据集对LLaVA-1.5进行微调，以创建遥感域视觉语言模型- GeoChat。我们的**LoRA微调**是高效的，避免了忘记完全调优的LLaVA模型中嵌入的必要上下文，该模型的MLP投影将图像对齐到LLM （Vicuna-v1.5）的词嵌入空间中。这使得GeoChat保留了LLaVA的对话和指令跟踪能力，并将其领域知识扩展到遥感任务。  
* **评价基准**。作者为RS中的会话基础设置了评估协议，并设置了一套任务，以便与该方向的未来工作进行比较。我们展示了不同遥感任务的各种监督评估和零样本评估，包括图像字幕，视觉问答和场景分类，以展示GeoChat会话VLM的通用性。

# Model Architecture
给定图像输入和用户查询，首先使用视觉主干通过**插值位置编码**以更高的分辨率编码**patch级令牌**。多层感知器（MLP）用于将视觉标记适应于适合大型语言模型（Vicuna 1.5）输入的语言空间。除了**视觉输入**外，还可以将**区域位置与特定于任务的提示**一起输入到模型中，这些提示指定用户所需的所需任务。在此背景下，LLM可以生成与相应对象位置交错的自然语言响应。GeoChat可以执行多个任务，如上图所示，例如场景分类，图像/区域字幕，VQA和接地对话。

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504041651575.png)


# RS Multipal Instruction Dataset
对于给定的RS图像，我们获得对象属性和关系信息，引用**表达式**和**区域caption**以及**相应的区域注释（显示在图像上）**。该结构化信息用于创建包含318k个图像指令对的丰富指令集。

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504041659876.png)
