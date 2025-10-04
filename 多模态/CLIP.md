# 模型总览图
![25aaa0d0-21c8-4d07-8e83-b6960e6d7cfa](file:///C:/Users/Lenovo/Pictures/Typedown/25aaa0d0-21c8-4d07-8e83-b6960e6d7cfa.png)

# 核心思想——对比学习来实现多模态融合

不用传统的标签来作为监督信号，而用自然语言文本来作为监督信号，这大大提升了模型的迁移学习能力，使得其不再是单模态模型，而是多模态模型。其最大的缺点就是需要一个特别大的数据集来做预训练，计算所需的资源和时间也十分的高昂。

首先在数据集方面，在整合分析前人工作之后，觉得应该要有足够大的数据样本来预训练才能达到一个好的效果，所以作者自己创建了一个4亿大小的“图片-文本对”数据集。

接着，数据这么大，预训练计算时肯定要非常久，所以要想怎么设计模型才能使得减少计算时间。

本想用类似GPT所用的”预测“框架来预训练，即给定一张图片，先生成该图片可能对应的文本，再与真实文本比较，以此预测该图片对应的文本。但会造成训练效率十分的低下。

所以作者采用 **“对比学习”** 的方式，只要图片和文本是配对的就行，不用预测，伪代码如下：

![0de1fa47-953c-4b69-b5df-5dccbf525a35](file:///C:/Users/Lenovo/Pictures/Typedown/0de1fa47-953c-4b69-b5df-5dccbf525a35.png)

先分别进行encoder处理，再投影（即dot，用来学习从单模态到多模态的一个过程），再归一化。再分别对它们各自生成的特征向量I_e和T_e进行相似度的计算，得到的相似度矩阵再与ground truth（即图中的labels）进行loss交叉熵损失计算。注意看，两个loss是不一样的，axis一个是0一个是1，一个是对行做softmax，一个是对列，一个是I×logT，一个是T×logI。
# 模型细节
因为数据太大了，所以不会出现过拟合的问题，也无需做很多复杂的数据增强，只是做了简单的裁剪罢了，还把对比学习中很重要的temperature这个超参数直接设置成了可自动优化的标量，因为模型太大，实在是太难调参了。所用到的两个encoder都是无需提前预训练的。最后的投射层用的不是非线性的投射层，而是线性的投射层，作者推测，对于单模态学习来说，需要非线性投射层，而对于多模态学习来说，是否线性都无所谓，所以就用了简单的线性投射层。

**prompt engineering and ensembling**：只用一个单词来作为prompt的话，经常出现歧义性这种问题，所以要将其转化为一个句子。
# 实验细节
32个epochs，optimizer=Adam，用cosine schedule衰减学习率。==初始超参数是在训练1个时期时使用网格搜索、随机搜索和基线ResNet50模型上的手动调整相结合来设置的。由于计算限制，超参数被启发式地适应更大的模型。==可学习的温度参数τ被初始化为0.07，并被裁剪以防止将logits缩放超过100，这对于防止训练不稳定是必要的。试了res50、res50×4、res50×64、VIT-L等等模型。调参时用的是最小的res50，只用了1个epoch来看效果。最后用了VIT-L/14。batch_size=32768。用了混精度训练（能够加速训练和省内存，transformer的模型基本都用到了混精度训练）。==为了节省额外的内存，使用了梯度检查点、半精度Adam统计和半精度随机舍入文本编码器权重。==最后还用了更大尺寸（336）的图片进行fine-tune，使得性能进一步提升。


# 一些问题
## CLIP是有监督学习还是无监督学习？

CLIP（Contrastive Language-Image Pretraining）是一种结合图像与文本的多模态模型，其训练方式介于**有监督学习**和**自监督学习**之间，但更倾向于被归类为**有监督学习**，具体分析如下：

---

### **1. CLIP的核心训练机制**

CLIP通过对比学习（Contrastive Learning）将图像和文本映射到同一语义空间。其训练数据是**大规模的图像-文本对**（例如网络图片及其自然语言描述）。模型的目标是让匹配的图像-文本对在嵌入空间中距离更近，不匹配的对更远。

**关键点**：

- **监督信号来源**：文本描述作为图像的“弱标签”提供监督信号。例如，一张猫的图片可能对应文本“一只躺在沙发上的猫”，而非简单的类别标签“猫”。
- **与传统监督学习的区别**：传统监督学习依赖人工标注的离散标签（如分类任务中的类别编号），而CLIP的监督信号是自然语言文本，信息更丰富但噪声更高。

---

### **2. 为何属于有监督学习？**

根据搜索结果中对监督学习的定义：

- **监督学习的核心特征**：使用**带标签的数据**进行训练，标签为模型提供明确的输出目标。
- **CLIP的标签形式**：虽然文本描述并非传统分类标签，但它们仍然是一种显式的监督信号，指导模型学习图像与文本的对应关系。因此，CLIP的训练可视为**基于自然语言监督的有监督学习**。

---

### **3. 与无监督学习的关联**

部分观点认为CLIP属于无监督学习，原因包括：

- **数据获取方式**：CLIP的训练数据来自互联网的公开图像-文本对，未经过人工标注（属于“弱监督”或“自监督”）。
- **任务目标**：CLIP的目标是学习通用的多模态表示，而非直接完成分类或回归任务，这与无监督学习中“发现数据内在结构”的目标有一定相似性。

但严格来说，无监督学习通常指**完全无标签的数据**（如聚类、降维任务），而CLIP依赖成对的图像-文本信息，因此更接近有监督学习的扩展形式。

---

### **4. 总结：CLIP的定位**

- **有监督学习框架**：CLIP明确使用图像-文本对作为监督信号，属于有监督学习的范畴，但采用了更灵活的监督形式（自然语言描述而非离散标签）。
- **与自监督学习的交叉**：CLIP通过对比学习自动生成正负样本对，这一机制与自监督学习中的“数据增强生成标签”有相似之处，但自监督学习通常不依赖外部信息（如文本），因此两者存在区别。

---

### **类比说明**

- **传统监督学习**：类似老师直接告诉学生“这是猫”（标签明确）。
- **CLIP的监督学习**：类似老师用一段描述解释“这是一只毛茸茸的、在沙发上打盹的动物”（标签为自然语言，信息更丰富但需模型自行提取关键特征）。

因此，CLIP可视为一种**基于自然语言监督的有监督学习模型**，其创新之处在于利用大规模弱标注数据实现了跨模态的语义对齐。

# 架构一览

## **CLIP模型（顶层）**
- 核心功能：学习图像与文本的联合表示，计算跨模态相似度
- 关键组件：视觉编码器、文本编码器、跨模态交互模块
- 输出：图像-文本相似度分数（logits_per_image、logits_per_text）

## **视觉编码器（visual）**
- 类型：根据`vision_layers`类型二选一
  - ### **ModifiedResNet（残差网络变体）**
    - #### **Stem模块（输入处理）**
      - conv1：3x3卷积（输入3通道→width//2通道，步长2，填充1）
      - bn1：BatchNorm2d（归一化）
      - relu1：ReLU（激活）
      - conv2：3x3卷积（width//2→width//2通道，填充1）
      - bn2：BatchNorm2d
      - relu2：ReLU
      - conv3：3x3卷积（width//2→width通道，填充1）
      - bn3：BatchNorm2d
      - relu3：ReLU
      - avgpool：AvgPool2d（步长2，压缩特征图）
    - #### **残差层（layer1~layer4）**
      - 每层由多个`Bottleneck`块组成（数量由`layers`参数指定）
      - 每个`Bottleneck`块：
        - conv1：1x1卷积（降维，inplanes→planes）
        - bn1：BatchNorm2d
        - relu1：ReLU
        - conv2：3x3卷积（特征提取，planes→planes，填充1）
        - bn2：BatchNorm2d
        - relu2：ReLU
        - avgpool：AvgPool2d（步长>1时使用，否则为Identity）
        - conv3：1x1卷积（升维，planes→planes×4，expansion=4）
        - bn3：BatchNorm2d
        - downsample（可选）：当步长>1或通道不匹配时
          - AvgPool2d（下采样）
          - 1x1卷积（匹配通道）
          - BatchNorm2d
        - relu3：ReLU（残差连接后激活，out += identity）
    - #### **AttentionPool2d（注意力池化）**
      - positional_embedding：位置嵌入参数
      - k_proj：Linear（键投影）
      - q_proj：Linear（查询投影）
      - v_proj：Linear（值投影）
      - c_proj：Linear（输出投影）
      - 多头自注意力计算：以全局平均特征为查询，结合位置嵌入
  - ### **VisionTransformer（视觉Transformer）**
    - conv1：patch卷积（3通道→width通道， kernel=patch_size，步长=patch_size，分割图像为patch）
    - class_embedding：类嵌入参数（类似[CLS]标记）
    - positional_embedding：patch位置嵌入参数
    - ln_pre：LayerNorm（Transformer前归一化）
    - #### **transformer（Transformer编码器）**
      - 由多个`ResidualAttentionBlock`组成（数量由`layers`参数指定）
      - 每个`ResidualAttentionBlock`：
        - ln_1：LayerNorm（注意力前归一化）
        - attn：MultiheadAttention（多头自注意力）
        - ln_2：LayerNorm（前馈网络前归一化）
        - mlp：前馈网络
          - c_fc：Linear（升维，d_model→d_model×4）
          - gelu：QuickGELU（激活）
          - c_proj：Linear（降维，d_model×4→d_model）
    - ln_post：LayerNorm（Transformer后归一化）
    - proj：Linear（投影至输出维度，width→output_dim）

## **文本编码器**
- #### **transformer（Transformer编码器）**
  - 结构同VisionTransformer的transformer，由多个`ResidualAttentionBlock`组成
  - 注意力掩码：因果掩码（上三角矩阵，确保自回归性）
- token_embedding：Embedding（词嵌入，vocab_size→transformer_width）
- positional_embedding：位置嵌入参数（长度=context_length）
- ln_final：LayerNorm（Transformer后归一化）
- text_projection：Linear（文本特征投影至公共嵌入空间，transformer_width→embed_dim）

## **跨模态交互**
- 特征归一化：图像/文本特征均做L2归一化
- 温度系数：logit_scale（参数，控制相似度缩放，exp后使用）
- 相似度计算：图像特征 × 文本特征转置 × 温度系数，得到双向logits


