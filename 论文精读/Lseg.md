![7494a199-479a-4b66-99d2-7f545c5a45d5](file:///C:/Users/Lenovo/Pictures/Typedown/7494a199-479a-4b66-99d2-7f545c5a45d5.png)



Image Encoder那里是一个DPT的结构。C一般是512或768。F是文本和图片结合后的特征结果，然后与ground truth进行loss。这是一个有监督学习的过程。这里的Text Encoder的参数是冻住的，直接用CLIP模型训练好的结果，不会更新。因为这里用于语义分割预训练的数据集很小（相对于CLIP预训练的数据来说），只有20万张图片，如果在这20万张的基础上对参数进行调整，可能会把原本在CLIP上训练好的泛化效果给带偏掉。而Image Encoder这边既可以用CLIP预训练好的参数，也可以用vit的一些模型的参数，后者的效果反而更好。



# dpt

DPT（Dense Prediction Transformer）是一种基于**Transformer架构**的深度学习模型，专为**密集预测任务**（Dense Prediction Tasks）设计，例如图像分割、深度估计、语义分割等。它结合了Transformer的全局上下文建模能力和卷积网络（CNN）的局部特征提取优势，在像素级预测任务中表现优异。

---

### **DPT的核心原理**

1. **Transformer骨干网络**：
   
   - 输入图像被分割为多个图像块（patches），每个块通过线性嵌入转换为序列，输入到Transformer编码器中。
   - Transformer通过**自注意力机制**（Self-Attention）捕捉图像全局的上下文信息，弥补传统CNN局部感受野的不足。

2. **多尺度特征融合**：
   
   - DPT通过不同层级的Transformer特征（浅层细节+深层语义）进行融合，生成高分辨率的密集预测结果。
   - 使用**特征金字塔**（Feature Pyramid）或**解码器模块**逐步上采样，恢复空间细节。

3. **轻量化设计**：
   
   - 通过减少Transformer层数或使用混合结构（如CNN+Transformer），平衡计算效率和性能。

---

### **DPT的典型结构**

以图像分割任务为例，DPT的流程如下：

1. **图像分块**：将图像分割为固定大小（如16×16）的块，展开为序列。
2. **Transformer编码器**：通过多层Transformer提取全局特征。
3. **多级特征提取**：从不同Transformer层获取多尺度特征图。
4. **解码器融合**：通过跳跃连接（Skip Connection）和上采样逐步融合多尺度特征，生成像素级预测结果。

---

### **DPT的优势**

1. **全局上下文建模**：自注意力机制能有效捕捉长距离依赖，适合需要全局信息的任务（如场景分割）。
2. **多尺度适应性**：通过特征融合保留细节和语义，提升密集预测的精度。
3. **灵活性**：可适配多种任务（深度估计、语义分割、图像修复等）。

---

### **应用场景**

- **深度估计**：从单张图像预测每个像素的深度值（如DPT在[MiDaS](https://github.com/isl-org/MiDaS)中的应用）。
- **语义分割**：对图像中的每个像素分类（如医学图像分析）。
- **图像增强**：超分辨率、去噪等需要密集输出的任务。

---

### **与CNN的对比**

| **特性** | **DPT**         | **传统CNN** |
| ------ | --------------- | --------- |
| 感受野    | 全局（自注意力）        | 局部（卷积核）   |
| 计算效率   | 较高内存需求（随序列长度增加） | 较低内存需求    |
| 任务适应性  | 适合长距离依赖任务       | 适合局部模式提取  |

---

### **代码示例（PyTorch）**

```python
import torch
from transformers import DPTForDepthEstimation, DPTImageProcessor

# 加载预训练模型和处理器
processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

# 处理输入图像
image = torch.randn(1, 3, 384, 384)  # 假设输入为RGB图像
inputs = processor(images=image, return_tensors="pt")

# 推理
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth  # 输出的深度图
```

---

### **总结**

DPT通过Transformer的全局建模能力与多尺度特征融合，显著提升了密集预测任务的性能。尽管计算成本较高，但在需要精确像素级预测的场景（如自动驾驶、医学影像）中，DPT已成为重要的技术选择。
