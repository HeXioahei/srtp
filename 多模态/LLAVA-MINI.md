**LLAVA-MINI: EFFICIENT IMAGE AND VIDEO LARGE MULTIMODAL MODELS WITH ONE VISION TOKEN (ICLR 2025)**

**方法：** 这篇文章提出了LLaVA-Mini，一种**仅需1个视觉token**的高效多模态大模型，解决了现有视觉语言模型**因大量视觉token导致计算开销大、延迟高的问题**，尤其在**处理高分辨率图像和长视频**时显著提升效率。

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504060930396.png)

**创新点：**

- 提出**模态预融合机制**，在LLM输入前将**视觉信息提前融合到文本token**中，突破传统方法中视觉token必须在LLM内部逐层交互的限制。
    
- 设计**极简视觉token压缩模块**，通过**可学习的压缩查询**实现**576:1的视觉信息压缩率**，大幅减少LLM处理的token数量。
    
- 首次实现**统一高效框架**，支持**标准/高分辨率图像和长视频（超万帧）处理**，在计算、内存和延迟上实现跨模态场景优化。

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504060930885.png)
