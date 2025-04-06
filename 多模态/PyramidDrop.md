**PyramidDrop: Accelerating Your Large Vision-Language Models viaPyramid Visual Redundancy Reduction (CVP R2025)**

**方法：** 这篇文章针对大型视觉语言模型(LVLMs)处理高分辨率图像时存在的**计算成本爆炸性增长问题**，提出了一种创新的**视觉冗余分层削减策略**，在保证多模态性能的同时实现了训练和推理效率的突破性提升。

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504060921578.png)

**创新点：**

- 首次通过实证研究发现LVLMs**浅层需要完整视觉token**，而**深层存在显著冗余**的层级化特征，颠覆了传统均匀压缩token的认知。
    
- 提出**金字塔式渐进token丢弃机制**，通过**多阶段动态调整token保留比例**，实现冗余消除与关键信息保留的精准平衡。
    
- 设计**轻量级注意力相似度计算模块**，在仅**增加O(n)** 时间复杂度的前提下，实现**token重要性评估**与**动态剪枝**的端到端优化。

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504060922604.png)

