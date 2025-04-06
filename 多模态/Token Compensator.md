**Token Compensator: Altering Inference Cost of Vision Transformer without Re-Tuning**

**方法：** 这篇文章解决了Vision Transformers（ViTs）中token压缩方法在训练和推理阶段**压缩程度不匹配**时导致的性能显著下降问题，提出了一种**无需重新调优的动态调整推理成本的方法**。

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504060934559.png)

**创新点：**

- **参数高效的Token Compensator（ToCom）**：通过**轻量级插件**建模不同压缩程度模型间的**参数差异**，实现跨压缩程度的性能补偿。
    
- **模型算术框架**：利用**LoRA模块**的**叠加和逆向**操作，支持任意压缩程度的动态调整，避免为每个压缩程度单独训练模型。
    
- **跨任务通用的自蒸馏预训练**：在预训练阶段通过**随机分配压缩程度**的自蒸馏策略，使ToCom能直接应用于任意下游任务，无需额外适配。

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504060936623.png)
