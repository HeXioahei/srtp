> github地址：[devinxzhang/MFuser](https://github.com/devinxzhang/MFuser)
> 论文地址：[2504.03193](https://arxiv.org/pdf/2504.03193)


*跨领域语义分割（Domain Generalized Semantic Segmentation，DGSS）*

**VFMs**（如 DINOv2）在捕捉**细粒度**特征方面表现优异，而 **VLMs**（如 CLIP）在**文本对齐**方面具有强大的**鲁棒性**，但在处理粗粒度信息时则存在一定困难。尽管它们在能力上互为补充，但**利用注意力机制将 VFMs 与 VLMs 有效融合仍具有挑战性**，因为更大量的 patch token 会加剧长序列建模的复杂性。

为了解决这一问题，我们提出了 **MFuser**——一种基于 **Mamba** 的新型融合框架，能够高效整合 VFMs 与 VLMs 的优势，并在序列长度上保持**线性扩展性**。MFuser 包含两个关键模块：
- **MVFuser**：一个协同适配器（co-adapter），通过捕捉时序与空间动态，实现对两个模型的联合微调；
- **MTEnhancer**：一个融合注意力机制与 Mamba 的模块，通过引入图像先验来优化文本嵌入表示。

我们的方法在实现精准的特征定位与强文本对齐能力的同时，并未带来显著的计算开销。大量实验证明，MFuser 在多个基准任务上显著优于当前最先进的 DGSS 方法：在合成到真实场景（synthetic-to-real）上取得 **68.20 mIoU**，在真实到真实场景（real-to-real）上取得 **71.87 mIoU** 的优异成绩。

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202505081102381.png)
