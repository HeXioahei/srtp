shortGPT：设计一个BI板块来计算每个层的重要性，根据计算出的重要性直接用”层删除“这种很直接的剪枝方式删除某些层，与原来的大模型相比可能会造成一些性能损失，但可以使得模型更轻量，实验证明效果还行。目前值用在了LLM上，尚未有VLM上的示例。如果可以的话，我们到时候也可以试试，说不定可以作为一个小小的创新点，虽然不一定能成功。

博客：[ShortGPT：删除冗余层让大模型运行更快 - 知乎](https://zhuanlan.zhihu.com/p/686170824)
论文：[[2403.03853] ShortGPT: Layers in Large Language Models are More Redundant Than You Expect](https://arxiv.org/abs/2403.03853)
github：[RayVentura/ShortGPT: 🚀🎬 ShortGPT - Experimental AI framework for youtube shorts / tiktok channel automation](https://github.com/RayVentura/ShortGPT?tab=readme-ov-file)

![06c87896013b857973b42eb6933afcf.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202503061141155.png)

![71d0f3d07b2095b502d48f17da57a85.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202503061141435.png)


![[轻量化相关/三个核心模块的优化/轻量化LLM/剪枝/相关引用/BI的设计.pdf]]
![[目前是否有VLM模型用了shortGPT中“层删除”这种很直接的剪枝技术.pdf]]
