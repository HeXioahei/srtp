> 代码：[jiaqihuang01/DETRIS: [AAAI-2025] The official code of Densely Connected Parameter-Efficient Tuning for Referring Image Segmentation](https://github.com/jiaqihuang01/DETRIS)
> 博客：[(aaai2025) Densely Connected Parameter-Efficient Tuning for Referring Image Segmentation - 知乎](https://zhuanlan.zhihu.com/p/1895502417946202721)


在大模型时代，各种 CV/NLP 的任务都在拥抱预训练+微调的范式，但是随着模型参数规模越来越大，在下游任务数据集上做全量微调的成本也越来越高，目前普遍采用的是 [Parameter-Efficient Tuning](https://zhida.zhihu.com/search?content_id=256439144&content_type=Article&match_order=1&q=Parameter-Efficient+Tuning&zhida_source=entity)（高效参数微调）。目前的PET主要包括三类：

- **[Adapter](https://zhida.zhihu.com/search?content_id=256439144&content_type=Article&match_order=1&q=Adapter&zhida_source=entity)：** 将较小的神经网络模块(称为adapter)插入到预训练模型的中间层，微调时只训练adapter参数
- **[Prefix Tunning](https://zhida.zhihu.com/search?content_id=256439144&content_type=Article&match_order=1&q=Prefix+Tunning&zhida_source=entity)：** 在模型的输入或隐层添加多个可学习的前缀tokens (不对应真实tokens)，微调时只训练前缀参数。相关工作：Prompt Tuning、P-Tuning
- **[LoRA](https://zhida.zhihu.com/search?content_id=256439144&content_type=Article&match_order=1&q=LoRA&zhida_source=entity)：** 通过学习小参数的低秩矩阵来近似模型权重矩阵的参数更新，微调时只训练低秩矩阵参数。

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202505142324435.png)

回到这个论文，作者解决的是 指代分割任务（Referring Image Segmentation）。

当前将[CLIP](https://zhida.zhihu.com/search?content_id=256439144&content_type=Article&match_order=1&q=CLIP&zhida_source=entity)应用于指代分割的方法主要面临两个问题：(1) 这些方法主要依赖于来自**骨干网络**的早期融合，对于**全局特征利用不足**。（2) 现有的**PET模块**对于**多尺度信息利用不足**。

为了解决上面两个问题，作者提出了[DenseAligner](https://zhida.zhihu.com/search?content_id=256439144&content_type=Article&match_order=1&q=DenseAligner&zhida_source=entity)的适配器，有两个创新点：

- dense mixture of convolution module, 可以从中间层获取语义特征
- cross-aligner module，促进视觉和文本特征间的信息交换

模型的整体架构如下图所示，图像标码器使用[DINOv2](https://zhida.zhihu.com/search?content_id=256439144&content_type=Article&match_order=1&q=DINOv2&zhida_source=entity)，文本编码器使用CLIP。可以看出，主要使用DenseAligner （DA）微调图像特征，使用 TextAdapter （TA）微调文本特征。DenseAligner 是论文的主要创新点，因为DINOv2没有考虑文本特征，所以 DA 还注入了文本特征提升多模态能力。

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202505142325403.png)

DA的结构如下图所示，图像特征使用多尺度卷积（[D-MoC](https://zhida.zhihu.com/search?content_id=256439144&content_type=Article&match_order=1&q=D-MoC&zhida_source=entity)）处理，同时还使用cross attention融合文本特征。D-MoC如下图所示，主要是密集的多尺度卷积。在TextAdaptor中，直接使用 D-MoC 提取特征。

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202505142325506.png)


