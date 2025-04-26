![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504051129634.png)

Image Encoder那里是一个[DPT](深度学习中的一些概念#dpt)的结构。C一般是512或768。F是文本和图片结合后的特征结果，然后与ground truth进行loss。这是一个**有监督学习**的过程。这里的Text Encoder的参数是冻住的，直接用CLIP模型训练好的结果，不会更新。因为这里用于语义分割预训练的数据集很小（相对于CLIP预训练的数据来说），只有20万张图片，如果在这20万张的基础上对参数进行调整，可能会把原本在CLIP上训练好的泛化效果给带偏掉。而Image Encoder这边既可以用CLIP预训练好的参数，也可以用vit的一些模型的参数，后者的效果反而更好。

在得到F后，作者还增加了一个Spatial Regularization Blocks模块，其实就是一些conv或者depthwise conv，这可能是为了增加一些可学习的参数，让视觉和文本特征能够进一步的融合。加2个blocks性能最好，当达到4个的时候性能就崩了，而作者并没有解释为什么。

**其实，如果不看文本的那条分支，只看下面那条图片的分支的话，它的原理就和传统的语义分割算法模型没有区别了。所以其实其最大的区别就是将文本特征也加进来了，最后将融合的特征与ground truth做loss。所以文本并不是被当作监督信号来使用，还是依赖手工标注的mask作为监督信号。**

评估时，将一个数据集平均分成四份，然后做few-shot和zero-shot的评估。

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504051125451.png)


