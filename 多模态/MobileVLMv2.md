> [美团/浙大等联合提出MobileVLM | 骁龙888就可以实时运行的多模态大模型，边缘端多模态大模型之战打响了！！！ - 知乎](https://zhuanlan.zhihu.com/p/675392936)
> [轻量化VLM探索：MobileVLM V2 - 知乎](https://zhuanlan.zhihu.com/p/681878699)
> [多模态小模型：MobileVLM V2：为视觉语言模型带来更快更强的基准 - 知乎](https://zhuanlan.zhihu.com/p/690082320)

> 代码：[https://github.com/Meituan-AutoML/MobileVLM](https://link.zhihu.com/?target=https%3A//github.com/Meituan-AutoML/MobileVLM)
> 模型：[mtgv (team of GV, Meituan)](https://link.zhihu.com/?target=https%3A//huggingface.co/mtgv)
> 论文地址：[https://arxiv.org/abs/2402.0376](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2402.03766)

参数量级：1-10B

主要亮点在于投影器的设计。

v2好于v1关键在于它对 Lightweight Downsample Projector（LDP）的改进：LDPv2。使得投影器有更少的参数量。

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202505252132652.png)

其中，MobileLLaMA是MobileVLM中提出的，采用了和LLaMA2相同的tokenizer，并对其进行蒸馏得到。在MobileVLM V2里采用了**MobileLLaMA-1.4B-Chat**和**MobileLLaMA-2.7B-Chat**。

**重点：LDPv2（Lightweight Downsample Projector）
1. 先通过两层point-wise卷积层（实际代码中直接用两个Linear层+一个GeLU层对输入的 (b, num_tokens, c) 的 x 进行投影），使图像token数和LLM的feature维度对齐，再用Avg Pooling进行压缩（代码中，将其展成 (b, c, h, h) 再用 AdaptiveAvgPool2d 进行压缩）
2. 压缩后通过一个PEG（Positional Encoding Generator） + skip connection 得到输出
3. token数从576减少到144个
4. 这一步节省了99.8%的projector参数（对整个速度有轻微提升），且所有算子都是deploy-friendly的
![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202505252136872.png)

