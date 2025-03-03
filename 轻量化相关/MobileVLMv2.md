[轻量化VLM探索：MobileVLM V2 - 知乎](https://zhuanlan.zhihu.com/p/681878699)
[多模态小模型：MobileVLM V2：为视觉语言模型带来更快更强的基准 - 知乎](https://zhuanlan.zhihu.com/p/690082320)

> 代码：[https://github.com/Meituan-AutoML/MobileVLM](https://link.zhihu.com/?target=https%3A//github.com/Meituan-AutoML/MobileVLM)
> 模型：[mtgv (team of GV, Meituan)](https://link.zhihu.com/?target=https%3A//huggingface.co/mtgv)
> 论文地址：[https://arxiv.org/abs/2402.0376](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2402.03766)

关键在于它对 Lightweight Downsample Projector（LDP）的改进：LDPv2。使得投影器有更少的参数量。主要用到了Depthwise卷积与Pointwise卷积以及残差连接。

# 