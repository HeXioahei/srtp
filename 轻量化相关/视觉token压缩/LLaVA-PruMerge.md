[[MLLM-算法推荐-2024.3.27] LLaVA-PruMerge 你应该选择你需要的 - 知乎](https://zhuanlan.zhihu.com/p/689233936)
（含论文和代码地址）

整个方法和 llava（[【多模态大模型】llava系列：llava、llava1.5、llava-next - 知乎](https://zhuanlan.zhihu.com/p/695100288)） 一致，只不过额外在 project 层前面插入了一个 token 修剪合并模块。主要有两个步骤：
* 采取用于异常值检测的 [四分位范围](https://zhida.zhihu.com/search?content_id=241293581&content_type=Article&match_order=1&q=%E5%9B%9B%E5%88%86%E4%BD%8D%E8%8C%83%E5%9B%B4&zhida_source=entity) (IQR) 方法来选择重要的token。
* 对非重要的token进行以重要token为中心的聚类处理，而不是无脑丢掉。

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202503081739112.png)

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202503081737656.png)


