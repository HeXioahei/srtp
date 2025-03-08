[【多模态大模型】llava系列：llava、llava1.5、llava-next - 知乎](https://zhuanlan.zhihu.com/p/695100288)

# llava

亮点：指令跟随数据集的构造，多模态大模型的设计

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202503081612615.png)

# llava 1.5

结构改进。数据集更丰富了，且有更多的处理策略

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202503081619767.png)

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202503081620703.png)


![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202503081623009.png)

**主要的限制和改进**：使用全图补丁，增加了训练时间，visual resamplers（如Qwen-VL、InstructBLIP、BLIP2中使用的Q-former）减少了visual patchs in LLMs，但是他们当前不能实现收敛，简单有效的visual resamplers的发展可以为未来加大指令微调多模态模型的尺度铺平道路

# llava 1.6 （llava-next）