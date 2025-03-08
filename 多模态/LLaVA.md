[【多模态大模型】llava系列：llava、llava1.5、llava-next - 知乎](https://zhuanlan.zhihu.com/p/695100288)

# llava

亮点：指令跟随数据集的构造，多模态大模型的设计

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202503081612615.png)

# llava 1.5

结构改进。数据集更丰富了，且有更多的处理策略

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202503081619767.png)

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202503081627635.png)


![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202503081620703.png)


![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202503081623009.png)

**主要的限制和改进**：使用全图补丁，增加了训练时间，visual resamplers（如Qwen-VL、InstructBLIP、BLIP2中使用的Q-former）减少了visual patchs in LLMs，但是他们当前不能实现收敛，简单有效的visual resamplers的发展可以为未来加大指令微调多模态模型的尺度铺平道路

# llava 1.6 （llava-next）

**主要改进**：
* **增加图片输入的分辨率**到总像素的4倍，减少幻觉，允许模型抓取更多的视觉细节。
* **更好的视觉推理能力和OCR能力**，通过一个改进的视觉指令调整数据混合
- **对于更多场景的更好的视觉对话**，覆盖不同的应用。更好的世界知识和逻辑推理
- 使用**SGLang**实现**有效的部署和推理（[https://github.com/sgl-project/sglang](https://link.zhihu.com/?target=https%3A//github.com/sgl-project/sglang)）**

