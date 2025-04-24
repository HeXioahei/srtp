博客推荐：[BLIP系列文章小结（BLIP, BLIP-2, InstructBLIP） - 知乎](https://zhuanlan.zhihu.com/p/664011842)
# 1. BLIP
## 1.1 主要创新点

提供了一种对 web datasets 的 caption 进行降噪的方法。

## 1.2 方法
### 1.2.1 模型架构

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504242049115.png)

BLIP多模态架构为[双塔架构](https://www.bing.com/search?q=%E5%8F%8C%E5%A1%94%E6%9E%B6%E6%9E%84&FORM=SSQNT1&adppc=EdgeStart&PC=LCTS&mkt=zh-CN)。论文中用3个vision language pretraining(VLP) task来激发模型的多模态能力。

### 1.2.2 三个VLP

#### 1.2.2.1 Image-Text Contrastive Loss (ITC)

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504242102332.png)

**核心思想**：和CLIP的对比学习原理类似。

由此训练出 Image Encoder。

[Co-Attention、Self-Attention 和 Bi-Attention](https://blog.csdn.net/qq_40133431/article/details/137018130)

#### 1.2.2.2 Image-text matching Loss (ITM)

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504242126685.png)

**核心思想**：给定一组图像-文本向量对。其训练目标为预测 某个图像-文本对是否来自同一个pair。是为1，否则为0。

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504242131033.png)

随后用cross-entropy计算损失。

由此训练出 Image-grounded Text Encoder。

#### 1.2.2.3 Language modeling loss（LM）

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504242137126.png)

**核心思想**：与NLP的LM类似，就是根据前面的词来预测下一个词，不同的是这里同时将image-embedding引入到上下文信息。

最大化自回归序列的似然概率进行训练：

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504242139204.png)

训练完成后得到 image-ground text decoder。

[Causal Attention](https://blog.csdn.net/qinduohao333/article/details/133875973)

### 1.2.3 boostrapping caption（核心）

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504242143382.png)

**局限性**：当数据和模型的规模增大时，boostrapping caption 的效果就没那么好了。
# 2. BLIP-2

## 2.1 主要创新点

从模态对齐、高效训练两个方向对VLP做出优化。
* **（核心）** 在模态对齐上：**QFormer**（querying transformer），建立图像-文本的桥梁。
* 在高效多模态训练上：**二阶段预训练范式**，将目前的视觉backbone与LLM模型链接起来。

## 2.2 方法
### 2.2.1 模型架构

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504242151856.png)
![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504242151253.png)

QFormer内部包含两个transformer子模块：image-transofmrer 和 text-transformer。image-transformer比text-transformer多了一个cross-attention层，这两个transformer共享Self-Attention参数。定义了learning query，通过训练将与文本对齐后的图片的信息融入到learning query中。

**shape变化如下**：

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504242200628.png)

### 2.2.2 两个预训练阶段

#### 2.2.2.1 第一个阶段
用于**对齐多模态表征**，主要通过Image-Text Contrastive Loss (ITC)、 Image-text matching (ITM)、Image-grounded Text Generation（ITG）3个任务的联合训练来实现。

#### 2.2.2.2 第二个阶段
用于**让LLM理解第一个阶段产生的soft visual prompt的语义**，从而借助LLM强大的知识库实现视觉推理等更为复杂的任务，主要通过language modeling（LM）任务的训练来实现。

BLIP2的预训练任务同样用了BLIP提出的boostrapping caption（也称为CapFilt method）技术。
