# 四种VLP模式
![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504250902589.png)

VSE、VSE++和SCAN属于(a)类型。

CLIP属于(b)类型。

ViLBERT、UNTER和Pixel-BERT属于(c)类型。

作者提出的ViLT属于(d)类型。
# Modality Interaction Schema（模态交互模式）

* **single-stream**(如BERT和UNITER)：对图像和文本concate然后进行交互操作
* **dual-stream**(如ViLBERT和LXMERT)：不对图像和文本concate然后进行交互操作

ViLT延用single-stream的交互方式，因为dual-stream会引入额外的计算量。

# Visual Embedding Schema（视觉嵌入模式）

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504250911383.png)

ViLT属于 Patch Projection。
# ViLT
## 模型架构

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202504250912685.png)

* 文本特征输入：将文本看成一个词序列，通过word embedding matrix转化成word embedding，然后和position embedding进行相加，最后和modal-type embedding进行concate。
* 图像特征输入：将图像切块看成一个图像块序列，通过linear projection转化成visual embedding，然后和postion embedding进行相加，最后和modal-type embedding进行concate。

其中word embedding和visual embedding通过可学习的modal-type embedding标志位来区分。

wrod embedding和visual embedding分别都嵌入了一个额外的可学习class embedding，方便和下游任务对接。

## 预训练目标

* **ImageText Matching**：随机以0.5的概率将文本对应的图片替换成不同的图片，然后对文本标志位对应输出使用一个线性的ITM head将输出feature映射成一个二值logits，用来判断图像文本是否匹配。
* **Masked Language Modeling**：MLM的目标是通过文本的上下文信息去预测masked的文本tokens。随机以0.15的概率mask掉tokens，然后文本输出接两层MLP预测mask掉的tokens。

另外ViLT还设计了一个**word patch alignment (WPA)** 来计算textual subset和visual subset的对齐分数。

另外ViLT还使用了**whole word masking**技巧。就是将句子中的整个word给mask掉，而不是只mask掉一个word中的子word。这样才能利用好图像的信息，而不是单靠文本序列进行预测。

## 结论

总体上来看基于patch projection的多模态方法速度优势非常大，但是ViLT整体上性能还是略低于region feature的方法。


# 其他相关

grid feature和region feature
[「AAAI2021」Image Captioning 图像描述生成，性能SoTA！ - 知乎](https://zhuanlan.zhihu.com/p/348220534#:~:text=region%E7%89%B9%E5%BE%81%E6%98%AF%E6%A3%80%E6%B5%8B%E5%87%BA%E6%9D%A5%E7%9A%84%E7%9B%AE%E6%A0%87%EF%BC%8C%E8%BF%99%E4%BA%9B%E7%89%B9%E5%BE%81%E7%9A%84%E8%AF%AD%E4%B9%89%E5%B1%82%E7%BA%A7%E7%9B%B8%E5%AF%B9%E8%BE%83%E9%AB%98%EF%BC%8C%E4%BD%86%E5%AE%83%E4%BB%AC%E6%9C%89%E4%B8%A4%E4%B8%AA%E7%BC%BA%E7%82%B9%EF%BC%8C%E4%B8%80%E6%98%AF%E5%9B%BE%E5%83%8F%E4%B8%AD%E9%9D%9E%E7%9B%AE%E6%A0%87%E7%9A%84%E5%8C%BA%E5%9F%9F%E4%BC%9A%E8%A2%AB%E5%BF%BD%E8%A7%86%EF%BC%88%E5%A6%82%E8%83%8C%E6%99%AF%E4%BF%A1%E6%81%AF%EF%BC%89%EF%BC%8C%E4%BA%8C%E6%98%AF%E5%A4%A7%E7%9B%AE%E6%A0%87%E7%9A%84%E5%B0%8F%E7%BB%86%E8%8A%82%E4%BC%9A%E8%A2%AB%E5%BF%BD%E8%A7%86%E3%80%82%20%E5%A6%82%E4%B8%8B%E5%9B%BE%E6%89%80%E7%A4%BA%E3%80%82,2.%20grid%E7%89%B9%E5%BE%81%E5%B0%B1%E6%98%AFFeature%20map%EF%BC%8C%E8%BF%99%E4%B8%AA%E7%89%B9%E5%BE%81%E7%9A%84%E8%AF%AD%E4%B9%89%E5%B1%82%E7%BA%A7%E7%9B%B8%E5%AF%B9%E8%BE%83%E4%BD%8E%EF%BC%8C%E5%BE%80%E5%BE%80%E4%B8%80%E5%A0%86%E7%BD%91%E6%A0%BC%E5%9C%A8%E4%B8%80%E8%B5%B7%E6%89%8D%E8%83%BD%E8%A6%86%E7%9B%96%E4%B8%80%E4%B8%AA%E7%9B%AE%E6%A0%87%EF%BC%8C%E4%BD%86%E4%BC%98%E7%82%B9%E6%98%AF%E5%AE%83%E8%83%BD%E8%A6%86%E7%9B%96%E6%95%B4%E5%BC%A0%E5%9B%BE%E7%89%87%EF%BC%8C%E5%90%8C%E6%97%B6%E4%B9%9F%E5%8C%85%E5%90%AB%E4%BA%86%E7%9B%AE%E6%A0%87%E7%9A%84%E7%BB%86%E8%8A%82%E4%BF%A1%E6%81%AF%E3%80%82)
提到一个几何对齐图的东西，将两种feature结合。