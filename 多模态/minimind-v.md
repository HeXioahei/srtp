![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202505221746954.png)

# 轻量化的有效因素

* RMSNorm：与layernorm相比，rmsnorm的计算更为简便，前者需要遍历两边样本分别求出均值和方差，而后者只需要一次遍历求出均方差。


# 特征融合的有效因素

* RoPE：提高了对图像中相对位置信息的提取，有利于提高对图像的细粒度理解水平。