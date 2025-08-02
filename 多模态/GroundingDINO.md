* GroundingDINO
	* backbone
	* transformer

**文本特征处理**：tokenlizer（由get_tokenlizer获得） -> bert（由get_pretrained_language_model获得） -> feat_map（一个线性投影层） -> 

**图像特征处理**：backbone（由set_image_tensor获得，返回多层特征，多个尺度的特征） -> input_proj（一个线性投影层）

transformer（即“Feature Enhancer”+“Language-guide Query Selection”+“Cross-Modality Decoder”） -> layer_bbox_embed（对每一层添加一个边界框嵌入层） -> class_embed_layerlist（为每个解码器层创建一个类别嵌入）




![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202508011635299.png)
