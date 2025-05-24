# 模型架构

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202505221746954.png)

* Vision Encoder：直接调用CLIP的最小模型，得到最后一层的隐藏状态。
* Vision Mapping：对vision encoder的输出特征进行投影，使得其维度与文本token的维度一致，方便后续嵌入到文本token中。
* 红色粗箭头：vision embedding 替换拿掉文本token矩阵中的image占位符@，以此来嵌入。（至于怎么寻找到对应的位置，我看代码里算法好像挺复杂的，我没怎么看懂。）
	* 流程图：
		![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202505241726433.png)
	* 代码：[[#^code1]]
* Tokenizer Encoder：调用transformers库。
* Transformer Layer：
	* GQA（group query attention）：RoPE位置编码，kv_cache存储kv。
	* FFN（feed forward network）：SiLU激活，dropout防过拟合。
* Tokenizer Decoder：调用transformer库。
* loss：交叉熵损失。
# 轻量化的有效因素

* 模型架构上：
	* 轻量级CLIP模型（vit-base-patch16）：最小的clip模型，产生的token数量少，token维度也小，为512或768。
	* 精简的Transformer Layer设计：
		* 层数少，仅8层。
		* 用GQA，kv_cache，注意力头少。
		* FFN较为精简。
		* RMSNorm：与LayerNorm相比，RMSNorm的计算更为简便，前者需要遍历两边样本分别求出均值和方差，而后者只需要一次遍历求出均方差。
	* 简单高效的投影器设计：
		* 用MLP，而非Cross-Attention机制。这与LLaVA-1的设计思路一致，但MiniMind-V通过参数精简实现更高效率。
* 数据集上：
	* 图像的size小，仅为128 * 128。
* 训练策略上：
	* 预训练：4 epoch，微调：2 epoch
	* 分布式训练（复现时没用上）
# 数据集

* 预训练：
	* 59万张图片
	* 格式：```
		{
		  "conversations": [
			{
			  "role": "user",
			  "content": "提供给定图像的简要描述。\n<image>"
			},
			{
			  "role": "assistant",
			  "content": "橄榄油是自由使用的健康成分。"
			}
		  ],
		  "image": "GCC_train_002582585.jpg"
		}
* 微调：
	* 15万张图片
	* 格式：```
		{
		  "conversations": [
		    {
		      "role": "user",
		      "content": "闹钟的位置对睡眠质量有什么影响？<image>"
		    },
		    {
		      "role": "assistant",
		      "content": "把数字闹钟放在床头柜..."
		    }
		  ],
		  "image": "train-00000-of-00001_image_0_0.jpg"
		}

# 附

```python

def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):

        """

        将视觉投影结果合并到隐藏状态中。

  

        参数:

            tokens (torch.Tensor): 输入的 token 序列。

            h (torch.Tensor): 隐藏状态。

            vision_tensors (torch.Tensor): 视觉张量，默认为 None。

            seqlen (int): 序列长度，默认为 512。

  

        返回:

            torch.Tensor: 合并后的隐藏状态。

        """

        def find_indices(tokens, image_ids):

            """

            在 token 序列中查找图像 token 的位置。

  

            参数:

                tokens (torch.Tensor): 输入的 token 序列。

                image_ids (list): 图像对应的 token ID 列表。

  

            返回:

                dict: 包含每个批次中图像 token 起始和结束位置的字典，如果未找到则返回 None。

            """

            # 将图像对应的 token ID 列表转换为张量，并将其放置到与输入 tokens 相同的设备上

            image_ids_tensor = torch.tensor(image_ids).to(tokens.device)

            # 获取图像对应的 token ID 列表的长度

            len_image_ids = len(image_ids)

            # 如果图像 token ID 列表的长度大于输入 token 序列的长度，无法匹配，直接返回 None

            if len_image_ids > tokens.size(1):

                return None

            # 使用 unfold 方法在输入 token 序列的第二个维度上滑动窗口，窗口大小为 len_image_ids，步长为 1

            # 这样可以得到所有可能的子序列，用于后续匹配

            tokens_view = tokens.unfold(1, len_image_ids, 1)

            # 将滑动窗口得到的子序列与图像 token ID 张量进行逐元素比较

            # 并在最后一个维度上使用 all 方法判断是否所有元素都匹配

            matches = (tokens_view == image_ids_tensor).all(dim=2)

            # 遍历每个批次，找到每个批次中匹配的子序列的起始索引

            # 对于每个匹配的起始索引，计算结束索引（起始索引 + 图像 token ID 列表长度 - 1）

            # 若存在匹配项，则返回包含起始和结束索引的字典，否则返回 None

            return {

                batch_idx: [(idx.item(), idx.item() + len_image_ids - 1) for idx in

                            matches[batch_idx].nonzero(as_tuple=True)[0]]

                for batch_idx in range(tokens.size(0)) if matches[batch_idx].any()

            } or None

  

        # 调用 find_indices 函数，在输入的 token 序列中查找图像 token 的位置

        image_indices = find_indices(tokens, self.params.image_ids)

        # 若视觉张量存在且找到了图像 token 的位置，则进行视觉投影合并操作

        if vision_tensors is not None and image_indices:

            # 对视觉张量进行投影操作

            vision_proj = self.vision_proj(vision_tensors)

            # 若投影结果的维度为 3，在第 0 维增加一个维度

            if len(vision_proj.shape) == 3:

                vision_proj = vision_proj.unsqueeze(0)

            new_h = []

            # 遍历每个批次的隐藏状态

            for i in range(h.size(0)):

                # 若当前批次存在图像 token

                if i in image_indices:

                    h_i = h[i]

                    img_idx = 0

                    # 遍历当前批次中每个图像 token 的起始和结束位置

                    for start_idx, end_idx in image_indices[i]:

                        # 若还有未处理的视觉投影结果

                        if img_idx < vision_proj.size(1):

                            # 将视觉投影结果插入到隐藏状态中图像 token 的位置，并截取到指定序列长度

                            h_i = torch.cat((h_i[:start_idx], vision_proj[i][img_idx], h_i[end_idx + 1:]), dim=0)[:seqlen]

                            img_idx += 1

                    new_h.append(h_i)

                else:

                    # 若当前批次不存在图像 token，直接添加原隐藏状态

                    new_h.append(h[i])

            # 将处理后的隐藏状态在第 0 维堆叠

            return torch.stack(new_h, dim=0)

        # 若视觉张量不存在或未找到图像 token 的位置，直接返回原隐藏状态

        return h

```
^code1