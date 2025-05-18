> 论文题目：ROS-SAM: High-Quality Interactive Segmentation for Remote Sensing Moving Object
> 
> 论文地址：https://arxiv.org/pdf/2503.12006
> 
> 论文代码：https://github.com/ShanZard/ROS-SAM
> 
> 发表时间：2025.3.15


# 摘要

遥感视频数据目标尺寸小、特征模糊以及泛化能力有限，本文提出ROS-SAM

1. 基于[[LoRA]]微调SAM
    
2. [^1]增强深度网络层，以提高提取特征的**可区分性**，从而减少误分类。
    
3. [^2]在Mask Decoder中**整合全局上下文与局部边界细节**，以生成高质量的分割掩码。
    
4. 此外，本文设计了数据管道（[[data pipeline]]），以确保模型在训练期间能够更好地处理不同尺度的目标，同时在推理时专注于高质量预测。
    

在遥感视频数据集上的实验表明，重新设计的数据管道使交并比（IoU）提高了6%，而ROS-SAM则使IoU提高了13%。

[^1]: 深层网络是提取特征的关键部分，深层网络的设计将影响特征提取的效果。

[^2]: 也是提到了整合全局和局部信息，这在很多论文中都有提及，只是整合的地点可能不太一样。

# 背景
- **遥感视频目标特性与标注难题**：遥感视频中的运动目标尺寸小、特征模糊且密度高，逐帧标注成本高昂，导致高质量像素级标注稀缺，限制了相关算法的训练和性能提升。
    
- **通用视觉模型的遥感适应性不足**：通用视觉模型（如SAM）在遥感任务中面临显著挑战：无法准确处理遥感目标的方向和特征（如区分船只与海浪），且生成的掩码边界粗糙、易碎片化，难以满足高质量分割需求。
    
- **遥感图像处理的技术瓶颈**：遥感图像尺寸大但目标小，而现有模型（如SAM）要求固定分辨率输入，导致下采样时目标丢失。此外，现有训练和推理流程无法有效处理[多尺度目标](../其他笔记/遥感领域的多尺度目标.md)，难以适应遥感图像的特殊性。

# 方法

## 模型总览

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202505181006936.png)

## Fine-tuning the image encoder

在image encoder的每个Transformer层的注意力计算阶段引入LoRA

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202505181011237.png)

*MatMul是指矩阵乘法*
## Mask decoder and high-quality mask decoder

- **采用HQ-SAM掩码解码器**：整合高级别目标上下文和低级别边缘信息，生成高质量掩码。
    
- **利用HQSeg-44K数据集**：提供丰富的边缘细节先验知识，增强模型对细粒度特征的捕捉能力。

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202505181011040.png)

# 实验
## 数据集

SAT-MTB数据集：唯一包含遥感视频**目标跟踪、检测**和**分割**任务的数据集。

1. **数据集内容**：包含249个视频，总计约50,000帧，涵盖四种常见地面目标：飞机、汽车、船只和火车。
    
2. **汽车掩码说明**：汽车通常只占据约10个像素且形状均匀，与检测框高度匹配，因此无需生成掩码。
    
3. **帧采样策略**：为缓解视频中**高帧相似性问题**，从每个视频中随机抽取**1/4的帧**构建最终数据集。

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202505181023267.png)

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202505181023409.png)

## 实验结果
![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202505181025031.png)

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202505181025983.png)

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202505181025868.png)

## 消融实验

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202505181028629.png)

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202505181029991.png)

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202505181029745.png)

![image.png](https://youki-1330066034.cos.ap-guangzhou.myqcloud.com/machine-learning/202505181029330.png)

==*不太明白这里的Last layer指的是什么。*==
# 结论
本文提出了**ROS-SAM**模型，该模型基于SAM，并对**数据管道**、**图像编码器**和**掩码解码器**三个关键组件进行了改进，专门用于提升遥感运动目标分割中的目标掩码预测质量。

通过消融研究和对比实验，验证了这些改进显著提升了SAM在**遥感视频运动目标（RSVMO）任务**中的性能。

通过跨数据集实验验证了ROS-SAM的泛化能力，预期其将成为推进遥感视频分析中**细粒度分割**任务的强大工具。
# loss的计算

## 代码

```python

import torch
from torch.nn import functional as F
from typing import List, Optional
import utils.misc as misc

def point_sample(input, point_coords, **kwargs):
    """
    对 torch.nn.functional.grid_sample 进行封装，以支持 3D 的 point_coords 张量。
    与 torch.nn.functional.grid_sample 不同，它假设 `point_coords` 位于 [0, 1] x [0, 1] 的正方形内。
    Args:
        input (Tensor): 形状为 (N, C, H, W) 的张量，包含 H x W 网格上的特征图。
        point_coords (Tensor): 形状为 (N, P, 2) 或 (N, Hgrid, Wgrid, 2) 的张量，包含
                             [0, 1] x [0, 1] 归一化的点坐标。
    Returns:
        output (Tensor): 形状为 (N, C, P) 或 (N, C, Hgrid, Wgrid) 的张量，包含
                         `point_coords` 中点的特征。这些特征通过对 `input` 进行双线性插值获得，
                         方式与 torch.nn.functional.grid_sample 相同。
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2) # 如果 point_coords 是 (N, P, 2) 的形状，则添加一个维度变为 (N, P, 1, 2) 以适应 grid_sample
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs) # 使用 grid_sample 进行采样，将 [0, 1] 范围的坐标转换为 [-1, 1] 范围
    if add_dim:
        output = output.squeeze(3) # 如果之前添加了维度，则将其移除，恢复形状为 (N, C, P)
    return output

def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    torch.cat 的高效版本，如果列表中只有一个元素，则避免复制。
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0] # 如果只有一个张量，直接返回，避免不必要的连接操作
    return torch.cat(tensors, dim) # 否则，沿指定维度连接张量列表

def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    """
    基于不确定性在 [0, 1] x [0, 1] 坐标空间中采样点。
    每个点的不确定性通过 `uncertainty_func` 函数计算，该函数以点的 logit 预测作为输入。
    详见 PointRend 论文。
    Args:
        coarse_logits (Tensor): 形状为 (N, C, Hmask, Wmask) 或 (N, 1, Hmask, Wmask) 的张量，
                                 表示类别特定的或类别无关的粗略预测。
        uncertainty_func: 一个函数，它接受形状为 (N, C, P) 或 (N, 1, P) 的张量（包含 P 个点的 logit 预测），
                          并返回形状为 (N, 1, P) 的张量（包含这些点的不确定性）。
        num_points (int): 要采样的点 P 的数量。
        oversample_ratio (int): 过采样率参数。
        importance_sample_ratio (float): 通过重要性采样采样的点所占的比例。
    Returns:
        point_coords (Tensor): 形状为 (N, P, 2) 的张量，包含 P 个采样点的坐标。
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0] # 获取批次大小
    num_sampled = int(num_points * oversample_ratio) # 计算过采样后的点数
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device) # 随机生成 [0, 1] x [0, 1] 范围内的过采样点坐标
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False) # 在粗略预测图上采样这些点的 logit 值
    # 基于采样点的预测值计算不确定性至关重要。
    # 如果先计算粗略预测的不确定性，然后为点采样这些不确定性，会导致不正确的结果。
    # 例如：假设 uncertainty_func(logits)=-abs(logits)，一个位于两个 logits 为 -1 和 1 的粗略预测之间的采样点将具有 0 logits，因此不确定性为 0。
    # 但是，如果我们先计算粗略预测的不确定性，两者都将具有 -1 的不确定性，而采样点将获得 -1 的不确定性。
    point_uncertainties = uncertainty_func(point_logits) # 计算采样点的不确定性
    num_uncertain_points = int(importance_sample_ratio * num_points) # 计算通过重要性采样选择的点数
    num_random_points = num_points - num_uncertain_points # 计算随机采样的点数
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1] # 获取不确定性最高的前 num_uncertain_points 个点的索引
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device) # 为每个批次中的索引添加偏移量
    idx += shift[:, None] # 将偏移量添加到索引，以便在展平的点坐标张量中正确索引
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    ) # 根据不确定性最高的索引选择对应的点坐标
    if num_random_points > 0:
        point_coords = cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device), # 随机生成剩余的点坐标
            ],
            dim=1,
        ) # 将通过重要性采样得到的点坐标和随机生成的点坐标连接起来
    return point_coords

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """
    计算 DICE 损失，类似于 masks 的广义 IOU。
    Args:
        inputs: 任意形状的浮点张量，表示每个示例的预测。
        targets: 与 inputs 形状相同的浮点张量，存储 inputs 中每个元素的二元分类标签
                 （0 表示负类，1 表示正类）。
        num_masks: 批次中的 mask 数量，用于归一化损失。
    """
    inputs = inputs.sigmoid() # 对预测值进行 Sigmoid 激活，将其转换为概率 [0, 1]
    inputs = inputs.flatten(1) # 将预测展平为 (N, P) 的形状，其中 P 是点的数量
    numerator = 2 * (inputs * targets).sum(-1) # 计算 Dice 系数的分子：2 * (预测与目标的交集)
    denominator = inputs.sum(-1) + targets.sum(-1) # 计算 Dice 系数的分母：预测的总和 + 目标的总和
    loss = 1 - (numerator + 1) / (denominator + 1) # 计算 Dice 损失，添加 1 以避免除以零
    return loss.sum() / num_masks # 对批次中的损失求和并除以 mask 的数量进行归一化


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule，使用 torch.jit 编译以提高效率


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
        foreground_weight: float = 10.0,
        background_weight: float = 1.0,
):
    """
    Args:
        inputs: 任意形状的浮点张量，表示每个示例的预测 logits。
        targets: 与 inputs 形状相同的浮点张量，存储 inputs 中每个元素的二元分类标签
                 （0 表示负类，1 表示正类）。
        num_masks: 批次中的 mask 数量，用于归一化损失。
        foreground_weight (float): 前景类别的损失权重。
        background_weight (float): 背景类别的损失权重。
    Returns:
        Loss tensor
    """
    foreground_weight = 10.0
    background_weight =1.0
    weights = targets * foreground_weight + (1 - targets) * background_weight # 为每个点分配权重，前景点具有较高的权重
    loss = F.binary_cross_entropy_with_logits(inputs, targets, weight=weights, reduction="none") # 计算带权重的二元交叉熵损失，不进行聚合

    return loss.mean(1).sum() / num_masks # 对每个点的损失取平均，然后对批次中的损失求和，并除以 mask 的数量进行归一化


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule，使用 torch.jit 编译以提高效率


def calculate_uncertainty(logits):
    """
    我们将不确定性估计为前景类别在 `logits` 中的 logit 预测值与 0.0 之间的 L1 距离。
    Args:
        logits (Tensor): 形状为 (R, 1, ...) 的张量，表示类别特定的或类别无关的 logit，
                         其中 R 是所有图像中预测的 mask 总数，C 是前景类别的数量。
                         这里假设只有一个前景类别，所以第二维是 1。
    Returns:
        scores (Tensor): 形状为 (R, 1, ...) 的张量，包含不确定性得分，
                         不确定性最高的位置具有最高的不确定性得分。
    """
    assert logits.shape[1] == 1 # 确保只有一个通道（对于二元分割或类别无关的情况）
    gt_class_logits = logits.clone() # 克隆 logits 以避免原地修改
    return -(torch.abs(gt_class_logits)) # 计算 logit 绝对值的负数作为不确定性得分。
                                      # 绝对值越大（越远离 0），不确定性越高，取负号后得分越低。
                                      # 因此，我们希望选择绝对值大的 logit 进行更精细的采样。

def loss_masks(src_masks, target_masks, num_masks, oversample_ratio=3.0):
    """计算与 masks 相关的损失：focal loss 和 dice loss。
    targets 字典必须包含键 "masks"，其值为形状为 [nb_target_boxes, h, w] 的张量。
    """

    # 由于我们使用的是归一化坐标，因此无需上采样预测 :)

    with torch.no_grad():
        # 采样 point_coords
        point_coords = get_uncertain_point_coords_with_randomness(
            src_masks, # 预测的 masks，用于指导点采样
            lambda logits: calculate_uncertainty(logits), # 使用 calculate_uncertainty 函数计算不确定性
            112 * 112, # 要采样的总点数
            oversample_ratio, # 过采样率
            0.75, # 重要性采样率
        )
        # 获取 ground truth 标签
        point_labels = point_sample(
            target_masks, # 真实的 masks
            point_coords, # 采样的点坐标
            align_corners=False,
        ).squeeze(1) # 在真实的 masks 上采样对应点的标签，并移除通道维度

    point_logits = point_sample(
        src_masks, # 预测的 masks
        point_coords, # 采样的点坐标
        align_corners=False,
    ).squeeze(1) # 在预测的 masks 上采样对应点的预测 logits，并移除通道维度

    loss_mask = sigmoid_ce_loss_jit(point_logits, point_labels, num_masks) # 计算基于采样点的 Sigmoid 交叉熵损失
    loss_dice = dice_loss_jit(point_logits, point_labels, num_masks) # 计算基于采样点的 Dice 损失

    del src_masks # 删除不再需要的预测 masks 以释放内存
    del target_masks # 删除不再需要的真实 masks 以释放内存
    return loss_mask, loss_dice # 返回计算得到的 mask 损失和 Dice 损失

```

## **ROM-SAM 中 Loss 的计算方式（基于提供的代码推测）：**

ROM-SAM 的 mask loss 可能会包含以下两个主要部分，并且是基于**采样点**进行计算的，而不是直接在整个 mask 上计算：

1. **Sigmoid 交叉熵损失 (Sigmoid Cross-Entropy Loss):**
    
    - 通过 `sigmoid_ce_loss_jit` 函数计算。
    - 这个损失衡量了模型预测的每个采样点属于前景或背景的置信度与真实标签之间的差异。
    - 它会对前景和背景点设置不同的权重 (`foreground_weight`, `background_weight`)，通常前景权重更高，以更关注前景区域的分割准确性。
2. **Dice 损失 (Dice Loss):**
    
    - 通过 `dice_loss_jit` 函数计算。
    - Dice 损失是一种基于区域相似性的损失函数，它衡量了模型预测的采样点概率分布与真实标签之间的重叠程度。
    - 它对于处理类别不平衡问题有一定的鲁棒性。

**计算流程推测:**

1. **点采样 (Point Sampling):**
    
    - `get_uncertain_point_coords_with_randomness` 函数负责根据预测的粗略 mask (`src_masks`) 的不确定性来采样一些关键点。不确定性高的区域会被采样更多点，这是一种关注难分割区域的策略。同时也会随机采样一些点以覆盖整个区域。
    - `point_sample` 函数用于在预测的 mask (`src_masks`) 和真实的 mask (`target_masks`) 上，根据采样得到的坐标 (`point_coords`) 提取对应位置的值（logits 和标签）。
2. **损失计算:**
    
    - `sigmoid_ce_loss_jit` 使用采样得到的预测 logits (`point_logits`) 和真实的标签 (`point_labels`) 计算 Sigmoid 交叉熵损失。
    - `dice_loss_jit` 同样使用 `point_logits` 和 `point_labels` 计算 Dice 损失。
3. **损失聚合:**
    
    - 两种损失会分别计算，并最终可能以加权的方式组合起来，作为 ROM-SAM 的 mask 分割损失。

**输入 (Input):**

- **`src_masks`**: 模型预测的 masks，形状通常为 `(N, C, H, W)` 或 `(N, 1, H, W)`。`N` 是批次大小，`C` 是类别数（对于二元分割通常为 1），`H` 和 `W` 是 mask 的高度和宽度。这个输入是用于指导点采样的不确定性估计以及后续的损失计算。
- **`target_masks`**: 真实的 masks (ground truth)，形状通常为 `(N, num_target_masks, H, W)`。`num_target_masks` 是该图像中目标实例的个数。在点采样时，会根据采样的点坐标从这些真实的 masks 中提取对应的标签。
- **`num_masks`**: 一个浮点数，表示批次中真实 mask 的总数，用于对损失进行归一化。

**输出 (Output):**

- **`loss_mask`**: 基于采样点计算的 Sigmoid 交叉熵损失，一个标量 `torch.Tensor`。
- **`loss_dice`**: 基于采样点计算的 Dice 损失，一个标量 `torch.Tensor`。

**总结:**

ROM-SAM (如果采用了类似的代码逻辑) 的 mask loss 计算不是直接在完整的预测 mask 和真实 mask 之间进行的，而是通过以下步骤：

1. **基于预测的不确定性采样一些代表性的点。**
2. **在预测和真实的 mask 上提取这些采样点的数值和标签。**
3. **使用这些采样点的值和标签计算 Sigmoid 交叉熵损失和 Dice 损失。**

这种基于点采样的损失计算方式的优点在于可以更高效地训练高分辨率的 mask，并且能够更关注难以分割的区域，从而提升分割的精度。最终的 mask loss 可能是 `loss_mask` 和 `loss_dice` 的加权和。

# 个人思考

ROM-SAM是基于SAM在遥感视频的相关任务上进行的扩展，其主要：
1. 用到了LoRA来对图像的编码进行微调训练，这一点感觉没什么特别的，很多论文都有用到这个方法，本质就是想让原来的模型能够在新的具体的领域里拥有适配的学习能力。
2. 引入了Feature Fusion HQ Mask Decoder来对对图像进行高质量的掩码，这个感觉比较新颖一些，其融合了早期的特征和最后的特征，而早期提取的主要是局部边缘特征，后期主要提取的是全局上下文特征，这样的多级特征融合有利于提高细粒度水平。
3. Feature Fusion HQ Mask Decoder的结果与SAM的Mask Decoder的结果再做矩阵乘法操作，也就融合和原本SAM模型的作用效果，得到最终的分割掩码。*==但是我不知道为什么要进行矩阵乘法操作，一般下意识会想到两个图进行融合应该使用相加的方式。==*

