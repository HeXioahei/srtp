经过广泛搜索，未能找到关于"OpenEarthMap_xBD"这个特定数据集上语义分割任务的最新SOTA（State-of-the-Art）性能的公开记录。这可能是因为该数据集较新，或者其详细的基准测试结果尚未被广泛报道。

不过，我可以为你提供一些相关信息和寻找SOTA的方法：

### 🔍 如何查找和确认SOTA

1.  **查阅官方来源**：优先访问 **OpenEarthMap** [官网](https://open-earth-map.org) 和 **xBD** [官网](https://xview2.org/dataset) 。这些网站通常会提供官方基准测试结果或链接到相关研究。
2.  **关注学术平台**：
    *   **Papers with Code**：这个网站通常会汇总各种数据集上不同任务的SOTA结果。你可以搜索 "OpenEarthMap" 或 "xBD semantic segmentation"。
    *   **arXiv** 和 **顶级会议论文**：关注CVPR、ICCV、ECCV等计算机视觉顶会中关于遥感图像分割、灾害评估、多模态学习等主题的最新论文。西安交大曹相湧副教授团队在CVPR 2025上发表的《SegEarth-OV》工作就是一个例子。
3.  **理解相关技术**：许多SOTA模型源于一些前沿的技术方向，例如：
    *   **开放词汇分割 (Open-Vocabulary Segmentation)**：旨在让模型能够识别训练时未见过的类别。
    *   **无监督域自适应 (Unsupervised Domain Adaptation, UDA)**：用于解决训练数据和测试数据分布不一致的问题，这在融合不同来源的数据集（如OpenEarthMap和xBD）时非常重要。
    *   **多模态融合 (Multimodal Fusion)**：结合不同传感器或来源的数据（如光学图像、雷达、深度信息等）来提升分割精度。

### 💎 一点建议

如果你正在进行相关研究，除了追逐SOTA指标外，更重要的是：
*   **清晰定义任务**：明确你的语义分割任务具体是什么，比如是土地覆盖分类还是灾害损坏评估。
*   **选择合适的模型**：根据你的任务特点（如是否需要识别新类别、数据来源是否单一等）选择技术路线。
*   **严谨的实验设计**：确保实验 comparisons 是公平和可复现的。

希望这些信息能为你提供一些方向。如果你有关于这些特定技术方向或者其他数据集（如标准的OpenEarthMap或xBD）的问题，我很乐意提供更多信息。

# 2
根据你提供的论文《SegEarth-OV: Towards Training-Free Open-Vocabulary Segmentation for Remote Sensing Images》，SegEarth-OV 在 **17 个遥感图像数据集**上进行了全面评估，涵盖了以下四类任务：

1. **语义分割（Semantic Segmentation）**
2. **建筑物提取（Building Extraction）**
3. **道路提取（Road Extraction）**
4. **洪水检测（Flood Detection）**

---

### 📊 SegEarth-OV 在各项任务上的性能（SOTA 水平）：

| 任务类型 | 数据集数量 | SegEarth-OV 平均提升 vs. 之前 SOTA | 备注 |
|----------|------------|-------------------------------------|------|
| 语义分割 | 8          | +5.8% mIoU                         | 在 UDD5 上达到 50.6% mIoU |
| 建筑物提取 | 4          | +8.2% IoU                          | 使用 896×896 图像时进一步提升 |
| 道路提取 | 4          | +4.0% IoU                          | 整体性能仍较低（最高 35.4% IoU）|
| 洪水检测 | 1          | +15.3% IoU                         | 达到 60.2% IoU |

---

### ✅ 是否已有超越 SegEarth-OV 的方法？

根据该论文（发表于 2024 年，可能是 CVPR 或 ICCV）：

- **SegEarth-OV 是第一个专门为遥感图像设计的 training-free OVSS 方法**。
- 对比的方法都是为自然图像设计的（如 MaskCLIP、SCLIP、ClearCLIP 等），在遥感图像上表现均不如 SegEarth-OV。
- 论文中提到的唯一同期遥感 OVSS 工作是 [6]（Cao et al., 2024），但它是 **training-required** 的，而 SegEarth-OV 是 **training-free** 的，因此不具备直接可比性。

因此，**在该论文发表时，SegEarth-OV 在所述 17 个测试基准上均达到了 SOTA（State-of-the-Art）水平**，并且没有其他方法在这些任务上全面超越它。

---

### 🔍 建议后续验证：

如果你想确认是否有新工作超越 SegEarth-OV，可以：

1. 检查 **CVPR 2025、ICCV 2025、ECCV 2024** 等顶会的最新论文；
2. 在 **arXiv** 上搜索关键词：`open-vocabulary segmentation remote sensing`、`OVSS remote sensing`；
3. 关注相关领域的最新综述（如 [68] 和 [81] 中提到的开放词汇分割进展）。

目前来看，**SegEarth-OV 是该领域的一个强基线，且尚未被全面超越**。

如果需要我帮你检索最新进展，也可以告诉我。