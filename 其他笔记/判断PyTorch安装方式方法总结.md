要判断你的 PyTorch 是通过 `pip` 还是 `conda` 安装的，可以通过以下方法进行验证：

---

### **方法 1：使用 `conda list` 命令（推荐）**
1. 打开终端（Windows：CMD/PowerShell；macOS/Linux：Terminal）。
2. 激活你的 Conda 环境（如果使用了 Conda 环境）：
   ```bash
   conda activate your_env_name  # 替换 your_env_name 为你的环境名
   ```
3. 运行以下命令：
   ```bash
   conda list | grep torch
   ```
   - **结果分析**：
     - 如果输出中包含 **`pypi`** 或 **`<pip>`** 标记，说明是通过 `pip` 安装的：
       ```
       torch                     2.0.1                   pypi_0    pypi
       ```
     - 如果输出中显示 **Conda 渠道**（如 `pytorch`、`conda-forge`），说明是通过 `conda` 安装的：
       ```
       torch                     2.0.1               py3.9_cuda11.8_0    pytorch
       ```

---

### **方法 2：使用 `pip show` 命令**
1. 在终端运行：
   ```bash
   pip show torch
   ```
2. 检查输出中的 **`Location`** 字段：
   - 如果路径包含 **`conda`**（例如 `.../anaconda3/envs/your_env/...`），可能是 Conda 环境中的 `pip` 安装。
   - 如果路径是系统 Python 的 `site-packages`（例如 `/usr/local/lib/python3.9/site-packages`），则是全局 `pip` 安装。

---

### **方法 3：检查 PyTorch 的安装元数据**
1. 在 Python 中运行以下代码：
   ```python
   import torch
   print(torch.__file__)
   ```
2. 观察输出的路径：
   - 若路径中包含 **`site-packages`** 且**不包含 `conda`**，通常是 `pip` 安装。
   - 若路径中包含 **`conda`** 且**不包含 `site-packages`**，通常是 `conda` 安装（例如 `.../anaconda3/envs/your_env/lib/python3.9/site-packages/...` 是 Conda 环境中的安装）。

---

### **总结判断逻辑**
| 观察项                     | `pip` 安装                          | `conda` 安装                        |
|---------------------------|-----------------------------------|-----------------------------------|
| **`conda list` 的 Channel** | 标记为 `pypi` 或 `<pip>`          | 标记为 `pytorch`/`conda-forge`   |
| **安装路径**              | 可能包含 `site-packages`          | 明确包含 `conda` 路径             |
| **`pip show` 的输出**     | 显示完整包信息                    | 在 Conda 环境中可能显示为 `pip` 安装 |

---

### 常见场景示例
- **场景 1**：在 Conda 环境中用 `pip install torch` 安装 → `conda list` 会显示 `pypi`。
- **场景 2**：用 `conda install pytorch torchvision -c pytorch` 安装 → `conda list` 显示渠道为 `pytorch`。
- **场景 3**：直接在系统 Python 中用 `pip3 install torch` 安装 → 路径不包含 `conda`。

根据上述方法，你可以快速确定 PyTorch 的安装方式！