

---
# 博客

综合：（以防万一网页丢失，我将全文复制到了下面）
[一文理顺：pytorch、cuda版本，从此不再为兼容问题头疼！ - 哔哩哔哩](https://www.bilibili.com/opus/926860762897448993)

这篇里面有提到要不要勾选driver（驱动）选项：
[(21 封私信 / 81 条消息) 我的电脑已经有cuda，再安装一个低版本的cuda会有什么影响？ - 知乎](https://www.zhihu.com/question/444878482#:~:text=%E7%AC%AC%E5%9B%9B%E6%AD%A5%EF%BC%8C%E6%8C%89%E7%85%A7%E5%AE%83%E6%8F%90%E7%A4%BA%E4%B8%80%E7%9B%B4%E5%BE%80%E4%B8%8B%EF%BC%8C%20%E4%BD%86%E4%B8%8D%E8%A6%81%E5%8B%BE%E9%80%89%20driver%20%EF%BC%88%E9%9C%80%E8%A6%81%E5%AE%89%E8%A3%85%E9%A9%B1%E5%8A%A8%E7%9A%84%E8%A7%81%E4%B8%8B%E4%B8%80%E7%AB%A0%EF%BC%89%EF%BC%81,%E5%A6%82%E6%9E%9C%E4%BD%A0%E6%B2%A1%E6%9C%89%E8%AE%A9%E5%AE%83%E5%B8%AE%E4%BD%A0%E6%B7%BB%E5%8A%A0%EF%BC%8C%E5%B0%B1%E6%8C%89%E7%85%A7%E7%AC%AC%E4%B8%80%E7%AB%A0%E7%9A%84%E6%96%B9%E5%BC%8F%E8%87%AA%E5%B7%B1%E6%B7%BB%E5%8A%A0%E3%80%82%20%E7%9C%8B%E6%87%82%E7%AC%AC%E4%B8%80%E7%AB%A0%EF%BC%8C%E5%AE%89%E8%A3%85%E5%AE%8C%E4%B9%8B%E5%90%8E%E6%89%BE%E4%B8%8D%E5%88%B0CUDA%E6%88%96%E8%80%85%E5%88%87%E6%8D%A2%E4%B8%8D%E4%BA%86%E5%B0%B1%E5%BE%88%E5%A5%BD%E8%A7%A3%E5%86%B3%E3%80%82%20%E6%88%91%E4%B8%8A%E4%B8%80%E7%AB%A0%E6%8F%90%E5%88%B0%E8%BF%87%EF%BC%8C%E5%8F%AA%E5%AE%89%E8%A3%85CUDA%E8%A6%81%E5%8E%BB%E6%8E%89driver%E7%9A%84%E9%82%A3%E4%B8%AA%E5%8B%BE%EF%BC%8C%E6%98%AF%E5%9B%A0%E4%B8%BACUDA%E7%9A%84%E5%AE%89%E8%A3%85%E5%8C%85%E6%98%AF%E5%B8%A6%E6%98%BE%E5%8D%A1%E9%A9%B1%E5%8A%A8%E7%9A%84%EF%BC%8C%E8%80%8C%E4%B8%94%E6%98%AF%E5%AF%B9%E5%BA%94%E7%89%88%E6%9C%AC%E7%9A%84%E3%80%82%20%E5%A6%82%E6%9E%9C%E4%BD%A0%E5%AE%89%E8%A3%85%E9%A9%B1%E5%8A%A8%E5%87%BA%E9%94%99%EF%BC%8C%E6%8C%89%E7%85%A7%E6%8A%A5%E9%94%99%E4%BF%A1%E6%81%AF%E7%99%BE%E5%BA%A6%EF%BC%8C%E5%A4%9A%E5%8D%8A%E6%98%AF%E4%BD%A0%E6%B2%A1%E5%8D%B8%E8%BD%BD%E5%B9%B2%E5%87%80%E3%80%82)
一般来说，两种情况需要安装显卡驱动：
1. 显卡驱动过低（nvidia-smi的cuda小于需要安装的cuda版本），需要重装
2. 新机没驱动，需要安装
3. 用着用着报错cuda与驱动不匹配，多半是显卡驱动自动升级了

其他相关博客：
[关于ubuntu22.04安装CUDA显示Existing package manager installation of the driver found. 的解决方案-CSDN博客](https://blog.csdn.net/Sillydust/article/details/146046407?ops_request_misc=%257B%2522request%255Fid%2522%253A%25228d7c107bbe8ccb3fea00237353d580c5%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=8d7c107bbe8ccb3fea00237353d580c5&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-146046407-null-null.142^v102^pc_search_result_base2&utm_term=Existing%20package%20manager%20installation%20of%20the%20driver%20found.%20It%20is%20strongly%20%20%20%20%E2%94%82%20%E2%94%82%20recommended%20that%20you%20remove%20this%20before%20continuing.%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%E2%94%82%20%E2%94%82%20Abort%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20&spm=1018.2226.3001.4187)

不同版本cuda的来回切换：

## **1 痛点**

无论是本地环境，还是云GPU环境，基本都事先装好了pytorch、cuda，想运行一个新项目，到底版本是否兼容？

解决思路： 从根本上出发：GPU、项目对pytorch的版本要求

最理想状态：如果能根据项目，直接选择完美匹配的平台，丝滑启动。

### **1.1 如果CUDA版本不对**

在我安装pytorch3d时，cuda版本不对，报错： 

要解决这个问题，需要先了解当前环境的信息，然后根据GPU和项目版本要求推算出合适的版本，再安装。具体如下：

## **2 查看当前环境信息**

### **2.1 使用shell查看**

```javascript
# 显卡驱动信息，主要看CUDA支持的最高版本
nvidia-smi

# 当前使用的CUDA的版本
nvcc -V

# 查看安装了几个CUDA，当前使用哪个版本的CUDA
ll /usr/local/

# 查看已安装的包的版本
conda list | grep cuda
conda list | grep torch
```

2.2 使用py脚本查看

vim version.py

```javascript
import torch
print(torch.__version__) # 查看torch版本
print(torch.cuda.is_available()) # 看安装好的torch和cuda能不能用，也就是看GPU能不能用

print(torch.version.cuda) # 输出一个 cuda 版本，注意：上述输出的 cuda 的版本并不一定是 Pytorch 在实际系统上运行时使用的 cuda 版本，而是编译该 Pytorch release 版本时使用的 cuda 版本，详见：https://blog.csdn.net/xiqi4145/article/details/110254093

import torch.utils
import torch.utils.cpp_extension
print(torch.utils.cpp_extension.CUDA_HOME) #输出 Pytorch 运行时使用的 cuda
```

3 推算合适的pytorch和cuda版本

安装CUDA过程并不难，主要是理解CUDA、cudatoolkit以及3个cuda版本的关系。理解到位之后，安装就是落地而已。在边踩坑边学习的过程中，学到以下文章：

### **3.1 pytorch和cuda的关系，看这篇：**

如何解决PyTorch版本和CUDA版本不匹配的关系 - 知乎 (zhihu.com) 

https://zhuanlan.zhihu.com/p/633473214

核心步骤：

1. 根据GPU型号，去官网CUDA GPUs上去查询版本号，下图1中显示，RTX 3090的计算能力架构版本号是8.6，对应sm_86。其中8是主版本号，6是次版本号。
2. 仍然是上面的网页中，点链接进去，可查看到该GPU的架构。比如RTX 3090架构为Ampere
3. 根据架构，从下图2中查到CUDA版本范围，比如Ampere为CUDA 11.0-12.2
4. 项目一般会指定PyTorch版本，然后去PyTorch官网Start Locally | PyTorch找到PyTorch和CUDA的交集，选择CUDA最高的（运算更快）
5. 官方提供的一般是pip方式安装，如果慢，可尝试换源、代理等方式。
6. 除了pip安装方式，也可以whl方式下载离线安装包：

```javascript
以Windows下为例。

假设在pytorch获得的pip安装命令为：
pip install torch===1.7.0+cu110 torchvision===0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

如何获取whl离线安装包并安装？

下载地址：https://download.pytorch.org/whl/torch_stable.html，下载以下安装包：

torch-1.7.0+cu110-cp37-cp37m-win_amd64.whl
torchvision-0.8.1+cu110-cp37-cp37m-win_amd64.whl
torchaudio-0.7.0-cp37-none-win_amd64.whl

注意：cu110表示CUDA是11.0版本的，cp37表示python3.7，win表示windows版本，具体选择什么版本，可以参考上图中的“Run this Command”。

安装方法：进入离线安装包所在位置，然后“shift+鼠标右键”，然后选择“在此处打开powershell窗口”，最后输入“pip install torch-1.7.0+cu110-cp37-cp37m-win_amd64.whl”，即输入“pip install xxxx.whl”。

有可能会出现[winError]拒绝访问的错误提示，并且要求你添加“--user”，你可以这样输入："pip install xxxx.whl --user"
```

![](https://i0.hdslb.com/bfs/new_dyn/6d68b4002ebddd4813cb9f80aaf01a0d423296573.png@1192w.webp)

![](https://i0.hdslb.com/bfs/new_dyn/5ff004cf8e3983e2bd893d9ae8086570423296573.png@1192w.webp)

### **3.2 深入了解cuda、cudatoolkit以及多版本cuda共存时pytorch调用哪个**

进一步，你有必要深入了解一下cuda、cudatoolkit以及多版本cuda共存时pytorch调用哪个 cuda和cudatoolkit-CSDN博客

https://blog.csdn.net/xiqi4145/article/details/110254093

### **3.3 安装需要的CUDA，多版本共存，并自由切换！**

了解完了以上知识，那么你需要的是：安装需要的CUDA，多版本共存，并自由切换！ 【多版本cuda自由切换】在ubuntu上安装多个版本的CUDA，并且可以随时切换cuda-11.3//cuda-11.8//cuda-11.6//cuda-11.2_ubuntu切换cuda-CSDN博客 

https://blog.csdn.net/BetrayFree/article/details/134870198

注意：

1. 安装包类型要选择runfile，其它二者据说会有一些自动升级的行为，比较麻烦。
2. 实际安装过程中，我选择了驱动，但是没选择kernel objects、nvidia-fs
3. 可能会报nvidia的错误，看下面的处理

### **3.4 nvidia报错的处理**

在安装过程中，会遇到报错,nvidia驱动需要卸载，参考： CUDA、驱动安装与踩坑记录 - 知乎 (zhihu.com) ubuntu升级NVIDIA驱动，遇到ERROR: An NVIDIA kernel module ‘nvidia-uvm‘ appears to already be loaded in your_error: an nvidia kernel module 'nvidia' appears to-CSDN博客

https://zhuanlan.zhihu.com/p/642632372

https://blog.csdn.net/hjxu2016/article/details/135128492

---

# 遇到的问题与解决

## **问题**：
```

(base) root@lab509:/mnt/newdisk/hjj/hjj/anaconda3/envs/ogdino/lib/python3.7/site-packages# sudo sh cuda_11.3.0_465.19.01_linux.run
 Failed to verify gcc version. See log at /var/log/cuda-installer.log for details.

/var/log/cuda-installer.log里的内容如下：

[INFO]: Driver installation detected by command: apt list --installed | grep -e nvidia-driver-[0-9][0-9][0-9] -e nvidia-[0-9][0-9][0-9]
[INFO]: Cleaning up window
[INFO]: Complete
[INFO]: Checking compiler version...
[INFO]: gcc location: /usr/bin/gcc

[INFO]: gcc version: gcc version 12.3.0 (Ubuntu 12.3.0-1ubuntu1~22.04) 

[ERROR]: unsupported compiler version: 12.3.0. Use --override to override this check.

```

## **解决**：
根据错误信息，CUDA 11.3 安装程序不支持您当前的 GCC 12.3.0 版本。以下是两种解决方案：

### 解决方案 1：安装兼容的 GCC 版本（推荐）
```bash
# 安装 GCC 9（CUDA 11.3 官方支持版本）
sudo apt install gcc-9 g++-9

# 创建版本切换配置
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9

# 切换到 GCC 9
sudo update-alternatives --config gcc  # 选择 gcc-9 的编号
sudo update-alternatives --config g++  # 选择 g++-9 的编号

# 验证版本
gcc --version  # 应显示 9.x.x

# 重新运行安装（保留原有参数）
sudo sh cuda_11.3.0_465.19.01_linux.run
```

### 解决方案 2：强制覆盖版本检查
```bash
# 添加 --override 参数跳过编译器检查
sudo sh cuda_11.3.0_465.19.01_linux.run --override
```

---

### 注意事项：
1. **推荐使用方案 1**，因为 CUDA 对编译器版本敏感，强制覆盖可能导致运行时错误
2. 安装完成后可恢复默认 GCC：
   ```bash
   sudo update-alternatives --config gcc  # 选择原始 GCC 12
   sudo update-alternatives --config g++  # 选择原始 G++ 12
   ```
3. 如果遇到依赖问题，可尝试：
   ```bash
   sudo apt install build-essential
   ```
4. CUDA 11.3 官方支持矩阵：
   - GCC ≤ 10（但推荐 ≤ 9）
   - Ubuntu 20.04 LTS

> 安装完成后验证：`nvcc --version` 应显示 11.3 版本


## 问题和解决二
点击continue之后，选完选项之后，可能会跳出来一个提示说“已经存在一些版本，是否要update to this installation”，可以选择“no”，这样就不会覆盖之前的版本。