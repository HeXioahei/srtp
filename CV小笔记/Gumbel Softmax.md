# Gumbel Softmax 详解

## 什么是Gumbel Softmax？

Gumbel Softmax是一种在深度学习中对离散随机变量进行可微采样的技术。它结合了Gumbel分布和softmax函数，使得从分类分布中采样变得可微分，从而可以通过反向传播进行优化。

### 核心问题

在深度学习中，当我们想要从离散分布中采样时（如在VAE、强化学习或网络结构搜索中），标准的采样操作是不可微的，这阻碍了梯度反向传播。Gumbel Softmax通过引入一个可微的近似解决了这个问题。

### Gumbel Max Trick

Gumbel Softmax基于Gumbel Max Trick，该技巧表明：

如果我们要从分类分布 $p_1, p_2, ..., p_k$ 中采样，可以：
1. 为每个类别 $i$ 采样一个Gumbel噪声 $g_i \sim \text{Gumbel}(0, 1)$
2. 计算 $z_i = \log(p_i) + g_i$
3. 选择 $y = \text{softmax}_i z_i$

这样得到的y服从原始的分类分布。

### Gumbel Softmax

Gumbel Softmax将不可微的argmax操作替换为可微的softmax：

$$y_i = \frac{\exp((\log(p_i) + g_i)/\tau)}{\sum_{j=1}^k \exp((\log(p_j) + g_j)/\tau)}$$

其中$\tau$是温度参数：
- 当$\tau \to 0$时，Gumbel Softmax接近one-hot向量
- 当$\tau \to \infty$时，Gumbel Softmax接近均匀分布

## 示例代码

```python
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def sample_gumbel(shape, eps=1e-20):
    """从Gumbel(0,1)分布中采样"""
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax(logits, temperature=1.0, hard=False):
    """
    Gumbel Softmax函数
    
    参数:
    - logits: 未归一化的log概率，形状为(*, num_classes)
    - temperature: 温度参数
    - hard: 如果为True，返回的样本将被离散化为one-hot向量
    
    返回:
    - 采样结果，形状与logits相同
    """
    # 采样Gumbel噪声
    gumbel_noise = sample_gumbel(logits.shape)
    
    # 添加Gumbel噪声到logits
    y = logits + gumbel_noise
    
    # 应用温度参数和softmax
    y_soft = F.softmax(y / temperature, dim=-1)
    
    if hard:
        # 在正向传播中使用离散样本，但在反向传播中使用连续近似
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        # 使用straight-through estimator
        y = y_hard - y_soft.detach() + y_soft
    
    return y

# 示例1: 基本使用
def basic_example():
    # 创建logits (未归一化的概率)
    logits = torch.tensor([[1.0, 2.0, 3.0]])  # 三个类别的logits
    
    print("原始logits:", logits)
    print("Softmax概率:", F.softmax(logits, dim=-1))
    
    # 使用不同的温度参数
    for temp in [1.0, 0.5, 0.1]:
        samples = []
        for _ in range(1000):
            sample = gumbel_softmax(logits, temperature=temp, hard=True)
            samples.append(sample.numpy())
        
        samples = np.array(samples)
        avg_sample = samples.mean(axis=0)
        print(f"温度={temp}, 平均采样结果: {avg_sample}")

# 示例2: 在简单分类任务中的应用
class SimpleClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, temperature=1.0, hard=False):
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)
        
        # 使用Gumbel Softmax采样
        if self.training:
            # 训练时使用Gumbel Softmax
            return gumbel_softmax(logits, temperature, hard)
        else:
            # 测试时直接使用argmax
            return F.softmax(logits, dim=-1)

# 示例3: 温度参数的影响可视化
def temperature_effect():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    temperatures = [10.0, 5.0, 2.0, 1.0, 0.5, 0.1, 0.05, 0.01]
    
    plt.figure(figsize=(12, 8))
    
    for i, temp in enumerate(temperatures):
        samples = []
        for _ in range(1000):
            sample = gumbel_softmax(logits, temperature=temp, hard=False)
            samples.append(sample.detach().numpy())
        
        samples = np.array(samples).squeeze()
        
        plt.subplot(2, 4, i+1)
        plt.hist(samples[:, 0], alpha=0.5, label='Class 0', bins=20)
        plt.hist(samples[:, 1], alpha=0.5, label='Class 1', bins=20)
        plt.hist(samples[:, 2], alpha=0.5, label='Class 2', bins=20)
        plt.title(f'Temperature = {temp}')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

# 运行示例
if __name__ == "__main__":
    print("=== 基本示例 ===")
    basic_example()
    
    print("\n=== 温度参数影响可视化 ===")
    temperature_effect()
    
    print("\n=== 简单分类器示例 ===")
    # 创建模拟数据
    input_size = 10
    hidden_size = 20
    num_classes = 3
    batch_size = 5
    
    model = SimpleClassifier(input_size, hidden_size, num_classes)
    x = torch.randn(batch_size, input_size)
    
    # 训练模式
    model.train()
    output_train = model(x, temperature=1.0, hard=True)
    print("训练模式输出:", output_train)
    
    # 测试模式
    model.eval()
    output_test = model(x)
    print("测试模式输出:", output_test)
```

## 关键点总结

1. **可微性**: Gumbel Softmax使得离散采样变得可微，允许梯度反向传播
2. **温度参数**: 控制采样结果的"尖锐"程度，需要在训练中适当调整
3. **Straight-through Estimator**: 当使用`hard=True`时，在前向传播中使用离散样本，在反向传播中使用连续近似
4. **应用场景**: VAE中的离散潜变量、强化学习中的离散动作选择、网络结构搜索等

Gumbel Softmax是处理深度学习模型中离散决策问题的强大工具，它通过巧妙的数学技巧解决了离散采样的可微性问题。