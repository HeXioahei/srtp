```shell
# 创建新的conda环境
conda create --name myenv python=3.9

# 删除conda环境
conda env remove -n <your_env>

# 查看现有的所有环境
conda env list

# 复制conda环境
conda create --name newenv --clone oldenv

# 列出当前环境下的库
conda list

# 只看torch相关的库
conda list | grep torch
```
