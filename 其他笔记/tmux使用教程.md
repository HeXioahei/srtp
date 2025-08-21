[一次完整的 tmux 使用流程（带示例）以后再也不怕跑代码的时候关机断网了_tmux send-keys-CSDN博客](https://blog.csdn.net/m0_67165162/article/details/149124187)

[Tmux教程 （一文就够）-CSDN博客](https://blog.csdn.net/CSSDCC/article/details/121231906)

# 简易教程
### tmux使用教程（使用tmux建立的会话运行代码，即使电脑断网，服务器上依然运行代码，好用！！！）

- sudo apt-get install tmux 安装tmux包
    
- tmux new -s test 创建一个新的会话，名为test
    
- 按ctrl+b松手再按d可以退出该会话
    
- tmux attach -t test 返回/连接test会话窗口
    
- tmux kill-session -t test 杀掉test会话
    
- tmux ls可以查看当前所有的会话

## 问题：
```

hjj@lab509:/$ conda activate gdino 
conda: command not found 
hjj@lab509:/$ 


[gdino] 0:bash* "lab509" 14:16 29-7月-25 


我在尝试使用tmux来在远程服务器上运行代码，当我创建了一个新的tmux会话之后，却无法在这个会话中运行conda命令，这是为什么？

```

## 解答：

### 原因分析
1. **Shell 初始化文件未加载**  
   tmux 默认启动的是 **非登录 shell**（non-login shell），仅加载 `~/.bashrc`（bash 用户）或 `~/.zshrc`（zsh 用户）。如果 Conda 的初始化脚本被配置在 `~/.bash_profile` 或 `~/.profile`（登录 shell 的配置文件）中，tmux 会话将无法加载 Conda。

2. **PATH 环境变量缺失**  
   Conda 的可执行文件路径（通常是 `~/miniconda3/bin` 或 `~/anaconda3/bin`）未包含在 tmux 会话的 `PATH` 中。

---

### 解决方案
#### 方法 1：手动加载 Conda 初始化脚本（推荐）
在 tmux 会话中直接运行以下命令：
```bash
source ~/.bashrc  # 如果使用 bash
# 或
source ~/.zshrc   # 如果使用 zsh
```
然后再次尝试激活环境：
```bash
conda activate gdino
```

#### 方法 2：确保 Conda 初始化代码在 `~/.bashrc` 中
检查 Conda 的初始化代码是否在 `~/.bashrc`（bash 用户）或 `~/.zshrc`（zsh 用户）中：
1. 打开配置文件：
   ```bash
   nano ~/.bashrc  # 或 nano ~/.zshrc
   ```
2. 确认存在类似以下代码（路径可能不同）：
   ```bash
   # >>> conda initialize >>>
   __conda_setup="$('/path/to/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
   if [ $? -eq 0 ]; then
       eval "$__conda_setup"
   else
       if [ -f "/path/to/miniconda3/etc/profile.d/conda.sh" ]; then
           . "/path/to/miniconda3/etc/profile.d/conda.sh"
       else
           export PATH="/path/to/miniconda3/bin:$PATH"
       fi
   fi
   unset __conda_setup
   # <<< conda initialize <<<
   ```
3. 如果不存在，**重新初始化 Conda**：
   ```bash
   # 退出 tmux，在普通终端中执行：
   conda init bash  # bash 用户
   conda init zsh   # zsh 用户
   ```

#### 方法 3：在 tmux 中启动登录 Shell
启动 tmux 时显式指定登录 Shell：
```bash
tmux new-session -s mysession bash --login
# 或进入已有会话后执行：
bash --login
```

#### 方法 4：检查 Conda 安装路径是否在 `PATH` 中
在 tmux 会话中临时添加 Conda 路径：
```bash
export PATH="/path/to/conda/bin:$PATH"  # 替换为你的 Conda 路径
conda activate gdino
```

---

### 验证步骤
1. 在 tmux 会话中执行：
   ```bash
   source ~/.bashrc   # 加载配置
   which conda         # 应显示 Conda 路径
   conda env list      # 应列出所有环境
   conda activate gdino
   ```
2. 如果仍失败，检查 Conda 安装路径是否正确：
   ```bash
   ls -l ~/miniconda3/bin/conda  # 或 ~/anaconda3/bin/conda
   ```

---

### 永久修复
确保 Conda 初始化代码位于 `~/.bashrc`（而非 `~/.bash_profile`）中，因为 tmux 默认加载 `~/.bashrc`。通过 `conda init bash` 可自动完成配置。

> **提示**：如果服务器多人共用，请联系管理员确认 Conda 的全局配置路径。