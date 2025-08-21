### 设置代理
```shell
# 查看代理设置
env | grep -i proxy

# 临时设置代理
export http_proxy="代理地址"
export https_proxy="代理地址"

# 关闭临时代理
unset http_proxy
unset https_proxy
```

### 查看Ubuntu版本的方法
```shell
# 方法一：使用/proc/version文件
# 在Linux系统中，_/proc_目录包含了当前系统运行的各种数据，其中_version_文件记录了系统的版本信息。可以通过_cat_命令来查看这个文件的内容，这种方法简单直接。例如：

cat /proc/version

# 执行这个命令后，你会看到类似以下的输出，其中包含了Linux内核的版本号和GCC的版本号，以及Ubuntu的版本号。
# Linux version 5.4.0-99-generic (buildd@lgw01-amd64-007) (gcc version 9.3.0 (Ubuntu 9.3.0-17ubuntu1~20.04)) #112-Ubuntu SMP Thu Feb 3 15:52:13 UTC 2022

# 方法二：使用uname命令
#_uname_命令可以显示系统的内核版本和系统架构（如32位或64位）。_-a_参数会显示所有可用的系统信息。例如：

uname -a

# 执行这个命令后，你会得到类似以下的输出，其中包含了内核版本和系统架构信息。
# Linux ubuntu 5.4.0-99-generic #112-Ubuntu SMP Thu Feb 3 15:52:13 UTC 2022 x86_64 x86_64 x86_64 GNU/Linux

# 方法三：使用lsb_release命令
# _lsb_release_命令提供了一个关于Linux标准基础的系统版本信息。_-a_参数会显示所有的LSB（Linux Standard Base）信息。例如：

lsb_release -a

# 执行这个命令后，你会看到类似以下的输出，这是最清晰的版本显示方式，它提供了发行版ID、描述、发行号和代号名称。
# No LSB modules are available
# Distributor ID: Ubuntu
# Description: Ubuntu 20.04.3 LTS
# Release: 20.04
# Codename: focal
```

### Linux中删除文件的方法
```shell

# 删除单个文件
rm 文件名
rm test.txt

# 批量删除文件
# 可以在_rm_命令后面列出所有文件名，并用空格分隔：
rm 文件1 文件2 文件3
rm file1.txt file2.txt file3.txt

# 删除文件夹及其内容
# 可以使用_-r_（递归删除）选项。这将确保连同子文件夹一起删除：
rm -r 文件夹名
rm -r my_folder

# 确认删除
# 默认情况下，_rm_命令会直接删除文件，不会询问确认。如果你希望在删除文件时得到确认提示，可以使用_-i_选项：
rm -i 文件名
rm -i important.txt

# 强制删除
# 如果你希望在删除文件时不接收任何确认提示，并强制删除文件，可以使用_-f_选项：
rm -f 文件名
rm -f temp.txt

# 请注意，使用_-f_选项时要非常小心，因为它会立即删除文件，且无法恢复。

```

# 重命名文件
```shell

# 前缀
rename -d 's/^/redcap12m-/' 目标路径

```

# 目录相关
```shell

# 显示当前目录位置
pwd

# 列出当前目录下的目录和文件
ls
# 得到所有关于ls的命令
ls --help

# 切换或进入目录命令

# 进入根目录：
cd /
# 目前所在目录：
cd .
# 返回上一层:
cd ..
# 返回上两层:
cd ../..
# 进入当前目录父目录的***目录：
cd ../***
# 进入root的根目录：
cd ~=cd /root

```

# 如何查看pip的版本
```shell

pip --version

```

[(21 封私信 / 81 条消息) linux下文件夹的创建、复制、剪切、重命名、清空和删除命令 - 知乎](https://zhuanlan.zhihu.com/p/26491512)

# 查看目录空间大小

```shell

df -h

```

