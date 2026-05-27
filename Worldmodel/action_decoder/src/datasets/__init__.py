"""
说明：这里把本地 `datasets` 目录声明为 TSformer-VO 的 Python 包。

易混点：仓库顶层目录也叫 `datasets`，在部分 Python 环境里可能和
HuggingFace 的 `datasets` 包重名。保留这个文件可以让解释器明确把本地目录
当作包导入；训练脚本还会把仓库根目录放到 `sys.path` 前面，进一步保证优先
使用本地实现。
"""
