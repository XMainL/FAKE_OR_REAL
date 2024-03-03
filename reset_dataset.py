import os
import shutil
import numpy as np

# 定义源文件夹和目标文件夹
source_dir = "./dataset/real_train"
train_dir = "./dataset/train/real"
valid_dir = "./dataset/valid/real"


# 获取源文件夹中的所有文件
files = os.listdir(source_dir)

# 确保目标文件夹存在，如果不存在，则创建它
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

# 打乱文件列表
np.random.shuffle(files)

# 计算训练集的文件数量
num_train = int(len(files) * 0.8)

# 将源文件夹中的文件复制到目标文件夹中
for i, file in enumerate(files):
    if i < num_train:
        shutil.copy(os.path.join(source_dir, file), train_dir)
    else:
        shutil.copy(os.path.join(source_dir, file), valid_dir)