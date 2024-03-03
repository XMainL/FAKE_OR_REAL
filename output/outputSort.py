import pandas as pd

# 读取 Excel 文件
df = pd.read_csv('./output/predictions (2).csv')

# 提取文件名中的数字部分，并转化为整数
df['number'] = df['image_path'].str.extract('(\d+)').astype(int)

# 按 'number' 列进行排序
df.sort_values('number', inplace=True)

# 删除 'number' 列
df.drop('number', axis=1, inplace=True)

# 将排序后的 DataFrame 写入新的 Excel 文件
df.to_csv('./output/sorted_file.csv', index=False)