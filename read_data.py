import pandas as pd

# 读取数据
data = pd.read_csv('risk_control/2024年四川省大学生数据科学与统计建模竞赛/训练集/train_data.csv')  # 替换为你的数据文件路径

# 显示数据的前几行
print("数据的前五行：")
print(data.head())

# 描述性统计
descriptive_stats = data.describe()

# 显示描述性统计结果
print("\n描述性统计结果：")
print(descriptive_stats)

# 检查缺失值
missing_values = data.isnull().sum()
print("\n缺失值统计：")
print(missing_values)

# 显示数据类型
data_types = data.dtypes
print("\n数据类型：")
print(data_types)

# 计算相关性矩阵
correlation_matrix = data.corr()
print("\n相关性矩阵：")
print(correlation_matrix)


