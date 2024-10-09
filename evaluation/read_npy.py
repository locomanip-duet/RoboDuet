import numpy as np

# 设置要读取的 .npy 文件路径
file_path = "samples/samples_part_1.npy"

# 读取 .npy 文件
samples = np.load(file_path)

# 打印样本的基本信息和统计数据
print(f"样本形状: {samples.shape}")
print(f"样本数量: {samples.shape[0]}")
print(f"每个样本的特征数: {samples.shape[1]}")
print(f"样本的最小值: {samples.min(axis=0)}")
print(f"样本的最大值: {samples.max(axis=0)}")
print(f"样本的平均值: {samples.mean(axis=0)}")
print(f"样本的标准差: {samples.std(axis=0)}")

# 可选：打印前几个样本
print("\n前5个样本:")
print(samples[:5])
