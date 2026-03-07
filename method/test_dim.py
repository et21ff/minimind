import torch
x = torch.ones(2, 3)  # 2行3列
# 计算每一行的平均值
mean_no_keep = x.mean(-1, keepdim=False) # 形状 [2]
mean_with_keep = x.mean(-1, keepdim=True) # 形状 [2, 1]

print(f"输入形状: {x.shape}")
print(f"不保留维度: {mean_no_keep.shape}") # 变成了一阶向量
print(f"保留维度: {mean_with_keep.shape}")  # 保持为二阶矩阵