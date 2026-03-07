import torch

t = torch.rsqrt(torch.tensor(4.0))
# 开方求倒数
# tensor(0.5000)
print(t)

t2 = torch.ones(3,4)
#创建一个三行4列 全1的张量
# tensor([[1., 1., 1., 1.],
#         [1., 1., 1., 1.],
#         [1., 1., 1., 1.]])
print(t2)

