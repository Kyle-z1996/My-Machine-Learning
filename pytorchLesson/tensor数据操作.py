import torch

a = torch.rand(1, 2, 3, 4)
print(a.shape)
print(a.size(0))  # 维度大小
print(a.dim())  # 总维度


