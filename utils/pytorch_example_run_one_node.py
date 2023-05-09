# Refer: https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html#torch.Tensor.scatter_
# Dependencies:
# $ pip install torch

import torch
src = torch.arange(1, 11).reshape((2, 5))
index = torch.tensor([[0, 1, 2, 0]])
dst = torch.zeros(3, 5, dtype=src.dtype)

print("dim = ", 0)
print("src=\n", src)
print("index=\n", index)
print("src dst=\n", dst)
dst.scatter_(0, index, src)
print("final dst=\n", dst)
