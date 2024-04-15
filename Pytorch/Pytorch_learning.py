
import torch

print(torch.__version__)

x = torch.empty(5,3,dtype=torch.long)

print(x)

y = torch.rand(5,3)
print(y)

