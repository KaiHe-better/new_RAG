import torch

a = torch.FloatTensor([0,1,2])
print(a.size())
b = torch.mean(a).reshape(1)
print(b.size())