import torch

a = torch.tensor([1., 2., 3.], requires_grad=True)
print(a.grad)

out = a.sigmoid()
print(out)

# 添加detach()，c的requires_grad为False
c = a.detach()
print(c)

# 这个时候没有对c进行更改，所以并不会影响backward()
out.sum().backward()
print(a.grad)
