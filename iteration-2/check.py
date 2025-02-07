import torch
from tensor import Tensor

"""My implementation"""
a = Tensor([1, 2, 3], requires_grad=True)
b = Tensor([4, 5, 6], requires_grad=True)

c = a @ b

print(f"""\na: {a}, a.grad: {a.grad}
b: {b}, b.grad: {b.grad}
c: {c}, c.grad: {c.grad}""")

c.backward()

print(f"\na.grad: {a.grad}\nb.grad: {b.grad}\n")

"""PyTorch"""
a_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
b_torch = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

c_torch = a_torch @ b_torch
c_torch.retain_grad()

print(f"""a_torch: {a_torch}, a_torch.grad: {a_torch.grad}
b_torch: {b_torch}, b_torch.grad: {b_torch.grad}
c_torch: {c_torch})""")

c_torch.backward()

print(f"\na_torch.grad: {a_torch.grad}\nb_torch.grad: {b_torch.grad}\n")
