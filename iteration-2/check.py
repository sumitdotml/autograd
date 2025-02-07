import torch
from tensor import Tensor

"""My implementation"""
a = Tensor([1, 2, 3], requires_grad=True)
b = Tensor([4, 5, 6], requires_grad=True)

c = a @ b
d = c + b
e = d * b
f = e / b
g = f ** b

print(f"""\n=================== My implementation ===================
a: {a}, a.grad: {a.grad}
b: {b}, b.grad: {b.grad}
c: {c}, c.grad: {c.grad}
d: {d}, d.grad: {d.grad}
e: {e}, e.grad: {e.grad}
f: {f}, f.grad: {f.grad}
g: {g}, g.grad: {g.grad}""")

g.backward()

print(f"""\nMy implementation's grad after backward pass:
a.grad: {a.grad}
b.grad: {b.grad}
c.grad: {c.grad}
d.grad: {d.grad}
e.grad: {e.grad}
f.grad: {f.grad}
g.grad: {g.grad}""")

"""PyTorch"""
a_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
b_torch = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

c_torch = a_torch @ b_torch; c_torch.retain_grad()
d_torch = c_torch + b_torch; d_torch.retain_grad()
e_torch = d_torch * b_torch; e_torch.retain_grad()
f_torch = e_torch / b_torch; f_torch.retain_grad()
g_torch = f_torch ** b_torch; g_torch.retain_grad()

print(f"""\n=================== PyTorch's implementation ===================
a_torch: {a_torch}, a_torch.grad: {a_torch.grad}
b_torch: {b_torch}, b_torch.grad: {b_torch.grad}
c_torch: {c_torch}, c_torch.grad: {c_torch.grad}
d_torch: {d_torch}, d_torch.grad: {d_torch.grad}
e_torch: {e_torch}, e_torch.grad: {e_torch.grad}
f_torch: {f_torch}, f_torch.grad: {f_torch.grad}
g_torch: {g_torch}, g_torch.grad: {g_torch.grad}""")

gradient_weights_g = torch.ones_like(g_torch)

g_torch.backward(gradient_weights_g)

print(f"""\nPyTorch's grad after backward pass:
a_torch.grad: {a_torch.grad}
b_torch.grad: {b_torch.grad}
c_torch.grad: {c_torch.grad}
d_torch.grad: {d_torch.grad}
e_torch.grad: {e_torch.grad}
f_torch.grad: {f_torch.grad}
g_torch.grad: {g_torch.grad}""")
