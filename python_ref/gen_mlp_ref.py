"""Reference for the Transformer MLP sub-block: Linear -> GELU -> Linear."""
import torch
import torch.nn.functional as F
from tensor_io import save_tensor

torch.manual_seed(0)

B, T, D = 2, 4, 16
D_hidden = 4 * D

x = torch.randn(B, T, D, requires_grad=True)
W1 = torch.randn(D, D_hidden, requires_grad=True) * 0.1
b1 = torch.randn(D_hidden, requires_grad=True) * 0.1
W2 = torch.randn(D_hidden, D, requires_grad=True) * 0.1
b2 = torch.randn(D, requires_grad=True) * 0.1

# We want to reuse the same weight tensors as leaves with .grad set, so
# convert to leaves with requires_grad=True after scaling.
W1 = (torch.randn(D, D_hidden) * 0.1).requires_grad_()
b1 = (torch.randn(D_hidden) * 0.1).requires_grad_()
W2 = (torch.randn(D_hidden, D) * 0.1).requires_grad_()
b2 = (torch.randn(D) * 0.1).requires_grad_()

# Forward: y = (GELU(x @ W1 + b1)) @ W2 + b2
h_pre  = x @ W1 + b1
h_post = F.gelu(h_pre, approximate='none')
y      = h_post @ W2 + b2

loss = (y ** 2).sum()
loss.backward()

save_tensor('data/ref/mlp_x.bin',   x.detach())
save_tensor('data/ref/mlp_W1.bin',  W1.detach())
save_tensor('data/ref/mlp_b1.bin',  b1.detach())
save_tensor('data/ref/mlp_W2.bin',  W2.detach())
save_tensor('data/ref/mlp_b2.bin',  b2.detach())
save_tensor('data/ref/mlp_y.bin',   y.detach())
save_tensor('data/ref/mlp_dy.bin',  (2.0 * y).detach())
save_tensor('data/ref/mlp_dx.bin',  x.grad.detach())
save_tensor('data/ref/mlp_dW1.bin', W1.grad.detach())
save_tensor('data/ref/mlp_db1.bin', b1.grad.detach())
save_tensor('data/ref/mlp_dW2.bin', W2.grad.detach())
save_tensor('data/ref/mlp_db2.bin', b2.grad.detach())

print(f"Saved MLP ref: B={B}, T={T}, D={D}, D_hidden={D_hidden}")
print(f"  y[0,0,:4] = {y[0,0,:4].detach().tolist()}")