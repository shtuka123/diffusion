"""Reference data for linear layer forward and backward."""
import torch
import torch.nn as nn
from tensor_io import save_tensor

torch.manual_seed(0)

B, D_in, D_out = 16, 32, 8
x = torch.randn(B, D_in, requires_grad=True)

# PyTorch Linear uses y = x @ W^T + b internally with W of shape (D_out, D_in).
# We use y = x @ W + b with W of shape (D_in, D_out).
# So: our W is PyTorch's W transposed.
layer = nn.Linear(D_in, D_out, bias=True)
W_torch = layer.weight.detach().clone()   # (D_out, D_in)
b_torch = layer.bias.detach().clone()     # (D_out,)

y = layer(x)

# Scalar loss: sum of squares, so dL/dy = 2*y
loss = (y ** 2).sum()
loss.backward()

# Convert PyTorch's W (D_out, D_in) to our W (D_in, D_out): transpose
W_ours = W_torch.t().contiguous()
b_ours = b_torch.clone()

# Gradients: PyTorch gives us x.grad, layer.weight.grad, layer.bias.grad
# layer.weight.grad is shape (D_out, D_in) -- transpose for our convention
dx = x.grad.detach().clone()
dW_ours = layer.weight.grad.t().contiguous()
db_ours = layer.bias.grad.clone()

# Upstream gradient dy = 2 * y (from our sum-of-squares loss)
dy = (2.0 * y).detach().clone()

save_tensor('data/ref/linear_x.bin',  x.detach())
save_tensor('data/ref/linear_W.bin',  W_ours)
save_tensor('data/ref/linear_b.bin',  b_ours)
save_tensor('data/ref/linear_y.bin',  y.detach())
save_tensor('data/ref/linear_dy.bin', dy)
save_tensor('data/ref/linear_dx.bin', dx)
save_tensor('data/ref/linear_dW.bin', dW_ours)
save_tensor('data/ref/linear_db.bin', db_ours)

print(f"Linear reference saved (B={B}, D_in={D_in}, D_out={D_out})")
print(f"  y[0,:3] = {y[0,:3].detach().tolist()}")
print(f"  loss = {loss.item():.4f}")