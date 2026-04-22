"""Reference for LayerNorm forward and backward."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensor_io import save_tensor

torch.manual_seed(0)

B, T, D = 2, 4, 16
x = torch.randn(B, T, D, requires_grad=True)
gamma = torch.randn(D, requires_grad=True)
beta = torch.randn(D, requires_grad=True)

y = F.layer_norm(x, (D,), weight=gamma, bias=beta, eps=1e-5)

# Scalar loss for backward: sum(y^2), so dL/dy = 2*y
loss = (y ** 2).sum()
loss.backward()

save_tensor('data/ref/ln_x.bin',      x.detach())
save_tensor('data/ref/ln_gamma.bin',  gamma.detach())
save_tensor('data/ref/ln_beta.bin',   beta.detach())
save_tensor('data/ref/ln_y.bin',      y.detach())
save_tensor('data/ref/ln_dy.bin',     (2.0 * y).detach())
save_tensor('data/ref/ln_dx.bin',     x.grad.detach())
save_tensor('data/ref/ln_dgamma.bin', gamma.grad.detach())
save_tensor('data/ref/ln_dbeta.bin',  beta.grad.detach())

print(f"Saved LayerNorm reference: B={B}, T={T}, D={D}")
print(f"  y[0,0,:4] = {y[0,0,:4].detach().tolist()}")
print(f"  dx[0,0,:4] = {x.grad[0,0,:4].tolist()}")