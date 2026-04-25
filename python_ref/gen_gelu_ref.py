"""Reference for GELU forward and backward (exact / 'none' approximation)."""
import torch
import torch.nn.functional as F
from tensor_io import save_tensor

torch.manual_seed(0)

# A reasonable spread including negatives, near-zero, and positives.
x = torch.randn(32, 64, requires_grad=True)
y = F.gelu(x, approximate='none')   # exact via erf

loss = (y ** 2).sum()
loss.backward()

save_tensor('data/ref/gelu_x.bin',  x.detach())
save_tensor('data/ref/gelu_y.bin',  y.detach())
save_tensor('data/ref/gelu_dy.bin', (2.0 * y).detach())
save_tensor('data/ref/gelu_dx.bin', x.grad.detach())

print(f"Saved GELU ref: shape {tuple(x.shape)}")
print(f"  y range: [{y.min().item():.3f}, {y.max().item():.3f}]")