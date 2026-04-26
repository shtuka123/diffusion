"""Reference: 5 steps of Adam on a small parameter, with fixed gradients."""
import torch
import numpy as np
from tensor_io import save_tensor

torch.manual_seed(0)

# A small parameter and a sequence of fixed gradients to apply.
# We save the trajectory of (w_after_step) to compare.
n = 64
w = torch.randn(n, requires_grad=False).clone()
w_init = w.clone()

# Pre-generate 5 gradients (we won't compute them; we'll inject them)
grads = [torch.randn(n) for _ in range(5)]

opt = torch.optim.Adam([w], lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

# Note: PyTorch's Adam expects w to require grad; we'll set .grad manually.
w.requires_grad_(True)
w_history = [w.detach().clone()]
for g in grads:
    opt.zero_grad()
    w.grad = g.clone()
    opt.step()
    w_history.append(w.detach().clone())

save_tensor('data/ref/adam_w_init.bin', w_init)
for i, g in enumerate(grads):
    save_tensor(f'data/ref/adam_grad_{i}.bin', g)
for i, ws in enumerate(w_history):
    save_tensor(f'data/ref/adam_w_{i}.bin', ws)

print(f"Saved Adam reference: n={n}, 5 steps")
print(f"  w_init[:4] = {w_init[:4].tolist()}")
print(f"  w_after_5[:4] = {w_history[-1][:4].tolist()}")