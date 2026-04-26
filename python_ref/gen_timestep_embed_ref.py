"""Reference for sinusoidal timestep embedding + SiLU."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from tensor_io import save_tensor

torch.manual_seed(0)

# Test sinusoidal embedding for a few timesteps
B = 8
d = 32  # embedding dim

# Sample some timesteps spanning [0, 999]
timesteps = torch.tensor([0, 1, 50, 100, 250, 500, 750, 999], dtype=torch.int32)

# Compute interleaved sinusoidal embedding (matches our C++)
def sinusoidal_emb(t, d, max_period=10000.0):
    half = d // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half) / half)  # (half,)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half)
    sin = torch.sin(args)  # (B, half)
    cos = torch.cos(args)  # (B, half)
    # Interleave: [sin_0, cos_0, sin_1, cos_1, ...]
    emb = torch.stack([sin, cos], dim=-1).reshape(t.shape[0], d)
    return emb

emb = sinusoidal_emb(timesteps, d)
np.array(timesteps.numpy(), dtype=np.int32).tofile('data/ref/temb_timesteps.bin')
save_tensor('data/ref/temb_emb.bin', emb)

print(f"Sinusoidal embedding test:")
print(f"  B={B}, d={d}, timesteps={timesteps.tolist()}")
print(f"  emb[0,:6] (t=0)   = {emb[0,:6].tolist()}")
print(f"  emb[7,:6] (t=999) = {emb[7,:6].tolist()}")

# SiLU forward + backward reference
torch.manual_seed(1)
x_silu = torch.randn(64, requires_grad=True)
y_silu = F.silu(x_silu)
loss = (y_silu ** 2).sum()
loss.backward()

save_tensor('data/ref/silu_x.bin', x_silu.detach())
save_tensor('data/ref/silu_y.bin', y_silu.detach())
save_tensor('data/ref/silu_dy.bin', (2.0 * y_silu).detach())
save_tensor('data/ref/silu_dx.bin', x_silu.grad.detach())
print(f"\nSiLU test: shape {x_silu.shape}")