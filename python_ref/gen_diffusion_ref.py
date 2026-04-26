"""Reference for noise schedule + q_sample."""
import numpy as np
import torch
from tensor_io import save_tensor

torch.manual_seed(0)

T = 1000
beta_start = 1e-4
beta_end = 2e-2

# Schedule
betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)
sqrt_ab = torch.sqrt(alpha_bar)
sqrt_omab = torch.sqrt(1.0 - alpha_bar)

# Sanity values (we'll check the C++ side computes the same)
print(f"T={T}")
print(f"  alpha_bar[0]   = {alpha_bar[0].item():.8f}  (≈ 0.9999)")
print(f"  alpha_bar[T-1] = {alpha_bar[-1].item():.8f}  (≈ 4e-5)")
print(f"  sqrt_ab[T-1]   = {sqrt_ab[-1].item():.8f}  (≈ 0.006)")

# q_sample test: B=4 samples, per-batch timesteps, noise
B = 4
D = 16
x0 = torch.randn(B, D)
eps = torch.randn(B, D)
timesteps = torch.tensor([0, 100, 500, 999], dtype=torch.int32)

# Compute x_t per batch element using the per-element timestep
xt = torch.zeros_like(x0)
for b in range(B):
    t = timesteps[b].item()
    xt[b] = sqrt_ab[t] * x0[b] + sqrt_omab[t] * eps[b]

# Save references
save_tensor('data/ref/diff_betas.bin', betas)
save_tensor('data/ref/diff_alpha_bar.bin', alpha_bar)
save_tensor('data/ref/diff_x0.bin', x0)
save_tensor('data/ref/diff_eps.bin', eps)
save_tensor('data/ref/diff_xt.bin', xt)
np.array(timesteps.numpy(), dtype=np.int32).tofile('data/ref/diff_timesteps.bin')

print(f"\nq_sample test:")
print(f"  B={B}, D={D}, timesteps={timesteps.tolist()}")
print(f"  xt[0,:4] = {xt[0, :4].tolist()}")