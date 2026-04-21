"""Generate reference inputs and outputs for the matmul test."""
import torch
from tensor_io import save_tensor

torch.manual_seed(0)

M, K, N = 128, 256, 64
A = torch.randn(M, K)
B = torch.randn(K, N)
C = A @ B

print(f"Generated matmul reference: A({M}x{K}) @ B({K}x{N}) = C({M}x{N})")
print(f"  C[0, 0] = {C[0, 0].item():.6f}")
print(f"  C max   = {C.max().item():.6f}")

save_tensor('data/ref/matmul_A.bin', A)
save_tensor('data/ref/matmul_B.bin', B)
save_tensor('data/ref/matmul_C.bin', C)

print("Saved to data/ref/")