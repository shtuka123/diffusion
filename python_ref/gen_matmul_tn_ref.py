"""Reference for A^T @ B."""
import torch
from tensor_io import save_tensor

torch.manual_seed(0)

K, M, N = 8, 12, 16
A = torch.randn(K, M)
B = torch.randn(K, N)
C = A.t() @ B   # (M, N)

save_tensor('data/ref/matmul_tn_A.bin', A)
save_tensor('data/ref/matmul_tn_B.bin', B)
save_tensor('data/ref/matmul_tn_C.bin', C)
print(f"matmul_tn reference: K={K}, M={M}, N={N}")