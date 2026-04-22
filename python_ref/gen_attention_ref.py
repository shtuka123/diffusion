"""Reference for single-head causal attention."""
import torch
import torch.nn.functional as F
from tensor_io import save_tensor

torch.manual_seed(0)

B, T, d_k, d_v = 2, 8, 16, 16
Q = torch.randn(B, T, d_k)
K = torch.randn(B, T, d_k)
V = torch.randn(B, T, d_v)

Y = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

save_tensor('data/ref/attn_Q.bin', Q)
save_tensor('data/ref/attn_K.bin', K)
save_tensor('data/ref/attn_V.bin', V)
save_tensor('data/ref/attn_Y.bin', Y)

print(f"Saved attention reference: B={B}, T={T}, d_k={d_k}, d_v={d_v}")
print(f"  Y[0, 0, :4] = {Y[0, 0, :4].tolist()}")