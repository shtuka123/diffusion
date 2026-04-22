"""Reference for multi-head causal self-attention."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensor_io import save_tensor

torch.manual_seed(0)

B, T, D = 2, 8, 16
n_heads = 4
d_k = D // n_heads

x = torch.randn(B, T, D)

# PyTorch's nn.MultiheadAttention packs Q/K/V projections.
# For clarity, we build the projections manually to match our implementation.

# Our weight shape is (D_in, D_out) = (D, D). Biases shape (D,).
# Initialize with nn.init.xavier_uniform_ for consistency.
torch.manual_seed(1)
def init_linear(in_f, out_f):
    w = torch.empty(in_f, out_f)
    nn.init.xavier_uniform_(w)
    b = torch.zeros(out_f)
    return w, b

W_Q, b_Q = init_linear(D, D)
W_K, b_K = init_linear(D, D)
W_V, b_V = init_linear(D, D)
W_O, b_O = init_linear(D, D)

# Forward: Q/K/V projections
Q_raw = x @ W_Q + b_Q   # (B, T, D)
K_raw = x @ W_K + b_K
V_raw = x @ W_V + b_V

# Split into heads: (B, T, D) -> (B, T, H, dk) -> (B, H, T, dk)
Q_heads = Q_raw.view(B, T, n_heads, d_k).transpose(1, 2)
K_heads = K_raw.view(B, T, n_heads, d_k).transpose(1, 2)
V_heads = V_raw.view(B, T, n_heads, d_k).transpose(1, 2)

# Attention per head (treat (B, H) as batch)
# Flatten: (B*H, T, dk)
Q_flat = Q_heads.reshape(B * n_heads, T, d_k)
K_flat = K_heads.reshape(B * n_heads, T, d_k)
V_flat = V_heads.reshape(B * n_heads, T, d_k)

attn_flat = F.scaled_dot_product_attention(Q_flat, K_flat, V_flat, is_causal=True)
# (B*H, T, dk) -> (B, H, T, dk)
attn_heads = attn_flat.reshape(B, n_heads, T, d_k)
# Transpose back: (B, H, T, dk) -> (B, T, H, dk) -> (B, T, D)
attn_concat = attn_heads.transpose(1, 2).reshape(B, T, D)

# Output projection
y = attn_concat @ W_O + b_O

save_tensor('data/ref/mha_x.bin', x)
save_tensor('data/ref/mha_W_Q.bin', W_Q)
save_tensor('data/ref/mha_b_Q.bin', b_Q)
save_tensor('data/ref/mha_W_K.bin', W_K)
save_tensor('data/ref/mha_b_K.bin', b_K)
save_tensor('data/ref/mha_W_V.bin', W_V)
save_tensor('data/ref/mha_b_V.bin', b_V)
save_tensor('data/ref/mha_W_O.bin', W_O)
save_tensor('data/ref/mha_b_O.bin', b_O)
save_tensor('data/ref/mha_y.bin', y)

print(f"Saved MHA reference: B={B}, T={T}, D={D}, n_heads={n_heads}, d_k={d_k}")
print(f"  y[0, 0, :4] = {y[0, 0, :4].tolist()}")