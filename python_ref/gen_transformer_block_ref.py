"""Reference for a Pre-LN Transformer block with causal MHA + MLP."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensor_io import save_tensor

torch.manual_seed(0)

B, T, D = 2, 4, 16
n_heads = 4
d_k = D // n_heads
D_mlp = 4 * D
eps = 1e-5

x = torch.randn(B, T, D, requires_grad=True)

# LN1 params
gamma1 = torch.randn(D, requires_grad=True)
beta1  = torch.randn(D, requires_grad=True)

# MHA params (our convention: W shape (D, D), so x @ W is the matmul)
W_Q = (torch.randn(D, D) * 0.1).requires_grad_()
b_Q = torch.randn(D, requires_grad=True) * 0.1
W_K = (torch.randn(D, D) * 0.1).requires_grad_()
b_K = torch.randn(D, requires_grad=True) * 0.1
W_V = (torch.randn(D, D) * 0.1).requires_grad_()
b_V = torch.randn(D, requires_grad=True) * 0.1
W_O = (torch.randn(D, D) * 0.1).requires_grad_()
b_O = torch.randn(D, requires_grad=True) * 0.1

# Make biases leaves with grad
b_Q = b_Q.detach().clone().requires_grad_()
b_K = b_K.detach().clone().requires_grad_()
b_V = b_V.detach().clone().requires_grad_()
b_O = b_O.detach().clone().requires_grad_()

# LN2 params
gamma2 = torch.randn(D, requires_grad=True)
beta2  = torch.randn(D, requires_grad=True)

# MLP params
W1_mlp = (torch.randn(D, D_mlp) * 0.1).requires_grad_()
b1_mlp = torch.randn(D_mlp, requires_grad=True) * 0.1
W2_mlp = (torch.randn(D_mlp, D) * 0.1).requires_grad_()
b2_mlp = torch.randn(D, requires_grad=True) * 0.1
b1_mlp = b1_mlp.detach().clone().requires_grad_()
b2_mlp = b2_mlp.detach().clone().requires_grad_()

# Forward
def block(x):
    t1 = F.layer_norm(x, (D,), gamma1, beta1, eps=eps)
    # MHA: x @ W_Q etc.
    Q = t1 @ W_Q + b_Q
    K = t1 @ W_K + b_K
    V = t1 @ W_V + b_V
    Q = Q.view(B, T, n_heads, d_k).transpose(1, 2).reshape(B * n_heads, T, d_k)
    K = K.view(B, T, n_heads, d_k).transpose(1, 2).reshape(B * n_heads, T, d_k)
    V = V.view(B, T, n_heads, d_k).transpose(1, 2).reshape(B * n_heads, T, d_k)
    a_flat = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    a_heads = a_flat.reshape(B, n_heads, T, d_k).transpose(1, 2).reshape(B, T, D)
    a = a_heads @ W_O + b_O
    r1 = x + a

    t2 = F.layer_norm(r1, (D,), gamma2, beta2, eps=eps)
    h = t2 @ W1_mlp + b1_mlp
    h = F.gelu(h, approximate='none')
    m = h @ W2_mlp + b2_mlp
    return r1 + m

y = block(x)
loss = (y ** 2).sum()
loss.backward()

# Save inputs and outputs
save_tensor('data/ref/tb_x.bin',       x.detach())
save_tensor('data/ref/tb_gamma1.bin',  gamma1.detach())
save_tensor('data/ref/tb_beta1.bin',   beta1.detach())
save_tensor('data/ref/tb_W_Q.bin',     W_Q.detach())
save_tensor('data/ref/tb_b_Q.bin',     b_Q.detach())
save_tensor('data/ref/tb_W_K.bin',     W_K.detach())
save_tensor('data/ref/tb_b_K.bin',     b_K.detach())
save_tensor('data/ref/tb_W_V.bin',     W_V.detach())
save_tensor('data/ref/tb_b_V.bin',     b_V.detach())
save_tensor('data/ref/tb_W_O.bin',     W_O.detach())
save_tensor('data/ref/tb_b_O.bin',     b_O.detach())
save_tensor('data/ref/tb_gamma2.bin',  gamma2.detach())
save_tensor('data/ref/tb_beta2.bin',   beta2.detach())
save_tensor('data/ref/tb_W1_mlp.bin',  W1_mlp.detach())
save_tensor('data/ref/tb_b1_mlp.bin',  b1_mlp.detach())
save_tensor('data/ref/tb_W2_mlp.bin',  W2_mlp.detach())
save_tensor('data/ref/tb_b2_mlp.bin',  b2_mlp.detach())
save_tensor('data/ref/tb_y.bin',       y.detach())
save_tensor('data/ref/tb_dy.bin',      (2.0 * y).detach())

# Save all gradients
save_tensor('data/ref/tb_dx.bin',       x.grad.detach())
save_tensor('data/ref/tb_dgamma1.bin',  gamma1.grad.detach())
save_tensor('data/ref/tb_dbeta1.bin',   beta1.grad.detach())
save_tensor('data/ref/tb_dW_Q.bin',     W_Q.grad.detach())
save_tensor('data/ref/tb_db_Q.bin',     b_Q.grad.detach())
save_tensor('data/ref/tb_dW_K.bin',     W_K.grad.detach())
save_tensor('data/ref/tb_db_K.bin',     b_K.grad.detach())
save_tensor('data/ref/tb_dW_V.bin',     W_V.grad.detach())
save_tensor('data/ref/tb_db_V.bin',     b_V.grad.detach())
save_tensor('data/ref/tb_dW_O.bin',     W_O.grad.detach())
save_tensor('data/ref/tb_db_O.bin',     b_O.grad.detach())
save_tensor('data/ref/tb_dgamma2.bin',  gamma2.grad.detach())
save_tensor('data/ref/tb_dbeta2.bin',   beta2.grad.detach())
save_tensor('data/ref/tb_dW1_mlp.bin',  W1_mlp.grad.detach())
save_tensor('data/ref/tb_db1_mlp.bin',  b1_mlp.grad.detach())
save_tensor('data/ref/tb_dW2_mlp.bin',  W2_mlp.grad.detach())
save_tensor('data/ref/tb_db2_mlp.bin',  b2_mlp.grad.detach())

print(f"Saved Transformer block ref: B={B}, T={T}, D={D}, n_heads={n_heads}")
print(f"  y[0,0,:4] = {y[0,0,:4].detach().tolist()}")