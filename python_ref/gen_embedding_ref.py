"""Reference for token + positional embedding."""
import torch
import torch.nn as nn
import numpy as np
from tensor_io import save_tensor

torch.manual_seed(0)

V = 64       # vocab size
D = 32       # embedding dim
T_max = 16   # max sequence length

B, T = 2, 8

# Random tokens in [0, V)
tokens = torch.randint(0, V, (B, T))

# Embedding tables (require grad for backward)
E = (torch.randn(V, D) * 0.1).requires_grad_()
P = (torch.randn(T_max, D) * 0.1).requires_grad_()

# Forward
y_token = E[tokens]                      # (B, T, D)
y = y_token + P[:T]                      # broadcast P over batch

# Scalar loss
loss = (y ** 2).sum()
loss.backward()

save_tensor('data/ref/emb_E.bin',  E.detach())
save_tensor('data/ref/emb_P.bin',  P.detach())
np.array(tokens.numpy(), dtype=np.int32).tofile('data/ref/emb_tokens.bin')
save_tensor('data/ref/emb_y.bin',  y.detach())
save_tensor('data/ref/emb_dy.bin', (2.0 * y).detach())
save_tensor('data/ref/emb_dE.bin', E.grad.detach())
save_tensor('data/ref/emb_dP.bin', P.grad.detach())

print(f"Saved embedding reference: V={V}, D={D}, T_max={T_max}, B={B}, T={T}")
print(f"  tokens[0] = {tokens[0].tolist()}")
print(f"  y[0,0,:4] = {y[0,0,:4].detach().tolist()}")