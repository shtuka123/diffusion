"""PyTorch reference implementation of GPT-tiny.

Same architecture as our CUDA version. Used for end-to-end validation:
- Load identical weights (saved from C++)
- Run forward/backward
- Compare outputs to C++ outputs
"""

import math
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """Pre-LN transformer block. Same topology as our C++ implementation."""

    def __init__(self, d_model, n_heads, d_mlp, max_seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_model = d_model

        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)

        # MHA: keep as separate Linears (not packed) to match our C++
        self.W_Q = nn.Linear(d_model, d_model, bias=True)
        self.W_K = nn.Linear(d_model, d_model, bias=True)
        self.W_V = nn.Linear(d_model, d_model, bias=True)
        self.W_O = nn.Linear(d_model, d_model, bias=True)

        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)

        self.W_mlp1 = nn.Linear(d_model, d_mlp, bias=True)
        self.W_mlp2 = nn.Linear(d_mlp, d_model, bias=True)

    def forward(self, x):
        # MHA sub-block
        t1 = self.ln1(x)
        B, T, D = t1.shape
        H, dk = self.n_heads, self.d_k

        Q = self.W_Q(t1).view(B, T, H, dk).transpose(1, 2).reshape(B * H, T, dk)
        K = self.W_K(t1).view(B, T, H, dk).transpose(1, 2).reshape(B * H, T, dk)
        V = self.W_V(t1).view(B, T, H, dk).transpose(1, 2).reshape(B * H, T, dk)

        a = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        a = a.reshape(B, H, T, dk).transpose(1, 2).reshape(B, T, D)
        a = self.W_O(a)

        r1 = x + a

        # MLP sub-block
        t2 = self.ln2(r1)
        h = F.gelu(self.W_mlp1(t2), approximate='none')
        m = self.W_mlp2(h)

        return r1 + m


class GPTTiny(nn.Module):
    def __init__(self, vocab_size=65, max_seq_len=64,
                 d_model=128, n_layers=4, n_heads=4, d_mlp=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_mlp = d_mlp

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_mlp, max_seq_len)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model, eps=1e-5)
        self.head = nn.Linear(d_model, vocab_size, bias=True)

    def forward(self, tokens):
        # tokens: (B, T) long
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Weight loading from our C++ checkpoint format

GPT_SAVE_MAGIC = 0x47505430  # 'GPT0'

def _read_param(f):
    """Read one parameter from the C++ checkpoint format."""
    ndim = struct.unpack('<I', f.read(4))[0]
    shape = struct.unpack(f'<{ndim}I', f.read(4 * ndim))
    numel = 1
    for d in shape:
        numel *= d
    raw = f.read(4 * numel)
    return torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape).clone()


def load_cpp_weights(model: GPTTiny, path: str):
    """Load a checkpoint saved by our C++ GPTTiny::save() into a PyTorch model.

    The C++ ordering of parameters (from params() method) is:
        E, P
        for each block:
            gamma1, beta1
            W_Q, b_Q, W_K, b_K, W_V, b_V, W_O, b_O
            gamma2, beta2
            W1_mlp, b1_mlp, W2_mlp, b2_mlp
        gamma_f, beta_f
        W_head, b_head

    Note: the C++ stores W shapes as (D_in, D_out) but PyTorch's nn.Linear
    stores weight as (D_out, D_in). So all linear weights need transposing.
    """
    with open(path, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        assert magic == GPT_SAVE_MAGIC, f"bad magic: 0x{magic:x}"

        # Embeddings: (V, D), no transpose needed.
        E = _read_param(f)
        model.token_emb.weight.data.copy_(E)
        P = _read_param(f)
        model.pos_emb.weight.data.copy_(P)

        for block in model.blocks:
            # LN1
            block.ln1.weight.data.copy_(_read_param(f))
            block.ln1.bias.data.copy_(_read_param(f))
            # MHA: 4 linears (Q, K, V, O), each W is (D, D) → transpose
            for linear in [block.W_Q, block.W_K, block.W_V, block.W_O]:
                W = _read_param(f)
                linear.weight.data.copy_(W.t())
                linear.bias.data.copy_(_read_param(f))
            # LN2
            block.ln2.weight.data.copy_(_read_param(f))
            block.ln2.bias.data.copy_(_read_param(f))
            # MLP: 2 linears, transpose weights
            W1 = _read_param(f)
            block.W_mlp1.weight.data.copy_(W1.t())
            block.W_mlp1.bias.data.copy_(_read_param(f))
            W2 = _read_param(f)
            block.W_mlp2.weight.data.copy_(W2.t())
            block.W_mlp2.bias.data.copy_(_read_param(f))

        # Final LN
        model.ln_f.weight.data.copy_(_read_param(f))
        model.ln_f.bias.data.copy_(_read_param(f))
        # Head
        W_head = _read_param(f)
        model.head.weight.data.copy_(W_head.t())
        model.head.bias.data.copy_(_read_param(f))

    print(f"Loaded weights from {path} into PyTorch model")