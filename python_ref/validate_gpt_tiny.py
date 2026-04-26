"""End-to-end validation of our CUDA GPT-tiny against PyTorch.

Both load the same weights (from the C++ checkpoint), run on the same tokens,
and we save the PyTorch outputs to disk so the C++ test can compare.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from gpt_tiny_ref import GPTTiny, load_cpp_weights
from tensor_io import save_tensor

torch.manual_seed(0)

# Match the C++ config exactly
cfg = dict(
    vocab_size=65,
    max_seq_len=64,
    d_model=128,
    n_layers=4,
    n_heads=4,
    d_mlp=512,
)

B = 2
T = 16  # short for fast validation

model = GPTTiny(**cfg)
load_cpp_weights(model, "checkpoints/gpt_tiny.bin")
model.eval()  # disable dropout etc. (not present anyway, defensive)

# Generate input tokens (use a fixed seed so C++ can reproduce them)
torch.manual_seed(42)
tokens = torch.randint(0, cfg['vocab_size'], (B, T))

# Forward
logits = model(tokens)  # (B, T, V)
print(f"logits shape: {logits.shape}, sample: {logits[0, 0, :5].tolist()}")

# Loss: predict tokens[:, 1:] from tokens[:, :-1]
# Standard next-token cross-entropy
labels = tokens.clone()  # for simplicity, use same as input — what matters is reproducibility
loss = F.cross_entropy(logits.reshape(B * T, cfg['vocab_size']), labels.reshape(-1))
print(f"loss: {loss.item():.6f}")

# Backward
loss.backward()

# Save outputs and inputs for the C++ side to load
out_dir = Path("data/ref/gpt_validate")
out_dir.mkdir(parents=True, exist_ok=True)

# Tokens as int32 binary
np.array(tokens.numpy(), dtype=np.int32).tofile(out_dir / "tokens.bin")
# Labels (same as tokens here)
np.array(labels.numpy(), dtype=np.int32).tofile(out_dir / "labels.bin")
# Logits and loss
save_tensor(out_dir / "logits.bin", logits.detach())
save_tensor(out_dir / "loss.bin", loss.detach().reshape(1))

# Save all parameter gradients in the same order as our C++ params() method
def save_grad(name, t):
    save_tensor(out_dir / f"grad_{name}.bin", t)

# Embeddings
save_grad("E", model.token_emb.weight.grad)
save_grad("P", model.pos_emb.weight.grad)
for l, block in enumerate(model.blocks):
    save_grad(f"b{l}_gamma1", block.ln1.weight.grad)
    save_grad(f"b{l}_beta1",  block.ln1.bias.grad)
    # Transpose weight gradients back to C++ convention
    save_grad(f"b{l}_W_Q", block.W_Q.weight.grad.t().contiguous())
    save_grad(f"b{l}_b_Q", block.W_Q.bias.grad)
    save_grad(f"b{l}_W_K", block.W_K.weight.grad.t().contiguous())
    save_grad(f"b{l}_b_K", block.W_K.bias.grad)
    save_grad(f"b{l}_W_V", block.W_V.weight.grad.t().contiguous())
    save_grad(f"b{l}_b_V", block.W_V.bias.grad)
    save_grad(f"b{l}_W_O", block.W_O.weight.grad.t().contiguous())
    save_grad(f"b{l}_b_O", block.W_O.bias.grad)
    save_grad(f"b{l}_gamma2", block.ln2.weight.grad)
    save_grad(f"b{l}_beta2",  block.ln2.bias.grad)
    save_grad(f"b{l}_W1_mlp", block.W_mlp1.weight.grad.t().contiguous())
    save_grad(f"b{l}_b1_mlp", block.W_mlp1.bias.grad)
    save_grad(f"b{l}_W2_mlp", block.W_mlp2.weight.grad.t().contiguous())
    save_grad(f"b{l}_b2_mlp", block.W_mlp2.bias.grad)
save_grad("gamma_f", model.ln_f.weight.grad)
save_grad("beta_f",  model.ln_f.bias.grad)
save_grad("W_head", model.head.weight.grad.t().contiguous())
save_grad("b_head", model.head.bias.grad)

print(f"\nSaved {B=}, {T=} validation data to {out_dir}/")
print("Now run: .\\build\\test_gpt_tiny_vs_ref.exe")