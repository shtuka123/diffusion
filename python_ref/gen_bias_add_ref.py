"""Reference data for bias_add."""
import torch
from tensor_io import save_tensor

torch.manual_seed(1)

B, D = 32, 128
x = torch.randn(B, D)
bias = torch.randn(D)
y = x + bias  # broadcasts

save_tensor('data/ref/bias_add_x.bin', x)
save_tensor('data/ref/bias_add_bias.bin', bias)
save_tensor('data/ref/bias_add_y.bin', y)
print(f"Saved bias_add reference: x ({B},{D}), bias ({D},)")