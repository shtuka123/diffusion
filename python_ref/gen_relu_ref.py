"""Reference data for ReLU forward."""
import torch
from tensor_io import save_tensor

torch.manual_seed(0)

# Test at a size that exercises full-block parallelism but stays small
x = torch.randn(32, 128)
y = torch.relu(x)

save_tensor('data/ref/relu_x.bin', x)
save_tensor('data/ref/relu_y.bin', y)
print(f"Saved ReLU reference: x shape {tuple(x.shape)}, y shape {tuple(y.shape)}")