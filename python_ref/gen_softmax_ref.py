"""Reference data for softmax, including extreme-logit stress test."""
import torch
from tensor_io import save_tensor

# Test 1: normal logits (values around 0). Any implementation passes this.
torch.manual_seed(0)
x_normal = torch.randn(32, 10)
y_normal = torch.softmax(x_normal, dim=-1)
save_tensor('data/ref/softmax_normal_x.bin', x_normal)
save_tensor('data/ref/softmax_normal_y.bin', y_normal)

# Test 2: large logits. A naive implementation (no max subtraction)
# overflows expf(x) to inf here and produces NaN.
x_large = torch.randn(32, 10) * 10 + 100  # ~100, range ~80-120
y_large = torch.softmax(x_large, dim=-1)
save_tensor('data/ref/softmax_large_x.bin', x_large)
save_tensor('data/ref/softmax_large_y.bin', y_large)

# Test 3: extremely large logits. Even more aggressive.
x_extreme = torch.randn(32, 10) * 50 + 500  # ~500, range ~350-650
y_extreme = torch.softmax(x_extreme, dim=-1)
save_tensor('data/ref/softmax_extreme_x.bin', x_extreme)
save_tensor('data/ref/softmax_extreme_y.bin', y_extreme)

# Test 4: larger class dim. Representative of GPT-tiny's vocab size.
x_wide = torch.randn(16, 65)
y_wide = torch.softmax(x_wide, dim=-1)
save_tensor('data/ref/softmax_wide_x.bin', x_wide)
save_tensor('data/ref/softmax_wide_y.bin', y_wide)

print("Saved softmax references:")
print(f"  normal  shape={tuple(x_normal.shape)}, logit range [{x_normal.min():.2f}, {x_normal.max():.2f}]")
print(f"  large   shape={tuple(x_large.shape)},  logit range [{x_large.min():.2f}, {x_large.max():.2f}]")
print(f"  extreme shape={tuple(x_extreme.shape)}, logit range [{x_extreme.min():.2f}, {x_extreme.max():.2f}]")
print(f"  wide    shape={tuple(x_wide.shape)},   logit range [{x_wide.min():.2f}, {x_wide.max():.2f}]")