"""Reference data for cross-entropy loss."""
import torch
import numpy as np
from tensor_io import save_tensor

# Test 1: a tiny hand-verifiable case
# Logits [[1, 2, 3], [3, 2, 1]], labels [2, 0]
# Row 0: lse = log(exp(1) + exp(2) + exp(3)) ≈ 3.407, loss = 3.407 - 3 = 0.407
# Row 1: same lse (symmetric), loss = 3.407 - 3 = 0.407
# Mean = 0.407
x_tiny = torch.tensor([[1., 2., 3.], [3., 2., 1.]])
y_tiny = torch.tensor([2, 0], dtype=torch.long)
loss_tiny = torch.nn.functional.cross_entropy(x_tiny, y_tiny)  # mean by default

save_tensor('data/ref/ce_tiny_logits.bin', x_tiny)
# Save labels as int32 binary (matches our C++ loader)
np.array(y_tiny.numpy(), dtype=np.int32).tofile('data/ref/ce_tiny_labels.bin')
save_tensor('data/ref/ce_tiny_loss.bin', loss_tiny.reshape(1))

# Test 2: realistic batch (MNIST-like)
torch.manual_seed(0)
B, C = 128, 10
x_batch = torch.randn(B, C) * 3.0   # moderate logits
y_batch = torch.randint(0, C, (B,))
loss_batch = torch.nn.functional.cross_entropy(x_batch, y_batch)

save_tensor('data/ref/ce_batch_logits.bin', x_batch)
np.array(y_batch.numpy(), dtype=np.int32).tofile('data/ref/ce_batch_labels.bin')
save_tensor('data/ref/ce_batch_loss.bin', loss_batch.reshape(1))

# Test 3: large logits (would NaN without the logsumexp trick)
x_large = torch.randn(64, 10) * 10 + 100
y_large = torch.randint(0, 10, (64,))
loss_large = torch.nn.functional.cross_entropy(x_large, y_large)

save_tensor('data/ref/ce_large_logits.bin', x_large)
np.array(y_large.numpy(), dtype=np.int32).tofile('data/ref/ce_large_labels.bin')
save_tensor('data/ref/ce_large_loss.bin', loss_large.reshape(1))

print("Cross-entropy references saved:")
print(f"  tiny  (2,3): loss = {loss_tiny.item():.6f}")
print(f"  batch (128,10): loss = {loss_batch.item():.6f}")
print(f"  large (64,10,+100): loss = {loss_large.item():.6f}")