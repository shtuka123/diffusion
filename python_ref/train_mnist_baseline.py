"""PyTorch baseline: (784 -> 256 -> 10) MLP on MNIST, same setup as our C++.

Should match our C++ test accuracy to within ~1% after 3 epochs.
"""
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- Load MNIST (same binary files we use in C++) -----
def _read_be32(f):
    return int.from_bytes(f.read(4), "big")

def load_mnist(img_path, lbl_path):
    with open(img_path, "rb") as f:
        assert _read_be32(f) == 0x00000803
        n = _read_be32(f); r = _read_be32(f); c = _read_be32(f)
        raw = f.read(n * r * c)
    images = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 255.0
    images = images.reshape(n, r * c)
    with open(lbl_path, "rb") as f:
        assert _read_be32(f) == 0x00000801
        n_lab = _read_be32(f)
        labels = np.frombuffer(f.read(n_lab), dtype=np.uint8).astype(np.int64)
    return images, labels

print("Loading MNIST...")
train_x, train_y = load_mnist("data/mnist/train-images-idx3-ubyte",
                              "data/mnist/train-labels-idx1-ubyte")
test_x, test_y = load_mnist("data/mnist/t10k-images-idx3-ubyte",
                            "data/mnist/t10k-labels-idx1-ubyte")
print(f"  train: {len(train_x)}  test: {len(test_x)}")

# ----- Model -----
class MLP(nn.Module):
    def __init__(self, D_in=784, H=256, D_out=10):
        super().__init__()
        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H, D_out)
        # He init on weights, zero biases — matches our C++.
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

torch.manual_seed(42)
model = MLP()
opt = torch.optim.SGD(model.parameters(), lr=0.1)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params}")

# ----- Training -----
B = 128
epochs = 10
train_tx = torch.from_numpy(train_x)
train_ty = torch.from_numpy(train_y)
test_tx = torch.from_numpy(test_x)
test_ty = torch.from_numpy(test_y)
N = len(train_tx)

for epoch in range(epochs):
    perm = torch.randperm(N)
    steps = N // B
    running_loss, running_count = 0.0, 0
    for step in range(steps):
        idx = perm[step * B : (step + 1) * B]
        x, y = train_tx[idx], train_ty[idx]
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        running_loss += loss.item(); running_count += 1
        if (step + 1) % 100 == 0:
            print(f"  epoch {epoch+1}  step {step+1:4d}/{steps}  "
                  f"loss {running_loss/running_count:.4f}")
            running_loss, running_count = 0.0, 0
    with torch.no_grad():
        preds = model(test_tx).argmax(dim=1)
        acc = (preds == test_ty).float().mean().item()
    print(f"  --- epoch {epoch+1} complete ---")
    print(f"  test accuracy: {acc:.4f}\n")