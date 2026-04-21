"""Read/write tensors in our binary format. The same format C++ uses."""
import struct
import numpy as np
import torch
from pathlib import Path

MAGIC = 0x544E5352  # 'TNSR'

def save_tensor(path, t):
    """Save a torch tensor or numpy array to a .bin file."""
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    t = np.ascontiguousarray(t, dtype=np.float32)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', MAGIC))
        f.write(struct.pack('<I', t.ndim))
        for s in t.shape:
            f.write(struct.pack('<I', s))
        f.write(t.tobytes())

def load_tensor(path):
    """Load a .bin file as a numpy array."""
    with open(path, 'rb') as f:
        magic, = struct.unpack('<I', f.read(4))
        assert magic == MAGIC, f"Bad magic in {path}: 0x{magic:x}"
        ndim, = struct.unpack('<I', f.read(4))
        shape = tuple(struct.unpack('<I', f.read(4))[0] for _ in range(ndim))
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data.reshape(shape)

if __name__ == "__main__":
    # Quick self-test
    import tempfile
    a = torch.randn(2, 3)
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        path = f.name
    save_tensor(path, a)
    b = load_tensor(path)
    assert np.allclose(a.numpy(), b), "round-trip failed"
    print("io.py self-test: PASS")