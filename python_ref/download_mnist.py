"""Download MNIST to data/mnist/. Uses a stable mirror (CVDF on Google Cloud)."""
import urllib.request, gzip, shutil
from pathlib import Path

BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

out_dir = Path("data/mnist")
out_dir.mkdir(parents=True, exist_ok=True)

for fname in FILES:
    gz_path = out_dir / fname
    bin_path = out_dir / fname[:-3]  # strip .gz

    if bin_path.exists():
        print(f"  skip: {bin_path} already exists")
        continue

    print(f"  downloading {fname}...")
    urllib.request.urlretrieve(BASE + fname, gz_path)

    print(f"  decompressing to {bin_path}")
    with gzip.open(gz_path, "rb") as f_in, open(bin_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()

print("\nDone. MNIST files in data/mnist/")