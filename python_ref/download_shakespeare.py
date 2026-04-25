"""Download tiny-shakespeare to data/tinyshakespeare.txt."""
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
out_path = Path("data/tinyshakespeare.txt")
out_path.parent.mkdir(parents=True, exist_ok=True)

if out_path.exists():
    print(f"  skip: {out_path} already exists")
else:
    print(f"  downloading from {URL}...")
    urllib.request.urlretrieve(URL, out_path)
    print(f"  saved {out_path.stat().st_size} bytes")

# Print stats
text = out_path.read_text()
chars = sorted(set(text))
print(f"\nTotal chars: {len(text):,}")
print(f"Vocab size: {len(chars)}")
print(f"Chars: {''.join(chars)!r}")