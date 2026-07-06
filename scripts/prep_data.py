"""Download the corpus (if missing) and smoke-test the data pipeline."""
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wordfusion.config import DEFAULT as cfg
from wordfusion.data import ShakespeareWindows, tensor_to_rgb
from wordfusion.codec import decode_image, NUM_TOKENS

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

os.makedirs("data", exist_ok=True)
if not os.path.exists(cfg.corpus_path):
    print("downloading corpus...")
    urllib.request.urlretrieve(URL, cfg.corpus_path)
sz = os.path.getsize(cfg.corpus_path)
raw = open(cfg.corpus_path, encoding="utf-8").read()
print(f"corpus: {sz} bytes, {len(raw)} chars, {len(set(raw))} unique")
print(f"vocab size: {NUM_TOKENS}")

ds = ShakespeareWindows(cfg.corpus_path, cfg.img_h, cfg.img_w, epoch_size=100)
rgb, idx = ds[0]
print(f"sample rgb {tuple(rgb.shape)} range [{rgb.min():.2f}, {rgb.max():.2f}]  idx {tuple(idx.shape)}")

# round-trip: rgb tensor -> uint8 -> text, and compare to idx-derived text
from wordfusion.codec import idx_to_text
text_from_idx = idx_to_text(idx.numpy())
text_from_rgb = decode_image(tensor_to_rgb(rgb))
print("--- sample text window ---")
print(repr(text_from_idx[:120]))
print("rgb-decode matches idx-decode:", text_from_rgb == text_from_idx)
