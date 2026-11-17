import os
from pathlib import Path

# Keep everything on this drive, next to this script (not in C:\Users\...\.cache).
BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(BASE_DIR / "data" / "hf_home"))

from datasets import load_dataset  # noqa: E402  (must come after HF_HOME is set)

CACHE_DIR = BASE_DIR / "data" / "hf_cache"
SAVE_DIR = BASE_DIR / "data" / "amazon-reviews-sentiment-analysis"

ds = load_dataset(
    "hugginglearners/amazon-reviews-sentiment-analysis",
    cache_dir=str(CACHE_DIR),
)
ds.save_to_disk(str(SAVE_DIR))
print(ds)
print(f"Saved to {SAVE_DIR}")

# Later, load it back without downloading:
# from datasets import load_from_disk
# ds = load_from_disk("data/amazon-reviews-sentiment-analysis")
