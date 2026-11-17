"""Shared pieces for the Amazon-review sentiment models: labels, paths, data split.

Output codes (what the app returns):  1 = positive, 2 = negative, 3 = neutral.
Star rating -> label:  4-5 stars = positive, 3 = neutral, 1-2 = negative.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
# Keep Hugging Face downloads (datasets + models) on this drive, not in C:\Users\...
os.environ.setdefault("HF_HOME", str(BASE_DIR / "data" / "hf_home"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

DATASET_DIR = BASE_DIR / "data" / "amazon-reviews-sentiment-analysis"
MODEL_DIR = BASE_DIR / "models" / "sentiment"   # the chosen ("best") local model
META_FILE = MODEL_DIR / "meta.json"

LABELS = ["positive", "negative", "neutral"]      # index i  ->  code i + 1
LABEL_CODES = {name: i + 1 for i, name in enumerate(LABELS)}

# If the local model's top probability is below this, the review goes to the LLM.
CONFIDENCE_THRESHOLD = 0.80

SEED = 42


def stars_to_label(stars: float) -> str:
    if stars >= 4:
        return "positive"
    if stars <= 2:
        return "negative"
    return "neutral"


def load_splits():
    """Return (train, val, test) DataFrames with columns text, label (stratified 70/15/15)."""
    from datasets import load_from_disk
    from sklearn.model_selection import train_test_split

    df = load_from_disk(str(DATASET_DIR))["train"].to_pandas()
    df = df.assign(text=df["reviewText"].fillna("").str.strip(), label=df["overall"].map(stars_to_label))
    df = df.loc[df["text"].str.len() > 0, ["text", "label"]].reset_index(drop=True)

    train, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=SEED)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
