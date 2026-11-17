"""Evaluate local-only vs hybrid (local + OpenAI fallback) on the held-out test split.

Run:  bodyvenv/Scripts/python.exe evaluate_hybrid.py
Only the reviews the local model is unsure about are sent to OpenAI.
"""

import json

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

from sentiment_common import LABELS, MODEL_DIR, load_splits
from sentiment_hybrid import HybridSentiment


def main():
    _, _, test = load_splits()
    hybrid = HybridSentiment()
    texts, y = test["text"].tolist(), test["label"].to_numpy()

    probs = np.concatenate([hybrid.local.predict_proba(texts[i:i + 32]) for i in range(0, len(texts), 32)])
    local_pred = np.array(LABELS)[probs.argmax(1)]
    unsure = np.where(probs.max(1) < hybrid.threshold)[0]
    print(f"local model: {hybrid.local.meta['best_model']}   threshold {hybrid.threshold:.0%}")
    print(f"{len(unsure)} of {len(texts)} test reviews ({len(unsure) / len(texts):.1%}) go to {hybrid.llm_model}")

    hybrid_pred = local_pred.copy()
    if hybrid.client is None:
        print("No OpenAI key set - hybrid == local.")
    for n, i in enumerate(unsure, 1):
        try:
            res = hybrid.predict(texts[i], probs[i])
            hybrid_pred[i] = res["label"]
        except Exception as exc:
            print("  LLM error:", exc)
        print(f"\r  LLM calls {n}/{len(unsure)}", end="", flush=True)
    print()

    summary = {}
    for name, pred in [("local only", local_pred), ("hybrid", hybrid_pred)]:
        summary[name] = {"macro_f1": round(f1_score(y, pred, average="macro"), 4), "accuracy": round(accuracy_score(y, pred), 4)}
        print(f"\n=== {name}: macro-F1 {summary[name]['macro_f1']:.4f}  accuracy {summary[name]['accuracy']:.4f}")
        print(classification_report(y, pred, digits=3, zero_division=0))
    if len(unsure):
        print(f"on the routed reviews: local acc {(local_pred[unsure] == y[unsure]).mean():.3f}"
              f"  ->  LLM acc {(hybrid_pred[unsure] == y[unsure]).mean():.3f}")
    (MODEL_DIR / "hybrid_eval.json").write_text(json.dumps({**summary, "routed": int(len(unsure)), "test_size": len(texts)}, indent=2))


if __name__ == "__main__":
    main()
