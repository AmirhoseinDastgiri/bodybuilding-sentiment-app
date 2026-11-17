"""Train several non-generative sentiment classifiers, pick the best, save it.

Run:  bodyvenv/Scripts/python.exe train_sentiment.py [--skip-transformer]

Candidates
  * TF-IDF (words + char n-grams) + Logistic Regression
  * TF-IDF + Linear SVM (probability-calibrated)
  * TF-IDF + Complement Naive Bayes
  * RoBERTa sentiment model, used as-is (zero-shot, no training)
  * RoBERTa sentiment model, fine-tuned on these reviews (GPU if available)

Selection: highest macro-F1 on the validation split (macro-F1 treats the rare
negative/neutral classes as important as the dominant positive one). The test
split is only used to report final numbers. The winner is saved to models/sentiment/.
"""

import argparse
import json
import shutil
import time

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

from sentiment_common import CONFIDENCE_THRESHOLD, LABELS, META_FILE, MODEL_DIR, SEED, load_splits

HF_BASE_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"  # 3-class sentiment encoder


# ---------------------------------------------------------------- helpers
def tfidf():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
    ])


def evaluate(probs: np.ndarray, y: np.ndarray) -> dict:
    """probs columns follow LABELS order; y holds label names."""
    pred = np.array(LABELS)[probs.argmax(1)]
    conf = probs.max(1)
    sure = conf >= CONFIDENCE_THRESHOLD
    return {
        "macro_f1": round(f1_score(y, pred, average="macro"), 4),
        "accuracy": round(accuracy_score(y, pred), 4),
        "confident_share": round(float(sure.mean()), 4),                   # handled locally
        "confident_accuracy": round(float((pred[sure] == y[sure]).mean()) if sure.any() else 0.0, 4),
    }


def sklearn_probs(model, texts) -> np.ndarray:
    p = model.predict_proba(texts)
    order = [list(model.classes_).index(lbl) for lbl in LABELS]
    return p[:, order]


# ---------------------------------------------------------------- transformer
class HFClassifier:
    """Thin wrapper so a Hugging Face model looks like the sklearn ones (probs in LABELS order)."""

    def __init__(self, path_or_name):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(path_or_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(path_or_name).to(self.device)
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.label_ids = [next(i for i, n in id2label.items() if n == lbl) for lbl in LABELS]

    def encode(self, texts):
        return self.tok(list(texts), truncation=True, max_length=256, padding=True, return_tensors="pt").to(self.device)

    def predict_proba(self, texts, batch_size=64) -> np.ndarray:
        torch = self.torch
        self.model.eval()
        out = []
        with torch.no_grad(), torch.autocast(self.device, enabled=self.device == "cuda"):
            for i in range(0, len(texts), batch_size):
                logits = self.model(**self.encode(texts[i:i + batch_size])).logits.float()
                out.append(torch.softmax(logits, -1)[:, self.label_ids].cpu().numpy())
        return np.concatenate(out)

    def fine_tune(self, train, val, epochs=4, lr=2e-5, batch_size=16):
        torch = self.torch
        from transformers import get_linear_schedule_with_warmup

        torch.manual_seed(SEED)
        texts = train["text"].tolist()
        # model's own ids for each training row
        name_to_id = dict(zip(LABELS, self.label_ids))
        y = torch.tensor([name_to_id[l] for l in train["label"]])

        # softened inverse-frequency class weights: helps the rare classes without over-shooting
        counts = torch.bincount(y, minlength=len(LABELS)).float()
        weights = ((counts.sum() / (len(counts) * counts.clamp(min=1))) ** 0.5).to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        steps = epochs * ((len(texts) + batch_size - 1) // batch_size)
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        scaler = torch.amp.GradScaler(enabled=self.device == "cuda")

        best_f1, best_state = -1.0, None
        for epoch in range(epochs):
            self.model.train()
            perm = torch.randperm(len(texts))
            total = 0.0
            for i in range(0, len(texts), batch_size):
                idx = perm[i:i + batch_size]
                batch = self.encode([texts[j] for j in idx])
                with torch.autocast(self.device, enabled=self.device == "cuda"):
                    loss = loss_fn(self.model(**batch).logits.float(), y[idx].to(self.device))
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                total += loss.item() * len(idx)
            m = evaluate(self.predict_proba(val["text"].tolist()), val["label"].to_numpy())
            print(f"    epoch {epoch + 1}/{epochs}  loss {total / len(texts):.4f}  val {m}")
            if m["macro_f1"] > best_f1:
                best_f1 = m["macro_f1"]
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(best_state)
        return self


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-transformer", action="store_true", help="only train the TF-IDF models")
    args = ap.parse_args()

    train, val, test = load_splits()
    print(f"train {len(train)}  val {len(val)}  test {len(test)}")
    print("train label counts:", train["label"].value_counts().to_dict())
    Xtr, ytr = train["text"].tolist(), train["label"].to_numpy()

    candidates = {}   # name -> (kind, model)
    sk_models = {
        "tfidf_logreg": Pipeline([("tfidf", tfidf()), ("clf", LogisticRegression(
            C=10, class_weight="balanced", max_iter=3000))]),
        "tfidf_linearsvm": Pipeline([("tfidf", tfidf()), ("clf", CalibratedClassifierCV(
            LinearSVC(C=0.5, class_weight="balanced"), cv=5))]),
        "tfidf_complement_nb": Pipeline([("tfidf", tfidf()), ("clf", ComplementNB(alpha=0.3))]),
    }
    for name, model in sk_models.items():
        t = time.time()
        model.fit(Xtr, ytr)
        print(f"  trained {name} in {time.time() - t:.1f}s")
        candidates[name] = ("sklearn", model)

    if not args.skip_transformer:
        t = time.time()
        zero = HFClassifier(HF_BASE_MODEL)
        print(f"  loaded {HF_BASE_MODEL} on {zero.device}")
        candidates["roberta_zero_shot"] = ("transformer", zero)
        print("  fine-tuning roberta ...")
        tuned = HFClassifier(HF_BASE_MODEL).fine_tune(train, val)
        print(f"  fine-tuned in {time.time() - t:.0f}s")
        candidates["roberta_finetuned"] = ("transformer", tuned)

    results = {}
    for name, (kind, model) in candidates.items():
        pv = sklearn_probs(model, val["text"].tolist()) if kind == "sklearn" else model.predict_proba(val["text"].tolist())
        pt = sklearn_probs(model, test["text"].tolist()) if kind == "sklearn" else model.predict_proba(test["text"].tolist())
        results[name] = {"val": evaluate(pv, val["label"].to_numpy()), "test": evaluate(pt, test["label"].to_numpy()),
                         "_test_probs": pt}

    print("\n%-22s %10s %10s %10s %12s" % ("model", "val F1", "test F1", "test acc", "local share"))
    for name, r in results.items():
        print("%-22s %10.4f %10.4f %10.4f %12.1f%%" % (name, r["val"]["macro_f1"], r["test"]["macro_f1"],
                                                     r["test"]["accuracy"], 100 * r["test"]["confident_share"]))

    best = max(results, key=lambda n: results[n]["val"]["macro_f1"])
    kind, model = candidates[best]
    print(f"\nBEST: {best}")
    pt = results[best]["_test_probs"]
    print(classification_report(test["label"], np.array(LABELS)[pt.argmax(1)], digits=3))

    # save the winner
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    MODEL_DIR.mkdir(parents=True)
    if kind == "sklearn":
        joblib.dump(model, MODEL_DIR / "model.joblib")
    else:
        model.model.save_pretrained(MODEL_DIR / "hf")
        model.tok.save_pretrained(MODEL_DIR / "hf")
    meta = {
        "best_model": best,
        "kind": kind,
        "base_model": HF_BASE_MODEL if kind == "transformer" else None,
        "labels": LABELS,
        "threshold": CONFIDENCE_THRESHOLD,
        "results": {n: {k: v for k, v in r.items() if not k.startswith("_")} for n, r in results.items()},
    }
    META_FILE.write_text(json.dumps(meta, indent=2))
    print(f"saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
