"""Hybrid sentiment decision: local (non-generative) classifier first, OpenAI LLM as the fallback.

    local top probability >= threshold  ->  local model decides
    otherwise                           ->  OpenAI LLM decides (local probabilities are passed as a hint)

Codes:  1 = positive, 2 = negative, 3 = neutral.
"""

import json
import os

import numpy as np
from dotenv import load_dotenv

from sentiment_common import LABEL_CODES, LABELS, META_FILE, MODEL_DIR

load_dotenv()

LLM_SYSTEM_PROMPT = """You are an expert at classifying the sentiment of Amazon product reviews.
Classify the review as exactly one of: positive, negative, neutral.
- positive: the customer is overall satisfied / would recommend (typically 4-5 stars).
- negative: the customer is overall dissatisfied, product failed, returned it (typically 1-2 stars).
- neutral: mixed or lukewarm, "it's okay", pros and cons roughly balanced (typically 3 stars).
Judge the overall verdict, not individual complaints. The review may be in any language.
You may get the guess of a smaller classifier that was unsure; use it only as a weak hint.
Reply with JSON only: {"label": "positive|negative|neutral", "confidence": <0..1>, "reason": "<one short sentence>"}"""


class LocalSentimentModel:
    def __init__(self):
        if not META_FILE.exists():
            raise FileNotFoundError("No trained model found. Run: python train_sentiment.py")
        self.meta = json.loads(META_FILE.read_text())
        self.kind = self.meta["kind"]
        self.threshold = float(os.getenv("SENTIMENT_THRESHOLD", self.meta["threshold"]))
        if self.kind == "sklearn":
            import joblib
            self.model = joblib.load(MODEL_DIR / "model.joblib")
            self._order = [list(self.model.classes_).index(l) for l in LABELS]
        else:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tok = AutoTokenizer.from_pretrained(MODEL_DIR / "hf")
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR / "hf").to(self.device).eval()
            id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
            self._order = [next(i for i, n in id2label.items() if n == l) for l in LABELS]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Probabilities, columns in LABELS order."""
        if self.kind == "sklearn":
            return self.model.predict_proba(texts)[:, self._order]
        torch = self.torch
        enc = self.tok(texts, truncation=True, max_length=256, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(**enc).logits.float(), -1)
        return probs[:, self._order].cpu().numpy()


class HybridSentiment:
    def __init__(self):
        self.local = LocalSentimentModel()
        key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        self.llm_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if key:
            from openai import OpenAI
            self.client = OpenAI(api_key=key)
        else:
            self.client = None

    @property
    def threshold(self) -> float:
        return self.local.threshold

    def ask_llm(self, text: str, local_probs: dict) -> dict:
        hint = ", ".join(f"{k} {v:.0%}" for k, v in local_probs.items())
        resp = self.client.chat.completions.create(
            model=self.llm_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": f"Small classifier (unsure): {hint}\n\nReview:\n{text}"},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        label = str(data.get("label", "")).lower().strip()
        if label not in LABEL_CODES:
            raise ValueError(f"LLM returned an unknown label: {label!r}")
        conf = data.get("confidence")
        return {"label": label, "confidence": float(conf) if conf is not None else None,
                "reason": data.get("reason", ""), "model": self.llm_model}

    def predict(self, text: str, probs: np.ndarray | None = None) -> dict:
        """`probs` lets the caller run the local model itself (e.g. under a lock)."""
        if probs is None:
            probs = self.local.predict_proba([text])[0]
        local_probs = {l: round(float(p), 4) for l, p in zip(LABELS, probs)}
        local_label = LABELS[int(probs.argmax())]
        local_conf = float(probs.max())

        result = {
            "local": {"label": local_label, "code": LABEL_CODES[local_label], "confidence": round(local_conf, 4),
                      "probabilities": local_probs, "model": self.local.meta["best_model"]},
            "threshold": self.threshold,
            "llm": None,
            "note": None,
        }
        decided_by, label, conf = "local", local_label, local_conf

        if local_conf < self.threshold:
            if self.client is None:
                result["note"] = "Local model was unsure, but no OpenAI API key is set, so its guess is used."
            else:
                try:
                    llm = self.ask_llm(text, local_probs)
                    result["llm"] = llm
                    decided_by, label, conf = "llm", llm["label"], llm["confidence"]
                except Exception as exc:  # network / quota / bad JSON -> keep local guess
                    result["note"] = f"LLM call failed ({type(exc).__name__}: {exc}); local guess used."

        result.update(code=LABEL_CODES[label], label=label, decided_by=decided_by,
                      confidence=round(conf, 4) if conf is not None else None)
        return result
