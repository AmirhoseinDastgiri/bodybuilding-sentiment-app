"""Hybrid Amazon-review sentiment web app.

Run:  bodyvenv/Scripts/python.exe sentiment_app.py   (then open http://127.0.0.1:8001)
Needs a trained model first:  bodyvenv/Scripts/python.exe train_sentiment.py
"""

import argparse
import threading
import webbrowser

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from sentiment_common import BASE_DIR
from sentiment_hybrid import HybridSentiment

PAGE = BASE_DIR / "static" / "sentiment.html"
PORT = 8001

hybrid = HybridSentiment()
predict_lock = threading.Lock()  # the local model is not guaranteed thread-safe (GPU)

app = FastAPI(title="Hybrid Review Sentiment")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(https?://(localhost|127\.0\.0\.1)(:\d+)?|vscode-webview://[^/]+)$",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class SentimentRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(PAGE)


@app.get("/api/info")
def info() -> dict:
    meta = hybrid.local.meta
    return {
        "local_model": meta["best_model"],
        "threshold": hybrid.threshold,
        "llm_model": hybrid.llm_model,
        "llm_enabled": hybrid.client is not None,
        "test_metrics": meta["results"][meta["best_model"]]["test"],
        "codes": {"1": "positive", "2": "negative", "3": "neutral"},
    }


def _predict(text: str) -> dict:
    with predict_lock:
        probs = hybrid.local.predict_proba([text])[0]
    return hybrid.predict(text, probs)


@app.post("/api/sentiment")
async def sentiment(req: SentimentRequest) -> dict:
    text = req.text.strip()
    if not text:
        raise HTTPException(422, "Empty review")
    return await run_in_threadpool(_predict, text)


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true", help="open the browser once the server is up")
    if ap.parse_args().open:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    uvicorn.run(app, host="127.0.0.1", port=PORT)
