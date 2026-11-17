"""FastAPI web server for the bodybuilding assistant.

Run:  python server.py   (then open http://127.0.0.1:8000)

Shares SYSTEM_PROMPT and chat_memory.json with the CLI version, so a chat
started in the terminal continues in the browser and vice versa.
"""

import os
import threading
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field

from bodybuilding_assistant import MEMORY_FILE, SYSTEM_PROMPT, load_memory, save_memory

load_dotenv()
API_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
STATIC_DIR = Path(__file__).with_name("static")

client = OpenAI(api_key=API_KEY) if API_KEY else None
memory_lock = threading.Lock()  # one writer at a time for chat_memory.json

app = FastAPI(title="Bodybuilding Coach")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# Let the page work when opened through a VS Code preview / Live Server on another
# local port. Only localhost origins are allowed, so other websites can't use your key.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(https?://(localhost|127\.0\.0\.1)(:\d+)?|vscode-webview://[^/]+)$",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


LANGUAGE_NAMES = {"en": "English", "fa": "Persian (Farsi)", "de": "German"}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    lang: Literal["en", "fa", "de"] = "en"  # language selected in the UI


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/history")
def history() -> list[dict]:
    return load_memory()


@app.post("/api/reset")
def reset() -> dict:
    with memory_lock:
        MEMORY_FILE.unlink(missing_ok=True)
    return {"ok": True}


@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    if client is None:
        raise HTTPException(500, "API_KEY not found. Add 'API_KEY=sk-...' to your .env file.")

    question = req.message.strip()
    past = load_memory()
    system = (f"{SYSTEM_PROMPT}\n\nThe user's interface is set to {LANGUAGE_NAMES[req.lang]}. "
              "If the language of their message is unclear (e.g. only numbers or a "
              "single exercise name), reply in that interface language.")
    messages = [{"role": "system", "content": system}, *past,
                {"role": "user", "content": question}]

    try:
        stream = client.chat.completions.create(model=MODEL, messages=messages, stream=True)
    except Exception as exc:  # auth / network / rate-limit errors before streaming starts
        raise HTTPException(502, f"OpenAI error: {exc}") from exc

    def generate():
        parts: list[str] = []
        try:
            for chunk in stream:
                if chunk.choices and (delta := chunk.choices[0].delta.content):
                    parts.append(delta)
                    yield delta
        except Exception as exc:
            yield f"\n\n[Connection error: {exc}]"
            return  # don't save a broken answer
        answer = "".join(parts)
        if answer:
            with memory_lock:
                save_memory(load_memory() + [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ])

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


def open_browser_when_ready(url: str, host: str, port: int, timeout: float = 30.0) -> None:
    """Wait until the server accepts connections, then open the browser."""
    import socket
    import time
    import webbrowser

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                webbrowser.open(url)
                return
        except OSError:
            time.sleep(0.3)


if __name__ == "__main__":
    import sys

    import uvicorn

    HOST, PORT = "127.0.0.1", 8000
    if "--open" in sys.argv:
        threading.Thread(target=open_browser_when_ready,
                         args=(f"http://{HOST}:{PORT}", HOST, PORT), daemon=True).start()
    uvicorn.run(app, host=HOST, port=PORT)
