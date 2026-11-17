# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Bodybuilding-coach chatbot using the OpenAI Chat Completions API, with two front ends sharing one memory file:
- `bodybuilding_assistant.py` — terminal CLI; also the source of `SYSTEM_PROMPT`, `MEMORY_FILE`, `load_memory`/`save_memory`.
- `server.py` — FastAPI app importing those helpers. Routes: `GET /` (serves `static/index.html`), `GET /api/history`, `POST /api/chat` (streams plain-text tokens; memory saved only after a complete answer), `POST /api/reset`.
- `static/index.html` — single-file UI (inline CSS/JS, marked + DOMPurify from cdnjs for Markdown, light/dark theme, dark by default).
- `static/background.jpg` — brand background photo, drawn by `body::before` with a theme-dependent `--overlay` in `body::after`. Its URL is set from JS (`--bg-image`, prefixed with `API`) so it resolves from VS Code previews too.

### Languages (English / Persian / German)
- The model replies in the language of the user's latest message (rule in `SYSTEM_PROMPT`). `POST /api/chat` takes `lang` (`en`|`fa`|`de`, the UI selection), which `server.py` appends to the system prompt as the fallback language for ambiguous messages.
- UI strings live in the `I18N` object in `index.html`; elements with `data-i18n="key"` are filled by `applyLang()`. Any new UI text needs entries in all three languages.
- Selecting `fa` sets `<html dir="rtl">`; CSS uses logical properties (`padding-inline-start`, `border-start-start-radius`, `text-align: start`) so layout mirrors automatically — keep using them instead of left/right. Message bubbles use `dir="auto"` so mixed-language chats render correctly.
- Language is chosen by `?lang=xx` URL param → saved choice in localStorage → browser language → English.
 Conversation history (user/assistant turns only) persists to `chat_memory.json` (gitignored) after each answer and is reloaded on startup; the system prompt is always injected from code, not stored. Typing `reset` deletes the memory file. No tests or linter are set up.

## Commands

The project uses a virtual environment named `bodyvenv` (gitignored). Use its interpreter directly:

- Create venv: `python -m venv bodyvenv`
- Install deps: `bodyvenv/Scripts/python.exe -m pip install -r requirements.txt`
- Run CLI: `bodyvenv/Scripts/python.exe bodybuilding_assistant.py`
- Run web app: `run.bat` (starts server, opens browser once the port is up via `--open`), F5 in VS Code (`.vscode/launch.json`), or `bodyvenv/Scripts/python.exe server.py`
- The page also works from VS Code previews / other local ports: `index.html` sends API calls to `http://127.0.0.1:8000` unless served from port 8000, and the server's CORS allows only localhost/`vscode-webview` origins. → http://127.0.0.1:8000 (dev reload: `bodyvenv/Scripts/uvicorn.exe server:app --reload`)
- Activate (PowerShell): `.\bodyvenv\Scripts\Activate.ps1`

## Configuration

Read from `.env` (gitignored) via python-dotenv:
- `API_KEY` — OpenAI key (falls back to `OPENAI_API_KEY`)
- `OPENAI_MODEL` — optional, defaults to `gpt-4o-mini`

## Environment notes

- Development happens on Windows; `.gitattributes` normalizes line endings to LF in the repo, so avoid committing CRLF-specific content.
