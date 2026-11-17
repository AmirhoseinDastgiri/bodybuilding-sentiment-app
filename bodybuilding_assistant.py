"""Interactive bodybuilding assistant powered by the OpenAI API.

Reads the API key from a .env file (API_KEY=...) and starts a chat loop
in the terminal. The conversation is saved to chat_memory.json after every
answer, so the next run continues where you left off.

Commands:
  exit / quit  - stop the program (the chat stays saved)
  reset        - forget the saved conversation and start fresh
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

MEMORY_FILE = Path(__file__).with_name("chat_memory.json")

SYSTEM_PROMPT = (
    "You are an expert bodybuilding and fitness coach. You answer questions about "
    "resistance training, workout programming, exercise technique, hypertrophy, "
    "strength, nutrition, macros, supplements, recovery, and contest preparation. "
    "Give practical, evidence-based answers, use concrete numbers (sets, reps, "
    "grams, calories) when helpful, and keep answers clear and well structured. "
    "If a question involves injury, pain, or a medical condition, recommend "
    "consulting a doctor or physiotherapist. If a question is unrelated to "
    "bodybuilding, fitness, or nutrition, politely steer the user back to those topics.\n\n"
    "Languages: you are fluent in English, Persian (Farsi), and German. Always reply in "
    "the language of the user's latest message, even if earlier messages used another "
    "language. Write Persian in natural, fluent Farsi script (not Finglish) and German "
    "in natural German (use 'du'). Keep common gym terms understandable, e.g. add the "
    "English term in parentheses when the local term is less known. Use metric units."
)


def load_memory() -> list[dict]:
    """Return saved user/assistant messages, or an empty list."""
    if not MEMORY_FILE.exists():
        return []
    try:
        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        return [m for m in data if m.get("role") in {"user", "assistant"}]
    except (json.JSONDecodeError, OSError, AttributeError):
        print(f"[warning] Could not read {MEMORY_FILE.name}; starting a new chat.")
        return []


def save_memory(history: list[dict]) -> None:
    """Write the conversation to disk (atomically, so a crash can't corrupt it)."""
    tmp = MEMORY_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(MEMORY_FILE)


def main() -> None:
    load_dotenv()
    api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Error: API_KEY not found. Add a line 'API_KEY=sk-...' to your .env file.")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    # Saved history holds only user/assistant turns; the system prompt is
    # always taken from the code so edits to it apply to old chats too.
    history = load_memory()

    print("Bodybuilding Assistant (type 'exit' to quit, 'reset' to start a new chat)")
    if history:
        print(f"Loaded previous chat ({len(history) // 2} exchanges) from {MEMORY_FILE.name}.")
    print()

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        if question.lower() == "reset":
            history = []
            MEMORY_FILE.unlink(missing_ok=True)
            print("\nMemory cleared. Starting a new chat.\n")
            continue

        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history,
                    {"role": "user", "content": question}]
        try:
            response = client.chat.completions.create(model=model, messages=messages)
        except Exception as exc:  # network/auth/rate-limit errors
            print(f"\n[API error] {exc}\n")
            continue

        answer = response.choices[0].message.content
        history += [{"role": "user", "content": question},
                    {"role": "assistant", "content": answer}]
        save_memory(history)
        print(f"\nCoach: {answer}\n")


if __name__ == "__main__":
    main()
