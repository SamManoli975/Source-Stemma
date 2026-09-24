"""
web_search_ollama.py — Ask a question, get an answer with cited sources.

Uses Ollama (local) for the LLM and DuckDuckGo for web search.
No Docker, no API keys, no signup required.

Setup:
    1. Install Ollama from https://ollama.com/download
    2. Pull a model:  ollama pull qwen3:4b
    3. pip install ddgs requests
    4. python web_search_ollama.py "your question here"
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import requests
from ddgs import DDGS

# ── config ──────────────────────────────────────────────────────────────

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3:4b"
MAX_SOURCES = 5


# ── step 1: search the web ─────────────────────────────────────────────

def web_search(query: str, max_results: int = MAX_SOURCES) -> list[dict]:
    """Search DuckDuckGo and return a list of {title, url, content} dicts."""
    raw = DDGS().text(query, max_results=max_results)
    results = []
    for r in raw:
        results.append({
            "title": (r.get("title") or r.get("href") or "").strip()[:200],
            "url": r.get("href", ""),
            "content": (r.get("body") or "").strip(),
        })
    return results


# ── step 2: ask the local model via Ollama (streaming, chat API) ──────

SYSTEM_PROMPT = (
    "You are a careful research assistant. Answer the user's question "
    "using ONLY the numbered sources provided. After every claim, cite "
    "the source(s) using bracketed numbers like [1] or [2][3]. If the "
    "sources don't cover something, say so rather than guessing. "
    "Be concise and direct."
)


def ask_ollama_stream(user_msg: str, model: str = DEFAULT_MODEL) -> tuple[str, dict]:
    """Stream tokens from Ollama chat API, printing them live."""
    try:
        resp = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "stream": True,
                "think": False,   # disable qwen3 thinking mode
                "options": {
                    "temperature": 0.2,
                    "num_predict": 1024,
                },
            },
            stream=True,
            timeout=(30, 600),
        )
        resp.raise_for_status()
    except requests.ConnectionError:
        print("ERROR: Cannot connect to Ollama. Is it running?", file=sys.stderr)
        print("  Start it with:  ollama serve", file=sys.stderr)
        sys.exit(1)
    except requests.HTTPError as e:
        print(f"ERROR from Ollama: {e}", file=sys.stderr)
        sys.exit(1)

    chunks = []
    stats = {}

    for line in resp.iter_lines():
        if not line:
            continue
        data = json.loads(line)

        if data.get("done"):
            stats = data
            break

        token = data.get("message", {}).get("content", "")
        if token:
            chunks.append(token)
            sys.stdout.write(token)
            sys.stdout.flush()

    full_response = "".join(chunks)
    # safety net: strip any think tags that slipped through
    cleaned = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
    return cleaned, stats


def build_user_message(query: str, sources: list[dict]) -> str:
    """Build the user message with numbered sources."""
    numbered = "\n\n".join(
        f"[{i+1}] {s['title']}\nURL: {s['url']}\n{s['content'][:600]}"
        for i, s in enumerate(sources)
    )
    return f"SOURCES:\n{numbered}\n\nQUESTION: {query}"


# ── step 3: orchestrate and print ──────────────────────────────────────

def answer(query: str, model: str, max_sources: int = MAX_SOURCES) -> None:
    print(f"\n  Searching the web for: {query!r} ...", file=sys.stderr)
    sources = web_search(query, max_sources)

    if not sources:
        print("  No results found — try rephrasing.", file=sys.stderr)
        return

    print(f"  Found {len(sources)} source(s). Asking {model} ...\n", file=sys.stderr)
    user_msg = build_user_message(query, sources)

    print("=" * 70)
    print("  ANSWER")
    print("=" * 70)

    result, stats = ask_ollama_stream(user_msg, model)

    print()  # newline after streamed answer

    # ── print the source list ──
    print("\n" + "=" * 70)
    print("  SOURCES")
    print("=" * 70)
    for i, s in enumerate(sources, 1):
        print(f"  [{i}] {s['title']}")
        print(f"      {s['url']}")

    # ── print stats ──
    total_ns = stats.get("total_duration", 0)
    prompt_tokens = stats.get("prompt_eval_count", 0)
    output_tokens = stats.get("eval_count", 0)
    eval_ns = stats.get("eval_duration", 0)
    load_ns = stats.get("load_duration", 0)

    total_secs = total_ns / 1e9
    eval_secs = eval_ns / 1e9
    load_secs = load_ns / 1e9
    tok_per_sec = output_tokens / eval_secs if eval_secs > 0 else 0

    print("\n" + "=" * 70)
    print("  STATS")
    print("=" * 70)
    print(f"  Prompt tokens:    {prompt_tokens}")
    print(f"  Output tokens:    {output_tokens}")
    print(f"  Total tokens:     {prompt_tokens + output_tokens}")
    print(f"  Generation speed: {tok_per_sec:.1f} tokens/sec")
    print(f"  Model load time:  {load_secs:.1f}s")
    print(f"  Generation time:  {eval_secs:.1f}s")
    print(f"  Total time:       {total_secs:.1f}s")
    print()


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Local LLM + web search with cited sources (via Ollama)."
    )
    parser.add_argument("query", nargs="*",
                        help="Question to ask (omit for interactive mode)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Ollama model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--max-sources", type=int, default=MAX_SOURCES,
                        help=f"Number of search results to fetch (default: {MAX_SOURCES})")
    args = parser.parse_args()

    # Check Ollama is reachable
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if not models:
            print("WARNING: No models installed. Run:  ollama pull qwen3:4b",
                  file=sys.stderr)
            sys.exit(1)
        if args.model not in models and f"{args.model}:latest" not in models:
            print(f"WARNING: Model '{args.model}' not found. Available: {', '.join(models)}",
                  file=sys.stderr)
            print(f"  Pull it with:  ollama pull {args.model}", file=sys.stderr)
            sys.exit(1)
    except requests.ConnectionError:
        print("ERROR: Ollama is not running. Start it with:  ollama serve",
              file=sys.stderr)
        sys.exit(1)

    print(f"  Model: {args.model}", file=sys.stderr)
    print(f"  Search: DuckDuckGo (no API key needed)", file=sys.stderr)

    if args.query:
        answer(" ".join(args.query), args.model, args.max_sources)
    else:
        print("\n  Interactive mode — type a question, press Enter. Ctrl-C to quit.\n")
        while True:
            try:
                q = input("  > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if q:
                answer(q, args.model, args.max_sources)


if __name__ == "__main__":
    main()
