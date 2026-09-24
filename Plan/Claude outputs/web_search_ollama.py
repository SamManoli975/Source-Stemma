"""
web_search_ollama.py — Ask a question, get an answer with cited sources.

Uses Ollama (local) for the LLM and DuckDuckGo for web search.
No Docker, no API keys, no signup required.

Setup:
    1. Install Ollama from https://ollama.com/download
    2. Pull a model:  ollama pull qwen3:8b
    3. pip install ddgs requests
    4. python web_search_ollama.py "your question here"
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap

import requests
from ddgs import DDGS

# ── config ──────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen3:8b"
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


# ── step 2: ask the local model via Ollama ──────────────────────────────

def ask_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Send a prompt to the local Ollama server and return the response."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 1024,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.ConnectionError:
        print("ERROR: Cannot connect to Ollama. Is it running?", file=sys.stderr)
        print("  Start it with:  ollama serve", file=sys.stderr)
        sys.exit(1)
    except requests.HTTPError as e:
        print(f"ERROR from Ollama: {e}", file=sys.stderr)
        print(f"  Response: {resp.text[:500]}", file=sys.stderr)
        sys.exit(1)


def build_prompt(query: str, sources: list[dict]) -> str:
    """Build a prompt with numbered sources, instructing the LLM to cite."""
    numbered = "\n\n".join(
        f"[{i+1}] {s['title']}\nURL: {s['url']}\n{s['content'][:1200]}"
        for i, s in enumerate(sources)
    )
    return (
        "You are a careful research assistant. Answer the user's question "
        "using ONLY the numbered sources below. After every claim, cite the "
        "source(s) using bracketed numbers like [1] or [2][3]. If the sources "
        "don't cover something, say so rather than guessing.\n\n"
        f"SOURCES:\n{numbered}\n\n"
        f"QUESTION: {query}\n\n"
        "ANSWER (with inline [n] citations):"
    )


# ── step 3: orchestrate and print ──────────────────────────────────────

def answer(query: str, model: str, max_sources: int = MAX_SOURCES) -> None:
    print(f"\n  Searching the web for: {query!r} ...", file=sys.stderr)
    sources = web_search(query, max_sources)

    if not sources:
        print("  No results found — try rephrasing.", file=sys.stderr)
        return

    print(f"  Found {len(sources)} source(s). Asking {model} ...", file=sys.stderr)
    prompt = build_prompt(query, sources)
    result = ask_ollama(prompt, model)

    # ── print the answer ──
    print("\n" + "=" * 70)
    print("  ANSWER")
    print("=" * 70)
    for line in result.splitlines():
        print(textwrap.fill(line, width=88) if line.strip() else "")

    # ── print the source list ──
    print("\n" + "=" * 70)
    print("  SOURCES")
    print("=" * 70)
    for i, s in enumerate(sources, 1):
        print(f"  [{i}] {s['title']}")
        print(f"      {s['url']}")
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
            print("WARNING: No models installed. Run:  ollama pull qwen3:8b",
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
