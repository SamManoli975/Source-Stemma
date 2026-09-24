"""
web_llm.py — local LLM + web search, with a Claude-style cited source list.

How it works, in three steps:
  1. SEARCH   — if a Tavily API key is set, use Tavily (returns fuller
               extracted page content, better citations). Otherwise fall
               back to DuckDuckGo via the `ddgs` library — free, no signup,
               no key at all, just shorter snippets.
  2. ANSWER   — build a prompt that hands the LLM those numbered sources and
               tells it to cite them inline as [1], [2], etc., and to say so
               rather than guess when the sources don't cover something.
  3. CITE     — print the answer, then a "Sources" list mapping each [n] back
               to its title and URL — the same shape as a Claude citation list.

Requirements:
    pip install llama-cpp-python ddgs requests

Setup:
    1. Download a GGUF model (e.g. a Qwen3-8B-Instruct GGUF from Hugging Face)
       and either set the LOCAL_LLM_MODEL environment variable to its path,
       or pass --model /path/to/model.gguf on the command line.
    2. (Optional, but recommended if you have one) Set a Tavily API key as
       the TAVILY_API_KEY environment variable, or pass --tavily-key. If you
       don't set one, the script automatically uses DuckDuckGo instead — no
       key needed either way, so this step is skippable.

Usage:
    python web_llm.py "what is the current status of the X project"
    python web_llm.py                      # interactive: type questions, Ctrl-C to quit
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

import requests
from ddgs import DDGS
from llama_cpp import Llama

# ── config ──────────────────────────────────────────────────────────────

TAVILY_URL = "https://api.tavily.com/search"
DEFAULT_MODEL_PATH = os.environ.get("LOCAL_LLM_MODEL", "")
DEFAULT_N_CTX = 8192
MAX_SOURCES = 5


# ── step 1: search ─────────────────────────────────────────────────────

def web_search_tavily(query: str, api_key: str, max_results: int = MAX_SOURCES) -> list[dict]:
    """Query Tavily and return a list of {title, url, content} dicts.
    Content here is fuller extracted page text, not just a snippet."""
    resp = requests.post(
        TAVILY_URL,
        json={
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "include_raw_content": False,
        },
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()

    results = []
    for r in data.get("results", []):
        results.append({
            "title": (r.get("title") or r.get("url") or "").strip()[:200],
            "url": r.get("url", ""),
            "content": (r.get("content") or "").strip(),
        })
    return results


def web_search_ddg(query: str, max_results: int = MAX_SOURCES) -> list[dict]:
    """Query DuckDuckGo and return a list of {title, url, content} dicts.

    No API key required. `content` here is DuckDuckGo's result snippet
    (a sentence or two), not the full page — shorter than Tavily's, but
    enough for the model to answer and cite from."""
    raw = DDGS().text(query, max_results=max_results)

    results = []
    for r in raw:
        results.append({
            "title": (r.get("title") or r.get("href") or "").strip()[:200],
            "url": r.get("href", ""),
            "content": (r.get("body") or "").strip(),
        })
    return results


def web_search(query: str, tavily_key: str = "", max_results: int = MAX_SOURCES) -> list[dict]:
    """Use Tavily if a key is available, otherwise fall back to DuckDuckGo."""
    if tavily_key:
        return web_search_tavily(query, tavily_key, max_results)
    return web_search_ddg(query, max_results)


# ── step 2: build the prompt and ask the local model ────────────────────

def build_prompt(query: str, sources: list[dict]) -> str:
    numbered = "\n\n".join(
        f"[{i + 1}] {s['title']}\nURL: {s['url']}\n{s['content'][:1200]}"
        for i, s in enumerate(sources)
    )
    system = (
        "You are a careful research assistant. Answer the user's question using ONLY "
        "the numbered sources below. After every claim, cite the source(s) it came from "
        "using bracketed numbers like [1] or [2][3]. If the sources don't cover something, "
        "say so plainly rather than guessing or inventing facts. Do not invent sources."
    )
    return (
        f"{system}\n\n"
        f"SOURCES:\n{numbered}\n\n"
        f"QUESTION: {query}\n\n"
        f"ANSWER (with inline [n] citations):"
    )


def run_local_llm(model_path: str, prompt: str,
                   n_ctx: int = DEFAULT_N_CTX, max_tokens: int = 800) -> str:
    llm = Llama(model_path=model_path, n_ctx=n_ctx, verbose=False)
    out = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.2,
        stop=["QUESTION:", "SOURCES:"],
    )
    return out["choices"][0]["text"].strip()


# ── step 3: orchestrate + print the answer with its source list ─────────

def answer(query: str, model_path: str, tavily_key: str = "",
           max_sources: int = MAX_SOURCES) -> None:
    print(f"\nSearching the web for: {query!r} ...", file=sys.stderr)
    sources = web_search(query, tavily_key, max_sources)
    if not sources:
        print("No search results found — try rephrasing the question.", file=sys.stderr)
        return

    print(f"Found {len(sources)} source(s). Loading local model "
          f"({model_path}) ...", file=sys.stderr)
    prompt = build_prompt(query, sources)
    result = run_local_llm(model_path, prompt)

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(textwrap.fill(result, width=88))

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)
    for i, s in enumerate(sources, 1):
        print(f"[{i}] {s['title']}")
        print(f"    {s['url']}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    global MAX_SOURCES

    parser = argparse.ArgumentParser(
        description="Local LLM + web search, with a Claude-style cited source list."
    )
    parser.add_argument("query", nargs="*",
                         help="question to ask; omit for interactive mode")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH,
                         help="path to a GGUF model file (or set LOCAL_LLM_MODEL)")
    parser.add_argument("--tavily-key", default=os.environ.get("TAVILY_API_KEY", ""),
                         help="Tavily API key (or set TAVILY_API_KEY). Optional — "
                              "falls back to DuckDuckGo if omitted.")
    parser.add_argument("--max-sources", type=int, default=MAX_SOURCES,
                         help=f"how many search results to fetch (default {MAX_SOURCES})")
    args = parser.parse_args()

    if not args.model:
        print("ERROR: no model path given. Pass --model /path/to/model.gguf "
              "or set the LOCAL_LLM_MODEL environment variable.", file=sys.stderr)
        sys.exit(1)

    MAX_SOURCES = args.max_sources

    print(f"Search backend: {'Tavily' if args.tavily_key else 'DuckDuckGo (no key set)'}",
          file=sys.stderr)

    if args.query:
        answer(" ".join(args.query), args.model, args.tavily_key)
    else:
        print("Interactive mode. Type a question and press enter (Ctrl-C to quit).\n")
        while True:
            try:
                q = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if q:
                answer(q, args.model, args.tavily_key)


if __name__ == "__main__":
    main()
