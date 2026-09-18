"""
localize.py — find the shared span between two texts of unequal scope,
BEFORE any variant analysis, and report which kind of overlap it is.

The problem this solves: classical stemmatics compares parallel witnesses
that line up start-to-finish. Real inputs often don't — one document is long
and contains, somewhere inside it, a passage that the other (short) document
also carries. Collating them whole measures "different lengths," not descent.

So we localize first. Two tiers, tried in order. The tier that succeeds is
itself the provenance signal:

  TIER 1  token-level local alignment (Smith-Waterman style).
          Succeeds when the overlap is near-VERBATIM.
          -> real shared TEXT. Variant analysis is meaningful.

  TIER 2  embedding-based sentence matching (fallback when tier 1 is weak).
          Succeeds when the overlap is a PARAPHRASE.
          -> shared CONTENT, not shared wording. This is the homoplasy /
             shared-fact case. Variant analysis would manufacture noise;
             the tool says so instead of pretending.

  NEITHER no real overlap. The short text is not a snippet of the long one.

The verdict routes the pair. Only VERBATIM overlaps are handed downstream
to the variant/conjunctive-error analysis.
"""

from __future__ import annotations
from dataclasses import dataclass
import re

# ----------------------------------------------------------------------
# Tokenisation (shared grain with pairwise.py)
# ----------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)

def sentences(text: str) -> list[str]:
    # cheap sentence split; good enough for localisation
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


# ----------------------------------------------------------------------
# Result type
# ----------------------------------------------------------------------

@dataclass
class SpanResult:
    verdict: str           # "verbatim" | "paraphrase" | "no_overlap"
    score: float           # tier-appropriate match score (0..1)
    long_span: str         # excised passage from the long text ('' if none)
    short_span: str        # matched region of the short text ('' if none)
    detail: str            # human-readable explanation

    @property
    def comparable(self) -> bool:
        """Only verbatim overlap yields text comparable for variant analysis."""
        return self.verdict == "verbatim"


# ----------------------------------------------------------------------
# TIER 1 — token-level local alignment (Smith-Waterman)
# ----------------------------------------------------------------------

def _smith_waterman(a: list[str], b: list[str],
                    match=2, mismatch=-1, gap=-1):
    """Local alignment over token sequences. Returns (score, i0,i1, j0,j1)
    giving the best-matching subregion a[i0:i1] vs b[j0:j1]."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0, 0, 0, 0, 0
    H = [[0] * (m + 1) for _ in range(n + 1)]
    best, bi, bj = 0, 0, 0
    for i in range(1, n + 1):
        ai = a[i - 1]
        row, prev = H[i], H[i - 1]
        for j in range(1, m + 1):
            s = match if ai == b[j - 1] else mismatch
            val = max(0, prev[j - 1] + s, prev[j] + gap, row[j - 1] + gap)
            row[j] = val
            if val > best:
                best, bi, bj = val, i, j
    # traceback to find span start
    i, j = bi, bj
    while i > 0 and j > 0 and H[i][j] > 0:
        diag = H[i - 1][j - 1]
        up = H[i - 1][j]
        left = H[i][j - 1]
        if H[i][j] == diag + (match if a[i - 1] == b[j - 1] else mismatch):
            i, j = i - 1, j - 1
        elif H[i][j] == up + gap:
            i -= 1
        else:
            j -= 1
    return best, i, bi, j, bj


def tier1_token(long_text: str, short_text: str,
                min_coverage: float = 0.6,
                min_identity: float = 0.7):
    """Locate short_text inside long_text by verbatim token overlap.
    Returns SpanResult or None if the match is too weak to call verbatim.

      min_coverage : matched region must cover this fraction of the SHORT text
                     (the snippet should be mostly accounted for).
      min_identity : within the matched region, this fraction of short-text
                     tokens must align exactly (guards paraphrase from
                     sneaking through as low-quality verbatim).
    """
    L = tokenize(long_text)
    S = tokenize(short_text)
    if not S:
        return None
    score, li0, li1, sj0, sj1 = _smith_waterman(L, S)
    matched_short = sj1 - sj0
    coverage = matched_short / len(S)
    # identity: exact-token overlap within the aligned short region
    aligned_long = L[li0:li1]
    aligned_short = S[sj0:sj1]
    exact = sum(1 for t in aligned_short if t in aligned_long)
    identity = exact / max(1, len(aligned_short))

    if coverage >= min_coverage and identity >= min_identity:
        return SpanResult(
            verdict="verbatim",
            score=round(min(coverage, identity), 3),
            long_span=" ".join(aligned_long),
            short_span=" ".join(aligned_short),
            detail=(f"verbatim overlap: coverage {coverage:.2f} of short text, "
                    f"identity {identity:.2f} within span. Shared TEXT — "
                    f"comparable for variant analysis."),
        )
    return None


# ----------------------------------------------------------------------
# TIER 2 — embedding-based sentence matching (paraphrase fallback)
# ----------------------------------------------------------------------

_model = None

def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def tier2_embedding(long_text: str, short_text: str,
                    min_similarity: float = 0.55):
    """Find the sentences in long_text most semantically similar to short_text.
    Succeeds (returns SpanResult) when content overlap is high but tier 1 failed
    -> paraphrase. Otherwise returns a no_overlap SpanResult."""
    from sentence_transformers import util
    try:
        model = _get_model()
    except Exception as e:
        return SpanResult(
            "model_unavailable", 0.0, "", "",
            f"tier-1 (verbatim) did not match, and the paraphrase model could "
            f"not be loaded ({type(e).__name__}). On first run this needs network "
            f"to download all-MiniLM-L6-v2 (~90MB); afterwards it works offline. "
            f"Verbatim result is unaffected.")
    long_sents = sentences(long_text)
    if not long_sents:
        return SpanResult("no_overlap", 0.0, "", "",
                          "long text has no sentences to match.")
    short_emb = model.encode(short_text, convert_to_tensor=True,
                             normalize_embeddings=True)
    long_emb = model.encode(long_sents, convert_to_tensor=True,
                            normalize_embeddings=True)
    sims = util.cos_sim(short_emb, long_emb)[0]
    best_sims = sims.tolist()
    # collect contiguous-ish best sentences above threshold
    keep = [(i, s) for i, s in enumerate(best_sims) if s >= min_similarity]
    top = max(best_sims)

    if keep:
        idxs = sorted(i for i, _ in keep)
        span = " ".join(long_sents[i] for i in idxs)
        return SpanResult(
            verdict="paraphrase",
            score=round(top, 3),
            long_span=span,
            short_span=short_text.strip(),
            detail=(f"paraphrase overlap: top sentence similarity {top:.2f}, "
                    f"{len(idxs)} sentence(s) above {min_similarity}. "
                    f"Shared CONTENT, not wording — homoplasy/shared-fact case. "
                    f"NOT handed to variant analysis."),
        )
    return SpanResult(
        verdict="no_overlap",
        score=round(top, 3),
        long_span="", short_span="",
        detail=(f"no overlap: best sentence similarity {top:.2f} < "
                f"{min_similarity}. Short text is not a snippet of the long one."),
    )


# ----------------------------------------------------------------------
# Diagnostic — full per-sentence similarity breakdown
# ----------------------------------------------------------------------

def similarity_breakdown(long_text: str, short_text: str):
    """Return every sentence of long_text paired with its cosine similarity
    to short_text, ranked highest first. Lets you SEE where a paraphrase
    score comes from, instead of only the single top number.

    Returns a list of (similarity, sentence) tuples."""
    from sentence_transformers import util
    model = _get_model()
    long_sents = sentences(long_text)
    if not long_sents:
        return []
    short_emb = model.encode(short_text, convert_to_tensor=True,
                             normalize_embeddings=True)
    long_emb = model.encode(long_sents, convert_to_tensor=True,
                            normalize_embeddings=True)
    sims = util.cos_sim(short_emb, long_emb)[0].tolist()
    ranked = sorted(zip(sims, long_sents), key=lambda x: x[0], reverse=True)
    return ranked


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

def localize(long_text: str, short_text: str) -> SpanResult:
    """Try verbatim first, fall back to paraphrase/embedding. The tier that
    fires is the provenance verdict."""
    t1 = tier1_token(long_text, short_text)
    if t1 is not None:
        return t1
    return tier2_embedding(long_text, short_text)


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------

if __name__ == "__main__":
    LONG = (
        "The committee convened at noon to review the quarterly figures. "
        "After lengthy debate the chair noted that the watchman stood upon "
        "the silver tower and saw the galleys approach the harbour at dawn. "
        "The minutes were approved and the meeting adjourned without further "
        "comment from the assembled members."
    )

    SHORT_VERBATIM = ("the watchman stood upon the silver tower and saw the "
                      "galleys approach the harbour at dawn")
    SHORT_PARAPHRASE = ("at daybreak the guard on the bright tower watched "
                        "the ships nearing the port")
    SHORT_UNRELATED = ("the inflation rate rose three points over the summer "
                       "before the central bank intervened")

    for label, short in [("VERBATIM", SHORT_VERBATIM),
                         ("PARAPHRASE", SHORT_PARAPHRASE),
                         ("UNRELATED", SHORT_UNRELATED)]:
        r = localize(LONG, short)
        print(f"=== {label} ===")
        print(f"verdict : {r.verdict}  (score {r.score}, comparable={r.comparable})")
        print(f"detail  : {r.detail}")
        if r.long_span:
            print(f"span    : {r.long_span[:90]}...")
        print()
