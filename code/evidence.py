"""
evidence.py — show the ACTUAL shared material between two texts, so a claim
that one depends on the other is backed by visible evidence, not just a score.

Two kinds of evidence, both filtered for DISTINCTIVENESS:

  1. SHARED PHRASES (verbatim): word sequences both texts contain.
     Filtered so that matches made only of common/function words
     ("the", "because", "and"...) are discarded — those are everywhere and
     prove nothing. Only distinctive shared sequences are reported.

  2. PARAPHRASE PAIRS (semantic): the sentence in text A and the sentence in
     text B that are closest in meaning, shown SIDE BY SIDE, so content
     dependence is visible even when the wording differs.

The point: when the tool claims dependence, it must point at the words that
justify the claim.
"""

from __future__ import annotations
from dataclasses import dataclass
from difflib import SequenceMatcher
import re

from localize import tokenize, sentences, _get_model

# ----------------------------------------------------------------------
# A small stoplist of words too common to count as evidence on their own.
# A shared phrase made ONLY of these is noise, not dependence.
# ----------------------------------------------------------------------
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "as", "of",
    "to", "in", "on", "at", "by", "for", "with", "from", "is", "are", "was",
    "were", "be", "been", "being", "it", "its", "he", "she", "they", "this",
    "that", "these", "those", "his", "her", "their", "which", "who", "whom",
    "not", "no", "nor", "also", "because", "therefore", "thus", "hence",
    "we", "i", "you", "him", "them", "there", "here", "what", "when", "where",
    "would", "could", "should", "can", "will", "may", "might", "do", "does",
    "did", "have", "has", "had", "one", "all", "any", "both", "each",
}


@dataclass
class SharedPhrase:
    text: str            # the shared word sequence
    n_tokens: int        # length in tokens
    n_content: int       # how many tokens are NOT stopwords (distinctiveness)

    def __str__(self):
        return f'[{self.n_content} content / {self.n_tokens} words] "{self.text}"'


@dataclass
class ParaphrasePair:
    similarity: float
    sentence_a: str
    sentence_b: str


@dataclass
class Evidence:
    shared_phrases: list[SharedPhrase]
    paraphrase_pairs: list[ParaphrasePair]

    def report(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("DEPENDENCE EVIDENCE")
        lines.append("=" * 60)

        lines.append("\n1. SHARED VERBATIM PHRASES (distinctive only)")
        if self.shared_phrases:
            for p in self.shared_phrases:
                lines.append("   " + str(p))
        else:
            lines.append("   none — no distinctive shared wording. "
                         "(Any overlap was only common words.)")

        lines.append("\n2. CLOSEST PARAPHRASE PAIRINGS")
        if self.paraphrase_pairs:
            for pp in self.paraphrase_pairs:
                lines.append(f"   similarity {pp.similarity:.3f}")
                lines.append(f"     A: {pp.sentence_a[:120]}")
                lines.append(f"     B: {pp.sentence_b[:120]}")
        else:
            lines.append("   none above threshold.")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# 1. Shared verbatim phrases, filtered for distinctiveness
# ----------------------------------------------------------------------

def shared_phrases(text_a: str, text_b: str,
                   min_tokens: int = 3,
                   min_content_tokens: int = 2) -> list[SharedPhrase]:
    """Find word sequences common to both texts, keeping only those with
    enough NON-stopword tokens to be distinctive.

      min_tokens         : ignore matched runs shorter than this.
      min_content_tokens : of those tokens, at least this many must be
                           content words (not in STOPWORDS). A 5-word match
                           that is 5 stopwords is discarded; a 3-word match
                           with 2 content words is kept.
    """
    a = tokenize(text_a)
    b = tokenize(text_b)
    sm = SequenceMatcher(None, a, b, autojunk=False)
    found = []
    for block in sm.get_matching_blocks():
        if block.size < min_tokens:
            continue
        seq = a[block.a: block.a + block.size]
        content = [t for t in seq if t not in STOPWORDS and t.isalnum()]
        if len(content) >= min_content_tokens:
            found.append(SharedPhrase(
                text=" ".join(seq),
                n_tokens=len(seq),
                n_content=len(content),
            ))
    # most distinctive (most content words) first
    found.sort(key=lambda p: p.n_content, reverse=True)
    return found


# ----------------------------------------------------------------------
# 2. Closest paraphrase pairings between sentences
# ----------------------------------------------------------------------

def paraphrase_pairs(text_a: str, text_b: str,
                     min_similarity: float = 0.55,
                     top_k: int = 3) -> list[ParaphrasePair]:
    """For each sentence in A, find its closest sentence in B; keep the
    strongest pairings above threshold. Shows content dependence even when
    wording differs."""
    from sentence_transformers import util
    model = _get_model()
    sa = sentences(text_a)
    sb = sentences(text_b)
    if not sa or not sb:
        return []
    ea = model.encode(sa, convert_to_tensor=True, normalize_embeddings=True)
    eb = model.encode(sb, convert_to_tensor=True, normalize_embeddings=True)
    sim = util.cos_sim(ea, eb)  # rows = A sentences, cols = B sentences

    pairs = []
    for i in range(len(sa)):
        row = sim[i].tolist()
        j = max(range(len(sb)), key=lambda k: row[k])
        if row[j] >= min_similarity:
            pairs.append(ParaphrasePair(round(row[j], 3), sa[i], sb[j]))
    pairs.sort(key=lambda p: p.similarity, reverse=True)
    return pairs[:top_k]


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

def extract_evidence(text_a: str, text_b: str) -> Evidence:
    return Evidence(
        shared_phrases=shared_phrases(text_a, text_b),
        paraphrase_pairs=paraphrase_pairs(text_a, text_b),
    )


# ----------------------------------------------------------------------
if __name__ == "__main__":
    A = ("The lighthouse keeper climbed the spiral stair each evening and lit "
         "the great lamp before the fog rolled in from the sea. He kept a "
         "careful log of every passing ship.")
    B = ("Local accounts describe how, every night, the keeper of the tower "
         "would ascend the winding steps and kindle the beacon as the mist "
         "crept inland. The lighthouse keeper climbed the spiral stair each "
         "evening, the chronicle notes, and recorded the vessels he saw.")

    ev = extract_evidence(A, B)
    print(ev.report())
