"""
citations.py — extract hard citation signals from a document.

Five signal types, each independently useful:

  1. FOOTNOTE / ENDNOTE markers   — [1], (1), ¹, fn. 3, note 12, etc.
     Presence means this text is academic and cites something.

  2. INLINE CITATIONS              — (Author, Year), (Author Year), Author (Year)
     These name who is being cited.

  3. BIBLIOGRAPHY ENTRIES          — lines that look like references:
     Author. Title. Publisher, Year.  or  Author (Year). Title. Journal…

  4. NAMED AUTHOR MENTIONS         — given a list of candidate author names,
     count how many times each appears and in what context (bare mention vs
     explicit cite phrase: "as X argues", "according to X", "X notes that").

  5. TITLE KEYWORDS                — given a source title, check whether its
     distinctive keywords appear verbatim in another document's text.

All extraction is regex-based (no ML), so it works offline and is fast.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field


# ── helpers ────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    """Normalise whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def _title_keywords(title: str, min_len: int = 6) -> list[str]:
    """Extract distinctive words from a title (skip short stop-words).

    min_len raised to 6 to avoid common words like 'letter', 'notes',
    'church', 'fathers', 'statement', 'counter' etc. that appear
    naturally in body text and create false positive title hits.
    """
    STOP = {
        "the","a","an","and","or","of","in","on","to","for","with",
        "from","by","at","is","was","are","were","this","that","its",
        "their","has","have","had","not","but","church","fathers","letter",
        "letters","notes","against","homily","sermon","commentary",
        "counter","statements","statement","fathers","father","saint",
        "according","because","therefore","however","although",
    }
    words = re.findall(r"[a-zA-Z]+", title.lower())
    return [w for w in words if len(w) >= min_len and w not in STOP]


# ── data structures ─────────────────────────────────────────────────────────

@dataclass
class FootnoteSignal:
    """A footnote or endnote marker found in the text."""
    marker: str        # e.g. "[1]", "¹", "fn. 3"
    context: str       # ~80 chars surrounding the marker


@dataclass
class InlineCitation:
    """An inline parenthetical citation."""
    raw: str           # e.g. "(Meyendorff, 1974)"
    author: str        # extracted author surname
    year: str          # extracted year ('' if absent)
    context: str


@dataclass
class BibEntry:
    """A line that looks like a bibliography/reference entry."""
    raw: str


@dataclass
class AuthorMention:
    """One occurrence of a named author in the text."""
    name: str
    context: str       # ~120 chars
    cite_phrase: bool  # True if accompanied by "argues", "notes", "writes", etc.


@dataclass
class TitleHit:
    """A title keyword found in the text."""
    keyword: str
    context: str


@dataclass
class CitationFingerprint:
    """All citation signals extracted from a single document."""
    url: str
    footnotes:       list[FootnoteSignal]  = field(default_factory=list)
    inline_citations: list[InlineCitation] = field(default_factory=list)
    bib_entries:     list[BibEntry]        = field(default_factory=list)
    author_mentions: list[AuthorMention]   = field(default_factory=list)
    title_hits:      list[TitleHit]        = field(default_factory=list)

    # ---- summary helpers ---------------------------------------------------

    @property
    def cited_authors(self) -> list[str]:
        """Unique authors mentioned in inline citations."""
        seen, out = set(), []
        for c in self.inline_citations:
            if c.author and c.author not in seen:
                seen.add(c.author)
                out.append(c.author)
        return out

    @property
    def cited_years(self) -> list[str]:
        return sorted({c.year for c in self.inline_citations if c.year})

    def cites_author(self, surname: str) -> bool:
        """Return True if this document explicitly cites the given surname."""
        s = surname.lower()
        for c in self.inline_citations:
            if s in c.author.lower():
                return True
        for m in self.author_mentions:
            if s in m.name.lower() and m.cite_phrase:
                return True
        return False

    def mentions_author(self, surname: str) -> int:
        """Count bare mentions of a surname (citations + plain mentions)."""
        s = surname.lower()
        count = sum(1 for c in self.inline_citations if s in c.author.lower())
        count += sum(1 for m in self.author_mentions if s in m.name.lower())
        return count

    def hits_title(self, title: str) -> int:
        """Count how many distinctive keywords from title appear in this doc."""
        kws = _title_keywords(title)
        return sum(1 for h in self.title_hits if h.keyword in kws)

    def report(self) -> str:
        lines = [f"Citation fingerprint for: {self.url}"]
        lines.append(f"  footnote markers  : {len(self.footnotes)}")
        lines.append(f"  inline citations  : {len(self.inline_citations)}")
        if self.cited_authors:
            lines.append(f"    authors cited   : {', '.join(self.cited_authors[:8])}")
        if self.cited_years:
            lines.append(f"    years cited     : {', '.join(self.cited_years[:8])}")
        lines.append(f"  bib entries       : {len(self.bib_entries)}")
        lines.append(f"  author mentions   : {len(self.author_mentions)}")
        lines.append(f"  title keyword hits: {len(self.title_hits)}")
        return "\n".join(lines)


# ── extraction functions ─────────────────────────────────────────────────────

# Footnote markers: [1], (1), fn. 3, note 3, ¹²³ superscripts
_FOOTNOTE_RE = re.compile(
    r"(\[\d{1,3}\]"            # [1]
    r"|\(\d{1,3}\)"            # (1)
    r"|(?:fn|footnote|note|endnote)\.?\s*\d{1,3}"  # fn. 3
    r"|[¹²³⁴⁵⁶⁷⁸⁹⁰]{1,3})",   # superscript unicode
    re.IGNORECASE
)

# Inline citations: (Author, 1999), (Author 1999), Author (1999)
_INLINE_RE = re.compile(
    r"(?:"
    r"\(([A-Z][a-záàäâéèëêíìïîóòöôúùüûñ\-']{2,}(?:\s+(?:and|&)\s+[A-Z][a-z\-']+)?)"
    r"(?:[,;]\s*|\s+)(\d{4}[a-z]?)\)"   # (Author, Year) or (Author Year)
    r"|([A-Z][a-záàäâéèëêíìïîóòöôúùüûñ\-']{2,})\s+\((\d{4}[a-z]?)\)"  # Author (Year)
    r")"
)

# Bibliography entry heuristic: line starting with capital, contains year,
# ends with page numbers or publisher info
_BIB_RE = re.compile(
    r"^[A-Z][A-Za-z\-',\.]{1,40}[,\.].*\b(1[0-9]{3}|20[0-2][0-9])\b.*[\.)]$"
)

# Cite-phrase verbs: signals that the author is being actively referenced
_CITE_VERBS = re.compile(
    r"\b(argues?|notes?|writes?|states?|claims?|observes?|suggests?|shows?|"
    r"demonstrates?|concludes?|explains?|contends?|maintains?|holds?|"
    r"points?\s+out|remarks?|emphasises?|emphasizes?|asserts?)\b",
    re.IGNORECASE
)


def extract_footnotes(text: str) -> list[FootnoteSignal]:
    out = []
    for m in _FOOTNOTE_RE.finditer(text):
        start = max(0, m.start() - 40)
        end   = min(len(text), m.end() + 40)
        out.append(FootnoteSignal(
            marker=m.group(0),
            context=_norm(text[start:end])
        ))
    return out


def extract_inline_citations(text: str) -> list[InlineCitation]:
    out = []
    for m in _INLINE_RE.finditer(text):
        if m.group(1):          # (Author, Year) form
            author, year = m.group(1), m.group(2)
        else:                   # Author (Year) form
            author, year = m.group(3), m.group(4)
        start = max(0, m.start() - 60)
        end   = min(len(text), m.end() + 60)
        out.append(InlineCitation(
            raw=m.group(0),
            author=author.strip(),
            year=year.strip(),
            context=_norm(text[start:end])
        ))
    return out


def extract_bib_entries(text: str) -> list[BibEntry]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if len(line) > 40 and _BIB_RE.match(line):
            out.append(BibEntry(raw=line))
    return out


def extract_author_mentions(text: str,
                             author_names: list[str]) -> list[AuthorMention]:
    """Scan text for occurrences of each surname in author_names."""
    out = []
    for name in author_names:
        if not name or len(name) < 3:
            continue
        pattern = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
        for m in pattern.finditer(text):
            start = max(0, m.start() - 60)
            end   = min(len(text), m.end() + 60)
            ctx   = _norm(text[start:end])
            cite  = bool(_CITE_VERBS.search(ctx))
            out.append(AuthorMention(name=name, context=ctx, cite_phrase=cite))
    return out


def extract_title_hits(text: str, titles: list[str]) -> list[TitleHit]:
    """Find occurrences of distinctive title keywords from other documents."""
    out = []
    text_lower = text.lower()
    seen = set()
    for title in titles:
        for kw in _title_keywords(title):
            if kw in seen:
                continue
            idx = text_lower.find(kw)
            if idx != -1:
                seen.add(kw)
                start = max(0, idx - 40)
                end   = min(len(text), idx + len(kw) + 40)
                out.append(TitleHit(
                    keyword=kw,
                    context=_norm(text[start:end])
                ))
    return out


def fingerprint(node: dict,
                other_nodes: list[dict]) -> CitationFingerprint:
    """Build a full CitationFingerprint for one node relative to others.

    other_nodes: the other documents in the corpus — used to check whether
                 their authors/titles appear in this document.
    """
    text = node.get("text", "")
    url  = node.get("url", "")

    # collect author surnames from other nodes' titles
    # (heuristic: capitalised words ≥4 chars from titles)
    # Crucially: exclude names that also appear in THIS node's own title,
    # because a shared author name (e.g. "Theodoret" in both texts) is not
    # a citation signal — it's just shared authorship.
    own_name_words = set(re.findall(r"\b[A-Z][a-z]{3,}\b", node.get("title", "")))
    GENERIC = {"Church","Fathers","Letter","Counter","Against",
               "Homily","Sermon","Commentary","Notes","From",
               "Saint","Blessed","According","Letters","Father"}

    other_authors: list[str] = []
    other_titles:  list[str] = []
    for nd in other_nodes:
        other_titles.append(nd.get("title", ""))
        for word in re.findall(r"\b[A-Z][a-z]{3,}\b", nd.get("title", "")):
            # skip if the word is generic, or appears in our own title
            # (shared author ≠ citation)
            if word not in GENERIC and word not in own_name_words:
                other_authors.append(word)

    fp = CitationFingerprint(url=url)
    fp.footnotes        = extract_footnotes(text)
    fp.inline_citations = extract_inline_citations(text)
    fp.bib_entries      = extract_bib_entries(text)
    fp.author_mentions  = extract_author_mentions(text, list(set(other_authors)))
    fp.title_hits       = extract_title_hits(text, other_titles)
    return fp


def citation_dependency_score(fp_src: CitationFingerprint,
                               fp_tgt: CitationFingerprint,
                               src_node: dict,
                               tgt_node: dict) -> tuple[float, list[str]]:
    """Score how strongly tgt cites/depends on src.

    Returns (score 0..1, list of human-readable evidence strings).

    Scoring:
      +0.5  tgt explicitly cites src's author in an inline citation
      +0.3  tgt mentions src's author with a cite-phrase verb
      +0.2  tgt contains ≥2 distinctive keywords from src's title
      +0.15 tgt has footnote markers (is a citing document at all)
      +0.1  tgt has bibliography entries
      +0.1  tgt's inline citations contain a year matching src's date
      cap at 1.0
    """
    score = 0.0
    evidence: list[str] = []

    src_title  = src_node.get("title", "")
    src_date   = src_node.get("date")
    src_year   = str(src_date.year) if src_date else ""

    # Author surnames from src title — same exclusion list as fingerprint()
    GENERIC = {"Church","Fathers","Letter","Counter","Against",
               "Homily","Sermon","Commentary","Notes","From",
               "Saint","Blessed","According","Letters","Father"}
    src_surnames = [w for w in re.findall(r"\b[A-Z][a-z]{3,}\b", src_title)
                    if w not in GENERIC]

    for surname in src_surnames:
        # +0.5 only for a real inline citation: (Author, Year) or Author (Year)
        if fp_tgt.cites_author(surname):
            score += 0.5
            evidence.append(f"inline citation of '{surname}'")
            break
        # +0.3 only for a mention accompanied by a cite-phrase verb
        # (filter: only count AuthorMentions where cite_phrase=True)
        cite_mentions = [m for m in fp_tgt.author_mentions
                         if surname.lower() in m.name.lower()
                         and m.cite_phrase]
        if cite_mentions:
            score += 0.3
            evidence.append(f"cite-phrase mention of '{surname}' "
                             f"({len(cite_mentions)}×)")
            break
        # bare mention (no cite verb) → weak signal, don't count as dependency

    kw_hits = fp_tgt.hits_title(src_title)
    if kw_hits >= 2:
        score += 0.2
        evidence.append(f"{kw_hits} title keywords from source found in target")
    elif kw_hits == 1:
        score += 0.05

    if len(fp_tgt.footnotes) >= 3:
        score += 0.15
        evidence.append(f"target has {len(fp_tgt.footnotes)} footnote markers")

    if len(fp_tgt.bib_entries) >= 1:
        score += 0.1
        evidence.append(f"target has {len(fp_tgt.bib_entries)} bibliography entries")

    if src_year and src_year in fp_tgt.cited_years:
        score += 0.1
        evidence.append(f"year {src_year} found in target's inline citations")

    return min(score, 1.0), evidence
