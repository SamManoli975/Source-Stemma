"""
pairwise.py — pairwise comparison of two sources for stemmatic thresholding.

The point of this module is NOT to give you one similarity number. A single
number (edit distance, % overlap, cosine) is the trap: it conflates two things
that stemmatics must keep apart.

    1. Shared CORRECT readings   -> agreement with a reference/archetype.
                                    Carries NO descent signal. Two independent
                                    copies of the same true text agree here.
    2. Shared DEVIANT readings   -> both sources depart from the reference in
                                    the SAME way. This is the only agreement
                                    that can indicate common descent
                                    (the candidate "conjunctive errors").

A threshold for "add this pair to the tree" must key on (2), never on
total similarity. This module measures both and reports them separately,
plus the alignment so you can inspect the actual variants.

Requires a REFERENCE text (your synthetic seed / archetype, or a chosen
base witness). Without a reference you cannot tell a shared error from a
shared truth — so it is a required argument, by design.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import re


# ----------------------------------------------------------------------
# Tokenisation
# ----------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Word-level tokens, lowercased, punctuation stripped to its own tokens.
    Word-level is the right grain for textual descent; character-level drowns
    real variants in spelling noise."""
    return re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)


# ----------------------------------------------------------------------
# Three-way alignment against a reference
# ----------------------------------------------------------------------

@dataclass
class VariantSite:
    """One position where A and/or B differ from the reference."""
    ref: tuple[str, ...]       # reference reading at this site ('' = omission)
    a: tuple[str, ...]         # source A reading
    b: tuple[str, ...]         # source B reading

    @property
    def a_deviates(self) -> bool:
        return self.a != self.ref

    @property
    def b_deviates(self) -> bool:
        return self.b != self.ref

    @property
    def shared_deviation(self) -> bool:
        """Both depart from the reference AND agree with each other.
        This is a candidate conjunctive error: the descent-bearing event."""
        return self.a_deviates and self.b_deviates and self.a == self.b

    @property
    def independent_deviation(self) -> bool:
        """Both depart, but differently. Homoplasy / independent variation.
        Looks like 'difference' but carries the OPPOSITE signal to (shared)."""
        return self.a_deviates and self.b_deviates and self.a != self.b


@dataclass
class PairResult:
    sites: list[VariantSite] = field(default_factory=list)
    ref_len: int = 0

    # ---- raw difference (the misleading number) ----
    @property
    def total_variant_sites(self) -> int:
        return len(self.sites)

    # ---- the numbers a threshold should actually use ----
    @property
    def shared_deviations(self) -> list[VariantSite]:
        return [s for s in self.sites if s.shared_deviation]

    @property
    def independent_deviations(self) -> list[VariantSite]:
        return [s for s in self.sites if s.independent_deviation]

    @property
    def n_shared(self) -> int:
        return len(self.shared_deviations)

    @property
    def n_independent(self) -> int:
        return len(self.independent_deviations)

    @property
    def signal_ratio(self) -> float:
        """Shared deviations as a fraction of all joint deviations.
        ~1.0 -> when both differ from the archetype they differ TOGETHER
                (strong descent candidate).
        ~0.0 -> when both differ they differ APART (homoplasy / no descent).
        Undefined (returns 0.0) if neither ever co-deviates."""
        denom = self.n_shared + self.n_independent
        return self.n_shared / denom if denom else 0.0

    def summary(self) -> dict:
        return {
            "ref_len": self.ref_len,
            "total_variant_sites": self.total_variant_sites,
            "shared_deviations": self.n_shared,
            "independent_deviations": self.n_independent,
            "signal_ratio": round(self.signal_ratio, 3),
        }


def _block_align(ref_t, a_t, b_t):
    """Align A and B each to the reference, then walk the reference positions
    so the two alignments share a coordinate frame. Yields (ref, a, b) chunks."""
    sm_a = SequenceMatcher(None, ref_t, a_t, autojunk=False)
    sm_b = SequenceMatcher(None, ref_t, b_t, autojunk=False)

    # Map each reference index -> (a_reading, b_reading) for that ref span.
    def ref_map(sm, other):
        m = {}  # ref_index -> list of other-tokens aligned to it
        last_ref = 0
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    m[i1 + k] = (other[j1 + k],)
            elif tag == "replace":
                # attach the whole replacement to the first ref index of the span
                m[i1] = tuple(other[j1:j2])
                for k in range(1, i2 - i1):
                    m[i1 + k] = ()  # consumed by the replacement above
            elif tag == "delete":
                m[i1] = ()  # ref tokens absent in other (omission)
                for k in range(1, i2 - i1):
                    m[i1 + k] = ()
            elif tag == "insert":
                # other has extra tokens not in ref; hang them on prior index
                idx = max(i1 - 1, 0)
                m.setdefault(idx, ())
                m[idx] = m[idx] + tuple(other[j1:j2])
        return m

    map_a = ref_map(sm_a, a_t)
    map_b = ref_map(sm_b, b_t)

    sites = []
    for i, r in enumerate(ref_t):
        a_read = map_a.get(i, (r,))
        b_read = map_b.get(i, (r,))
        site = VariantSite(ref=(r,), a=tuple(a_read), b=tuple(b_read))
        if site.a_deviates or site.b_deviates:
            sites.append(site)
    return sites


def compare(reference: str, source_a: str, source_b: str) -> PairResult:
    """Compare two sources against a reference and return the variant analysis."""
    ref_t = tokenize(reference)
    a_t = tokenize(source_a)
    b_t = tokenize(source_b)
    sites = _block_align(ref_t, a_t, b_t)
    return PairResult(sites=sites, ref_len=len(ref_t))


# ----------------------------------------------------------------------
# Thresholding helper
# ----------------------------------------------------------------------

def admit_to_tree(result: PairResult,
                  min_shared: int = 2,
                  min_signal_ratio: float = 0.5) -> tuple[bool, str]:
    """Decide whether a pair carries enough descent signal to be a tree edge
    candidate. Returns (admit?, human-readable reason).

    Defaults are deliberately conservative and meant to be tuned on your
    synthetic corpus in Phase 1 — that calibration IS the experiment.

      min_shared        : need at least this many candidate conjunctive errors.
                          One shared deviation is too easily coincidence.
      min_signal_ratio  : of the sites where both sources deviate, this fraction
                          must be SHARED rather than independent. Guards against
                          the homoplasy case where two texts differ from the
                          archetype a lot but rarely in the same way.
    """
    if result.n_shared < min_shared:
        return False, (f"only {result.n_shared} shared deviation(s); "
                       f"need >= {min_shared}. No defensible conjunctive error.")
    if result.signal_ratio < min_signal_ratio:
        return False, (f"signal_ratio {result.signal_ratio:.2f} < {min_signal_ratio}: "
                       f"co-deviations are mostly independent (homoplasy), "
                       f"not shared. Looks like convergence, not descent.")
    return True, (f"{result.n_shared} shared deviation(s), "
                  f"signal_ratio {result.signal_ratio:.2f}. Admit as edge candidate.")


# ----------------------------------------------------------------------
# Demo / self-test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    REF = ("the watchman stood upon the high tower and saw the ships "
           "approach the harbour at the break of dawn")

    # A and B share the SAME error: 'silver tower' (descent signal),
    # plus 'galleys' for 'ships'. A also has its own slip ('morning').
    A = ("the watchman stood upon the silver tower and saw the galleys "
         "approach the harbour at the break of morning")
    B = ("the watchman stood upon the silver tower and saw the galleys "
         "approach the port at the break of dawn")

    # C diverges a LOT from the reference but never in the same way as A.
    C = ("a guard sat below the broken wall and watched the boats "
         "drift toward the bay near the end of night")

    print("=== A vs B (expect: real shared signal) ===")
    r1 = compare(REF, A, B)
    print(r1.summary())
    print(admit_to_tree(r1)[1])
    for s in r1.shared_deviations:
        print("   shared:", " ".join(s.ref), "->", " ".join(s.a))

    print("\n=== A vs C (expect: high difference, low signal) ===")
    r2 = compare(REF, A, C)
    print(r2.summary())
    print(admit_to_tree(r2)[1])
