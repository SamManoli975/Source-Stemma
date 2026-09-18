"""
relate.py — given two sources, determine their relationship.

The task: source A is a quotation (a passage from a work). Source B is a
commentary (it quotes that passage AND discusses it). We want the tool to,
on its own:

  STAGE 1  find the shared passage           -> uses localize.py
           (verbatim OR paraphrased; the tier that fires is recorded)

  STAGE 2  decide which text is the COMMENTARY and which the QUOTATION
           -> role classification, by structural asymmetry:
              a commentary = the shared span + a lot of OTHER text (discussion)
              a quotation  = mostly just the span, little else

  STAGE 3  infer PRIORITY (which came first)
           -> IMPORTANT: this is NOT read from the wording. A commentary is
              logically posterior to the work it comments on — that is a fact
              about the GENRE, not a textual measurement. So priority is
              inferred from role ("the commentary depends on the quoted work,
              therefore the quoted work is prior"), OR overridden by explicit
              dates if you supply them. The tool labels which basis it used.

This is a DEPENDENCY relation (B draws on A), not a stemmatic DESCENT relation
(one manuscript copied from another). The tool says so, so the claim is not
overstated.
"""

from __future__ import annotations
from dataclasses import dataclass
from Pairwise.text.old1localize import localize, tokenize, SpanResult
from datetime import date


@dataclass
class RelationResult:
    overlap: SpanResult            # stage 1: the shared-passage finding
    role_a: str                    # "commentary" | "quotation" | "unclear"
    role_b: str
    coverage_a: float              # fraction of A that is the shared span
    coverage_b: float              # fraction of B that is the shared span
    prior: str                     # "A" | "B" | "unknown"
    priority_basis: str            # "date" | "role-inference" | "none"
    detail: str

    def report(self) -> str:
        lines = [
            f"shared passage : {self.overlap.verdict} (score {self.overlap.score})",
            f"  A is span-fraction {self.coverage_a:.2f}  -> role: {self.role_a}",
            f"  B is span-fraction {self.coverage_b:.2f}  -> role: {self.role_b}",
            f"priority       : {self._prior_text()}  [basis: {self.priority_basis}]",
            f"note           : {self.detail}",
        ]
        return "\n".join(lines)

    def _prior_text(self):
        if self.prior == "unknown":
            return "could not determine which came first"
        return f"source {self.prior} came first"


def _coverage(full_text: str, span: str) -> float:
    """What fraction of full_text's tokens are accounted for by the matched span."""
    full = tokenize(full_text)
    sp = tokenize(span)
    if not full:
        return 0.0
    return min(1.0, len(sp) / len(full))


def relate(source_a: str, source_b: str,
           date_a: date | None = None,
           date_b: date | None = None,
           commentary_threshold: float = 0.6) -> RelationResult:
    """Determine the relationship between two sources.

    date_a / date_b : optional. If BOTH given, priority comes from dates and
                      overrides role inference (dates are hard evidence; role
                      is a structural guess).
    commentary_threshold : if the shared span covers MORE than this fraction of
                      a text, that text is mostly-quote -> quotation. If it
                      covers LESS, the text has substantial other material ->
                      commentary.
    """
    # ---- STAGE 1: find the shared passage --------------------------------
    # localize expects (long, short). The commentary is usually longer, but we
    # don't know roles yet, so feed longer-as-long and keep the span either way.
    if len(source_a) >= len(source_b):
        overlap = localize(source_a, source_b)
    else:
        overlap = localize(source_b, source_a)

    if overlap.verdict in ("no_overlap", "model_unavailable"):
        return RelationResult(
            overlap=overlap, role_a="unclear", role_b="unclear",
            coverage_a=0.0, coverage_b=0.0, prior="unknown",
            priority_basis="none",
            detail=("no shared passage established, so role and priority cannot "
                    "be determined. " + overlap.detail),
        )

    # ---- STAGE 2: role classification by coverage ------------------------
    # How much of each source is the shared passage?
    cov_a = _coverage(source_a, overlap.long_span if len(source_a) >= len(source_b)
                      else overlap.short_span)
    cov_b = _coverage(source_b, overlap.short_span if len(source_a) >= len(source_b)
                      else overlap.long_span)

    def role(cov):
        if cov >= commentary_threshold:
            return "quotation"   # mostly the passage itself
        return "commentary"     # passage embedded in much other text

    role_a = role(cov_a)
    role_b = role(cov_b)

    # guard the ambiguous case: both look like quotations (two bare quotes)
    if role_a == role_b == "quotation":
        detail_role = ("both texts are mostly the shared passage — these look "
                       "like two quotations of the same work, not a "
                       "commentary/quotation pair.")
    elif role_a == role_b == "commentary":
        detail_role = ("neither text is mostly the shared passage — both embed "
                       "it in substantial discussion. Two commentaries?")
    else:
        commentary = "A" if role_a == "commentary" else "B"
        quotation = "B" if commentary == "A" else "A"
        detail_role = (f"source {commentary} embeds the passage in discussion "
                       f"(commentary); source {quotation} is mostly the passage "
                       f"(quotation).")

    # ---- STAGE 3: priority -----------------------------------------------
    prior, basis = "unknown", "none"

    if date_a is not None and date_b is not None:
        basis = "date"
        if date_a < date_b:
            prior = "A"
        elif date_b < date_a:
            prior = "B"
        else:
            prior, basis = "unknown", "date"  # same date, can't order
    else:
        # infer from role: the commentary depends on the quoted work,
        # so the quotation is prior. Only valid in the clean one-of-each case.
        if {role_a, role_b} == {"commentary", "quotation"}:
            basis = "role-inference"
            prior = "A" if role_a == "quotation" else "B"

    detail = detail_role
    if basis == "role-inference":
        detail += (" Priority inferred from role: a commentary is logically "
                   "posterior to the work it discusses. This is a DEPENDENCY "
                   "claim (the commentary draws on the quotation), not a proven "
                   "descent. Supply dates to confirm.")
    elif basis == "date":
        detail += " Priority is from the dates you supplied (hard evidence)."
    else:
        detail += (" Priority undetermined: roles are not a clean "
                   "commentary/quotation pair, and no dates were given.")

    return RelationResult(
        overlap=overlap, role_a=role_a, role_b=role_b,
        coverage_a=round(cov_a, 3), coverage_b=round(cov_b, 3),
        prior=prior, priority_basis=basis, detail=detail,
    )


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # A: a bare quotation of the passage.
    A = ("But the Holy Spirit is said to be from the Father and is testified "
         "to be the Son's. For it says: If any man have not the Spirit of "
         "Christ, he is none of His. Hence the Spirit that is from God is also "
         "Christ's Spirit.")

    # B: a commentary that quotes (a paraphrase of) the passage and discusses it.
    B = ("In this homily Gregory turns to the relation of the persons. He "
         "writes that the Holy Spirit is said to be from the Father and is "
         "testified to be the Son's, citing Romans: if any man have not the "
         "Spirit of Christ he is none of His. The crucial question for later "
         "readers is whether the preposition ek stood in the original, since "
         "the Greeks alleged a Latin interpolation, and the whole Filioque "
         "debate at Florence turned on exactly this clause. We will argue the "
         "shorter reading is prior and the ek a later addition.")

    print("=== no dates (priority from role) ===")
    print(relate(A, B).report())

    print("\n=== with dates ===")
    print(relate(A, B, date_a=date(380, 1, 1), date_b=date(2026, 1, 1)).report())
