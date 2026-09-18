from Pairwise.text.old1localize import localize

pairs = [
    ("VERBATIM expected", "polycarpwiki.txt", "polycarpwork.txt")
]

for label, lf, sf in pairs:
    long_text = open(lf, encoding="utf-8").read()
    short_text = open(sf, encoding="utf-8").read()
    r = localize(long_text, short_text)
    print(f"=== {label} ===")
    print(f"got: {r.verdict} (score {r.score})")
    print(f"     {r.detail}\n")