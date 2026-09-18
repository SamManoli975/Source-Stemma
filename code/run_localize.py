from Pairwise.text.old1localize import localize

with open("source_a.txt", encoding="utf-8") as f:
    long_text = f.read()

with open("source_b.txt", encoding="utf-8") as f:
    short_text = f.read()

r = localize(long_text, short_text)
print("verdict :", r.verdict, " comparable:", r.comparable)
print("detail  :", r.detail)
if r.long_span:
    print("span    :", r.long_span)
