from relate import relate
from datetime import date

with open("source_a.txt", encoding="utf-8") as f:
    a = f.read()
with open("source_b.txt", encoding="utf-8") as f:
    b = f.read()

# add dates if you have them, e.g. date_a=date(380,1,1), date_b=date(1840,1,1)
result = relate(a, b)
print(result.report())
