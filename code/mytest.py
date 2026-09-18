from Pairwise.pairwise import compare, admit_to_tree

# reference stays inline; sources are read from files
reference = "paste your archetype / seed text here"

with open("source_a.txt", encoding="utf-8") as f:
    source_a = f.read()

with open("source_b.txt", encoding="utf-8") as f:
    source_b = f.read()

result = compare(reference, source_a, source_b)
print(result.summary())
print(admit_to_tree(result)[1])