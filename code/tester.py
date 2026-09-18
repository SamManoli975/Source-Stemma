from evidence import extract_evidence

a = open("polycarpwiki.txt", encoding="utf-8").read()
b = open("polycarpwork.txt", encoding="utf-8").read()

print(extract_evidence(a, b).report())