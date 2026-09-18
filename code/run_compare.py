"""
run_compare.py — compare two source files against a reference file.

Usage:
    python3 run_compare.py reference.txt source_a.txt source_b.txt

If you run it with no arguments it looks for these three files in the
current folder:
    reference.txt   the archetype / seed you measure deviation against
    source_a.txt    first source
    source_b.txt    second source
"""

import sys
from Pairwise.pairwise import compare, admit_to_tree


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def main():
    args = sys.argv[1:]
    if len(args) == 3:
        ref_path, a_path, b_path = args
    elif len(args) == 0:
        ref_path, a_path, b_path = "reference.txt", "source_a.txt", "source_b.txt"
    else:
        print("usage: python3 run_compare.py reference.txt source_a.txt source_b.txt")
        sys.exit(1)

    try:
        reference = read(ref_path)
        source_a = read(a_path)
        source_b = read(b_path)
    except FileNotFoundError as e:
        print(f"Could not find a file: {e.filename}")
        print("Make sure the three .txt files exist in this folder, "
              "or pass their paths as arguments.")
        sys.exit(1)

    result = compare(reference, source_a, source_b)

    print(f"reference: {ref_path}")
    print(f"source A : {a_path}")
    print(f"source B : {b_path}\n")
    print(result.summary())
    print(admit_to_tree(result)[1])

    if result.shared_deviations:
        print("\nshared deviations (candidate conjunctive errors):")
        for s in result.shared_deviations:
            print("  ", " ".join(s.ref), "->", " ".join(s.a))
    else:
        print("\nno shared deviations found.")


if __name__ == "__main__":
    main()
