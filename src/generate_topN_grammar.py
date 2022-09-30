import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Get top N grammar rules")
parser.add_argument("--input", "-i", type=str, help="input file", required=True)
parser.add_argument(
    "--number", "-n", type=int, help="number of grammars", required=True
)
args = parser.parse_args()

input_file = args.input
n = args.number

df = pd.read_csv(input_file, usecols=[0], names=["grammar"], header=None)
value_counts = df["grammar"].value_counts()[:n]

grammar = list()
for i in range(n):
    print(f"{value_counts.index[i]}\t{value_counts.values[i]}")
    grammar.append(value_counts.index[i].split())

print(f"\nFor copy-pasting:\n\n{grammar}")
