#!/usr/bin/python3

import pandas as pd
import sys

def main(argv):

    try:
        inputfile = argv[1]
        n = int(argv[2])

    except:
        print("Usage : python getTopNGrammars.py ../data/grammar/grammars.csv N")
        sys.exit(1)

    df = pd.read_csv(inputfile, usecols=[0], names=['grammar'], header=None)
    # print(df)

    out = df['grammar'].value_counts()[:n].index.tolist()
    for i in out:
      print(i.split())


if __name__ == "__main__":
    main(sys.argv)
