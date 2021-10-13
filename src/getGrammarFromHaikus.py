#!/usr/bin/python3
import os
import sys
import random
from statistics import mean
import string
import uuid
import sys
import tracery
import spacy
import csv

nlp = spacy.load('en')

def parse_words(row):
    poem = ' '.join(row)
    word_list = []
    for word in poem.split():
        word = word.translate(str.maketrans({a:None for a in string.punctuation})).replace("\n", " ")
        word_info = {}
        word_info["text"] = word
        word_list.append(word_info)

    sent = ' '.join([w["text"] for w in word_list])
    doc = nlp(sent)
    grammar = []
    for token in doc:
        for word in word_list:
            text = word['text']
            if token.text == text:
                # word['token'] = token
                # word['pos'] = token.pos_
                grammar.append(token.pos_)
    return grammar

def main(argv):
    try:
        inputfile = argv[1]
        outputfile = argv[2]

    except:
        print("Usage : python3 getGrammarFromHaikus.py ../data/grammar/haiku.csv ../data/grammar/grammars.csv")
        sys.exit(1)


    file = open(outputfile,'w')

    with open(inputfile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            out = ' '.join(parse_words(row))
            file.write(out)
            file.write('\n')
    file.close() 

if __name__ == "__main__":
    main(sys.argv)
