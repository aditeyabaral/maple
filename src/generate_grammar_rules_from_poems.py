import string
import spacy
import argparse
from tqdm.auto import tqdm

nlp = spacy.load('en_core_web_sm')
parser = argparse.ArgumentParser(description='Get grammar from poems')
parser.add_argument('-i', '--input', help='Input file', required=True)
parser.add_argument('-o', '--output', help='Output file', required=True)
args = parser.parse_args()

input_file = args.input
output_file = args.output

def parseWords(row):
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
                grammar.append(token.pos_)
    return grammar

with open(input_file, encoding='utf8') as input_file, open(output_file, 'w', encoding='utf8') as output_file:
    lines = input_file.readlines()
    for line in tqdm(lines):
        out = ' '.join(parseWords(line))
        output_file.write(out)
        output_file.write('\n')
