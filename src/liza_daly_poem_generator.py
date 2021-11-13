import random
import string
import spacy
import json
import argparse
from tqdm.auto import tqdm

nlp = spacy.load('en_core_web_sm')
parser = argparse.ArgumentParser(description='Generate random blackout poems')
parser.add_argument('-i', '--input', type=str, help='Input file')
parser.add_argument('-o', '--output', type=str, help='Output file')
args = parser.parse_args()

input_file = args.input
output_file = args.output

def parse_words(poem):
    word_list = []
    for word in poem.split():
        word = word.translate(str.maketrans(
            {a: None for a in string.punctuation})).replace("\n", " ")
        word_info = {}
        word_info["text"] = word
        # print(word_info)
        word_list.append(word_info)

    sent = ' '.join([w["text"] for w in word_list])
    # print("sent", sent)
    doc = nlp(sent)
    for token in doc:
        # print("token:", token)
        for word in word_list:
            text = word['text']
            if token.text == text:
                word['token'] = token
                word['pos'] = token.pos_
    return word_list


def is_plural(word):
    if len(word['text']) == 0:
        return False
    # Special case this since one comes up a lot
    if word['text'] == 'men' or word['text'] == 'women':
        return True
    return word['text'][-1] == 's'


def is_plural_verb(word):
    if len(word['text']) == 0:
        return False
    if word['text'] == 'have':
        return True
    return word['text'][-1] != 's'


def is_present(word):
    if len(word['text']) == 0:
        return False
    return word['text'][-1] == 's'


def starts_with_vowel(word):
    vowels = set(['a', 'e', 'i', 'o', 'u'])
    if len(word['text']) == 0:
        return False
    return word['text'][0] in vowels


def generateBlackoutPoem(input_poem):
    words = parse_words(input_poem)

    text_vector = [i["text"] for i in words]
    poem_vector = list()  # [0]*len(text_vector)
    out_poem = dict()

    # print('parse output', words, '\n')
    # print(len(words))
    grammars = [
        ['DET', 'NOUN', 'VERB', 'NOUN'],
        ['ADJ', 'NOUN', 'VERB', 'NOUN'],
        ['VERB', 'DET', 'NOUN'],
        ['ADV', 'VERB', 'NOUN', 'CONJ', 'NOUN'],
        ['NOUN', 'NOUN', 'NOUN'],
        ['VERB', 'PROPN', 'DET', 'NOUN', 'NOUN'],
        ['NOUN', 'ADJ', 'NOUN', 'ADP', 'DET', 'NOUN'],
        ['VERB', 'DET', 'NOUN', 'ADP', 'DET', 'NOUN'],
        ['VERB', 'DET', 'NOUN', 'ADP', 'PROPN', 'NOUN'],
        ['PROPN', 'PROPN', 'DET', 'ADJ', 'NOUN', 'ADP', 'DET', 'NOUN'],
        ['ADJ', 'NOUN', 'DET', 'NOUN', 'ADP', 'DET', 'NOUN'],
        ['NOUN', 'NOUN', 'DET', 'ADJ', 'NOUN', 'ADP', 'DET', 'NOUN'],
        ['ADJ', 'NOUN', 'DET', 'NOUN', 'VERB', 'ADP', 'DET', 'NOUN'],
        ['PRON', 'VERB', 'AUX', 'VERB', 'SCONJ', 'DET', 'DET',
            'NOUN', 'VERB', 'ADP', 'DET', 'DET', 'NOUN'],
        ['PROPN', 'PROPN', 'DET', 'ADJ', 'NOUN', 'ADP', 'DET', 'NOUN', 'NOUN'],
        ['NOUN', 'NOUN', 'NOUN', 'VERB', 'ADP', 'DET', 'NOUN']
    ]
    grammar = random.choice(grammars)
    grammar_index = grammars.index(grammar)

    for g in range(len(grammars)):
        poem_vector = list()  # [0]*len(text_vector)
        grammar = grammars[g]
        grammar_index = g

        picks = []
        word_index = 0
        prev_word = None
        prev_pos = None
        length = len(words)
        n = 0
        for pos in grammar:
            while n < length:
                n += 1
                word = words[word_index]
                # print("word", word)
                # break
                if len(picks) > 0:
                    prev_word = picks[-1]
                    prev_pos = prev_word['pos']
                pick_this = True
                if prev_pos == 'DET':
                    if prev_word['text'] == 'a' or prev_word['text'] == 'an':
                        # Pick this if it's singular
                        pick_this = not is_plural(word)
                    if prev_word['text'] == 'a':
                        # Pick this if it doesn't start with a vowel
                        pick_this = not starts_with_vowel(word) and pick_this
                    if prev_word['text'] == 'an':
                        pick_this = starts_with_vowel(word) and pick_this
                    if prev_word['text'] == 'this':
                        pick_this = not is_plural(word) and pick_this
                    if prev_word['text'] == 'these':
                        pick_this = is_plural(word) and pick_this
                if prev_pos == 'NOUN':
                    # If the previous noun was plural, the verb must be plural
                    if is_plural(prev_word):
                        pick_this = is_plural_verb(word) and pick_this
                    if not is_plural(prev_word):
                        pick_this = not is_plural_verb(word) and pick_this
                if prev_pos == 'VERB':
                    # If the verb was plural, the noun must be
                    if is_plural_verb(prev_word):
                        pick_this = is_plural(word) and pick_this
                    if not is_plural_verb(prev_word):
                        pick_this = not is_plural(word) and pick_this
                if pos == 'VERB':
                    # Don't pick auxilliary verbs as they won't have a helper
                    if 'token' in word:
                        pick_this = word['token'].dep_ != 'aux' and pick_this

                if 'pos' in word and word['pos'] == pos and pick_this:
                    #print("Picking ", word['text'], " ", word['token'].dep_)
                    # poem_vector[word_index] = 1
                    poem_vector.append(word_index)
                    picks.append(word)
                    prev_pos = pos
                    word_index += 1
                    break

                word_index += 1

        final_text = [p['text'] for p in picks]

        if len(final_text) <= 2:
            grammar_index = -1
            final_text = ""
            poem_vector = []

        if len(final_text) > 2:
            # return grammar_index, final_text, poem_vector
            if grammar_index == -1:
                continue

            final_text = ' '.join(final_text).strip().lower()
            if grammar_index not in out_poem:
                out_poem[grammar_index] = {final_text: poem_vector}

    out_poem = {k: out_poem[k] for k in sorted(out_poem)}

    print("Output Generated")

    return out_poem

poems = list()
with open(input_file) as passage_file, open(output_file, 'w') as output_file:
    passages = passage_file.readlines()
    for passage in tqdm(passages):
        poem = generateBlackoutPoem(passage)
        poems.append(poem)
    json.dump(poems, output_file)