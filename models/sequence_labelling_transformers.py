import re
import json
import string
import nltk
import argparse
import numpy as np
import pandas as pd
from simpletransformers.ner import NERModel

np.random.seed(0)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def cleanText(text):
    text = text.lower().strip()
    text = re.sub("\n+", " ", text)
    text = re.sub(" +", " ", text)
    text = re.sub("\t+", " ", text)
    return text


def createDataFrame(input_file, clean=True):
    with open(input_file) as d:
        dfd_json = json.load(d)

    poems = list()
    haikus = list()
    indices = list()
    for dataset in dfd_json:
        for poem in dataset:
            for grammar_index in poem:
                if grammar_index == "poem":
                    continue
                else:
                    haiku_data = poem[grammar_index]
                    for haiku, index in list(haiku_data.items()):
                        if len(index) >= poem_length and len(poem["poem"].split()) <= text_length:
                            poems.append(poem["poem"])
                            haikus.append(haiku)
                            indices.append(index)

    cleaned_poems = poems
    if clean:
        cleaned_poems = list(map(cleanText, poems))

    df = pd.DataFrame()
    df["poem"] = cleaned_poems
    df["cleaned_poem"] = poems
    df["haiku"] = haikus
    df["indices"] = indices
    df = df.drop_duplicates(subset=["poem"])
    df = df.reset_index(drop=True)
    return df


def removeBadSymbols(s):
    s = s.translate(str.maketrans('', '', string.punctuation)).strip().lower()
    return s


def createWordTagDataFrame(poems, tags):
    poem_no = []
    word = []
    tag = []
    for i in range(len(poems)):
        poem = poems[i].split()
        count_poem = [i+1] * len(poem)
        poem_no.extend(count_poem)
        word.extend(poem)
        count_tag = ['0'] * len(poem)
        for j in tags[i]:
            try:
                count_tag[j] = '1'
            except:
                print(f"{poem}\n{tags[i]}\n\n")
        tag.extend(count_tag)
    word = list(map(removeBadSymbols, word))
    dataset = {"sentence_id": poem_no, "words": word, "labels": tag}
    df = pd.DataFrame(dataset)
    df = df[df["words"] != ""]
    return df


def displayPredictions(predictions, fname, display=False):
    data = list()
    for p in predictions:
        poem = []
        haiku = []
        for d in p:
            for word, tag in d.items():
                poem.append(word)
                if tag == '1':
                    haiku.append(word)
        if len(haiku) >= 4:
            poem = " ".join(poem)
            haiku = " ".join(haiku)
            if display:
              print(f"Poem:  {poem}")
              print(f"Haiku:  {haiku}\n")
            data.append({"passage": poem, "poem": haiku})

    with open(fname, "w") as f:
        json.dump(data, f)


parser = argparse.ArgumentParser("Train MAPLE using Sequence Labelling")
parser.add_argument("--input", "-i", type=str,
                    help="Path to the training data", required=True)
parser.add_argument("--output", "-o", type=str,
                    help="Path to store generated results", required=True)
parser.add_argument("--model_class", "-mc", type=str,
                    default="bert", help="Transformer base model class")
parser.add_argument("--model_type", "-mt", type=str,
                    default="bert-base-cased", help="Transformer model type")
parser.add_argument("--poem_length", "-pl", type=str,
                    default=5, help="Minimum poem length to train on")
parser.add_argument("--text_length", "-tl", type=str,
                    default=120, help="Maximum passage length to train on")
parser.add_argument("--split", "-s", type=bool, default=True,
                    help="Whether to split the dataset into train/test")
args = parser.parse_args()

input_file = args.input
output_file = args.output
model_type = args.model_type
model_class = args.model_class
poem_length = int(args.poem_length)
text_length = int(args.text_length)
split = args.split

df = createDataFrame(input_file, clean=True)
all_poems = df["poem"].values
all_indices = df["indices"].values
df = createWordTagDataFrame(all_poems, all_indices)

if split:
    mask = np.random.rand(len(df)) < 0.8
    train_df = df[mask]
    test_df = df[~mask]
    train_poems = train_df["poem"].values
    test_poems = test_df["poem"].values
    train_indices = train_df["indices"].values
    test_indices = test_df["indices"].values
    train_df = createWordTagDataFrame(train_poems, train_indices)
    test_df = createWordTagDataFrame(test_poems, test_indices)

    model = NERModel(model_class, model_type, args={
                     "overwrite_output_dir": True, "reprocess_input_data": True}, labels=["0", "1"])
    model.train_model(train_df)

    predictions, raw_outputs = model.predict(test_poems)
    displayPredictions(predictions, f"{output_file}_test.json")

    predictions, raw_outputs = model.predict(train_poems)
    displayPredictions(predictions, f"{output_file}_train.json")

    exit(0)

model = NERModel(model_class, model_type, args={
                 "overwrite_output_dir": True, "reprocess_input_data": True}, labels=["0", "1"])
model.train_model(df)
predictions, raw_outputs = model.predict(all_poems)
displayPredictions(predictions, f"{output_file}_train.json")
