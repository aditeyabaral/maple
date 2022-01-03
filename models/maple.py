import os
import nltk
import argparse
import platform
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import GPT2TokenizerFast, RobertaTokenizerFast, XLMRobertaTokenizerFast

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.tokenize import word_tokenize


class MapleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = list()
    for i, label in enumerate(labels):
        word_ids = encodings.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = list()
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # previous_word_idx = word_idx
    encoded_labels.append(label_ids)
    return encoded_labels


parser = argparse.ArgumentParser(description='Train a MAPLE Transformer Model')
parser.add_argument('--model', '-m', type=str,
                    help='Transformer model name/path to siamese pre-train', required=True)
parser.add_argument('--dataset', '-d', type=str,
                    help='Path to dataset in required format', required=True)
parser.add_argument('--hub', '-hf', type=bool,
                    help='Push model to HuggingFace Hub', required=False, default=False)
parser.add_argument('--batch_size', '-b', type=int,
                    help='Batch size', required=False, default=8)
parser.add_argument('--learning_rate', '-lr', type=float,
                    help='Learning rate', required=False, default=2e-5)
parser.add_argument('--epochs', '-e', type=int,
                    help='Number of epochs', required=False, default=20)
parser.add_argument('--username', '-u', type=str,
                    help='Username for HuggingFace Hub', required=False)
parser.add_argument('--password', '-p', type=str,
                    help='Password for HuggingFace Hub', required=False)
parser.add_argument('--output', '-o', type=str,
                    help='Output directory path', required=False, default='saved_models/')
parser.add_argument('--hub_name', '-hn', type=str,
                    help='Name of the model in the HuggingFace Hub', required=False)
args = parser.parse_args()
print(args)

MODEL_NAME = args.model
DATASET_PATH = args.dataset
PUSH_TO_HUB = args.hub
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
USERNAME = args.username
PASSWORD = args.password
OUTPUT_PATH = args.output
HUB_NAME = args.hub_name

if PUSH_TO_HUB is not None and PUSH_TO_HUB:
    if USERNAME is None or PASSWORD is None:
        print("Please provide username and password for pushing to HuggingFace Hub!\nRun the script with python maple.py -h for help.")
        exit()
    else:
        print("Logging into HuggingFace Hub!")
        if platform.system() == "Linux":
            os.system(
                f"printf '{USERNAME}\{PASSWORD}' | transformers-cli login")
        else:
            print(
                "Could not login to HuggingFace Hub automatically! Please enter credentials again")
            os.system("transformers-cli login")

df = pd.read_json(DATASET_PATH)
df = df.drop_duplicates(subset=["passage", "poem"])
df["tokens"] = df["poem"].apply(lambda x: word_tokenize(x))
ner_tags = list()
for i in range(df.shape[0]):
    indices = df["indices"][i]
    length = len(df["tokens"][i])
    ner_tag = ['O' for _ in range(length)]
    for idx in indices:
        ner_tag[idx] = 'W'
    ner_tags.append(ner_tag)
df["ner_tags"] = ner_tag

train_df = df
test_df = df.sample(frac=0.1, random_state=0)

tokens = df["tokens"]
tags = df["ner_tags"]
unique_tags = ["O", "W"]
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
train_texts = list(train_df["tokens"].values)
val_texts = list(test_df["tokens"].values)
train_tags = list(train_df["ner_tags"].values)
val_tags = list(test_df["ner_tags"].values)

if "xlm-roberta" in MODEL_NAME:
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(
        MODEL_NAME, add_prefix_space=True)
elif "roberta" in MODEL_NAME:
    tokenizer = RobertaTokenizerFast.from_pretrained(
        MODEL_NAME, add_prefix_space=True)
elif "gpt" in MODEL_NAME:
    tokenizer = GPT2TokenizerFast.from_pretrained(
        MODEL_NAME, add_prefix_space=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_encodings = tokenizer(train_texts, is_split_into_words=True,
                            return_offsets_mapping=True, padding=True, truncation=True)
val_encodings = tokenizer(val_texts, is_split_into_words=True,
                          return_offsets_mapping=True, padding=True, truncation=True)
train_labels = encode_tags(train_tags, train_encodings)
val_labels = encode_tags(val_tags, val_encodings)

offset_mapping_train = train_encodings.pop("offset_mapping")
offset_mapping_val = val_encodings.pop("offset_mapping")
train_dataset = MapleDataset(train_encodings, train_labels)
val_dataset = MapleDataset(val_encodings, val_labels)
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME, num_labels=2)

args = TrainingArguments(
    f"{OUTPUT_PATH}/{MODEL_NAME}",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.1,
    # save_total_limit=1,   # to save only the last checkpoint
    logging_steps=50
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()
# trainer.evaluate()

if PUSH_TO_HUB is not None and PUSH_TO_HUB:
    print("Pushing to HuggingFace Hub!")
    model.push_to_hub(HUB_NAME)
    tokenizer.push_to_hub(HUB_NAME)
