import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    RobertaTokenizerFast,
    XLMRobertaTokenizerFast,
)


class MAPLE(nn.Module):
    def __init__(self, transformer_model_path="roberta-base", device="cpu"):
        super(MAPLE, self).__init__()
        self.device = device
        self.model = AutoModelForTokenClassification.from_pretrained(
            transformer_model_path
        )
        self.tokenizer = self.get_tokenizer(transformer_model_path)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def get_tokenizer(self, transformer_model_path):
        if "xlm-roberta" in transformer_model_path:
            tokenizer = XLMRobertaTokenizerFast.from_pretrained(
                transformer_model_path, add_prefix_space=True
            )
        elif "roberta" in transformer_model_path:
            tokenizer = RobertaTokenizerFast.from_pretrained(
                transformer_model_path, add_prefix_space=True
            )
        elif "gpt" in transformer_model_path:
            tokenizer = GPT2TokenizerFast.from_pretrained(
                transformer_model_path, add_prefix_space=True
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(transformer_model_path)

        special_tokens = {
            "bos_token": "<|startoftext|>",
            "pad_token": "<|padtext|>",
            "sep_token": "<|septext|>",
        }
        tokenizer.add_special_tokens(special_tokens)
        return tokenizer

    def get_encodings_and_labels(self, tokens, labels, label_all_tokens=False):
        token_encodings = list()
        attention_masks = list()
        label_encodings = list()
        for idx, t in enumerate(tokens):
            tokenized_input = self.tokenizer.encode_plus(
                t,
                is_split_into_words=True,
                max_length=128,
                padding="max_length",
                truncation=True,
            )
            input_ids = torch.tensor(tokenized_input["input_ids"])
            attention_mask = torch.tensor(tokenized_input["attention_mask"])
            token_encodings.append(input_ids)
            attention_masks.append(attention_mask)

            word_ids = tokenized_input.word_ids(batch_index=0)
            previous_word_idx = None
            label_ids = list()
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(labels[idx][word_idx])
                else:
                    label_ids.append(
                        labels[idx][word_idx] if label_all_tokens else -100
                    )
                previous_word_idx = word_idx
            label_encodings.append(torch.tensor(label_ids))

        token_encodings = torch.stack(token_encodings).to(self.device)
        attention_masks = torch.stack(attention_masks).to(self.device)
        label_encodings = torch.stack(label_encodings).to(self.device)
        return token_encodings, attention_masks, label_encodings

    def forward(self, tokens, labels):
        output_sequences = list()
        (
            token_encodings,
            attention_masks,
            label_encodings,
        ) = self.get_encodings_and_labels(tokens, labels)
        outputs = self.model(
            input_ids=token_encodings,
            attention_mask=attention_masks,
            labels=label_encodings,
        )
        loss, logits = outputs[:2]
        logits = torch.softmax(logits, dim=2).argmax(dim=2)

        for i in range(token_encodings.shape[0]):
            input_sequence = token_encodings[i]
            logit_sequence = logits[i]
            attention_mask = attention_masks[i]
            output_sequence = list()
            for j in range(input_sequence.shape[0]):
                if attention_mask[j] and logit_sequence[j]:
                    output_sequence.append(input_sequence[j])
            output_sequence = self.tokenizer.decode(output_sequence).strip()
            if not output_sequence:  # if no words are chosen, take empty string
                output_sequence = ""
            output_sequences.append(output_sequence)

        return loss, output_sequences

    def generate(self, tokens):
        if isinstance(tokens[0], str):
            tokens = [tokens]
        output_sequences = list()
        token_encodings = list()
        attention_masks = list()
        for t in tokens:
            tokenized_input = self.tokenizer.encode_plus(
                t,
                is_split_into_words=True,
                max_length=128,
                padding="max_length",
                truncation=True,
            )
            input_ids = torch.tensor(tokenized_input["input_ids"])
            attention_mask = torch.tensor(tokenized_input["attention_mask"])
            token_encodings.append(input_ids)
            attention_masks.append(attention_mask)
            outputs = self.model(
                input_ids=input_ids.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
                labels=None,
            )
            loss, logits = outputs[:2]
            logits = torch.softmax(logits, dim=2).argmax(dim=2)
            logit_sequence = logits[0]
            output_sequence = list()
            for j in range(input_ids.shape[0]):
                if attention_mask[j] and logit_sequence[j]:
                    output_sequence.append(input_ids[j])
            output_sequence = self.tokenizer.decode(output_sequence).strip()
            if not output_sequence:  # if no words are chosen, take empty string
                output_sequence = ""
            output_sequences.append(output_sequence)

        return output_sequences
