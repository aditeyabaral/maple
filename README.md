# MAPLE - Masking words to generate blackout Poetry using sequence-to-sequence LEarning
Implementation of the paper, **Maple - Masking words to generate blackout Poetry using sequence-to-sequence LEarning**, ICNLSP 2021 

[[PDF]](https://aclanthology.org/2021.icnlsp-1.6.pdf)
[[MAPLE MODELS]](https://huggingface.co/maple)

If you would like to cite our work, please use the following BiBTex

```tex
@inproceedings{baral-etal-2021-maple,
    title = "{MAPLE} {--} {MA}sking words to generate blackout Poetry using sequence-to-sequence {LE}arning",
    author = "Baral, Aditeya  and
      Jain, Himanshu  and
      D, Deeksha  and
      R, Dr. Mamatha H",
    booktitle = "Proceedings of The Fourth International Conference on Natural Language and Speech Processing (ICNLSP 2021)",
    month = "12--13 " # nov,
    year = "2021",
    address = "Trento, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.icnlsp-1.6",
    pages = "47--54",
}
```

# How to use MAPLE

## Setup MAPLE Environment

1. Clone this repository
2. Setup your environment with:
```bash
conda env create -f environment.yml
```

## Create Training Data

If you do not already have a training dataset, you can create one by combining Liza Daly's pattern extraction approach with statistically found rules (as described in the paper).

1. Obtain a large collection of short, one line poems, haikus or phrases and save them in a file named `poems.txt`, one poem per line. These will be used to find repeating PoS grammar rules. Once you do so, run 
```bash
python generate_grammar_rules_from_poems.py -i poems.txt -o all_grammar.txt
```

2. Obtain the top `N` repeating PoS grammar rules from these poems. These will be used to extract sequences from passages. Copy the list generated at the end.
```bash
python generate_topN_grammar.py -i all_grammar_rules.txt -o top_grammar_rules.txt -n 7
```

3. Liza Daly's apprach (read paper) is used to create the training dataset. Copy the output from the previous command and replace the `grammars` list in `liza_daly_poem_generator.py` with the list generated in the previous step. Obtain a large collection of passages to generate blackout poems from and name it `passages.txt`. Then run
```bash
python liza_daly_poem_generator.py -i passages.txt -o dataset.json
```

The file created at the end is the final dataset. The dataset *might* have to be processed a little to fit the format required by the scripts.

## Dataset Format

MAPLE required a dataset in the following format:

| passage                                           | poem                                  | indices           |
|---------------------------------------------------|---------------------------------------|-------------------|
| Did the CIA tell the FBI that it knows the wor... | cia fbi the biggest weapon            | [2, 5, 9, 24, 25] |
| A vigilante lacking of heroic qualities that\n... | lacking qualities that damn criminals | [2, 5, 6, 11, 12] |

The passage is the text from which the poem is generated. The poem is the generated poem. The indices are the indices of the words in the text that are chosen for the poem.

## Train MAPLE

To train the primary Transformer based MAPLE model, use the `maple.py` script inside `models/`. The arguments are self explanatory:
```bash
usage: python maple.py [-h] --model MODEL --dataset DATASET [--hub HUB] [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE]
                [--epochs EPOCHS] [--username USERNAME] [--password PASSWORD] [--output OUTPUT] [--hub_name HUB_NAME]


optional arguments:
  -h, --help            show this help message and exit
  --model MODEL, -m MODEL
                        Transformer model name/path to siamese pre-train
  --dataset DATASET, -d DATASET
                        Path to dataset in required format
  --hub HUB, -hf HUB    Push model to HuggingFace Hub
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        Batch size
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
                        Learning rate
  --epochs EPOCHS, -e EPOCHS
                        Number of epochs
  --username USERNAME, -u USERNAME
                        Username for HuggingFace Hub
  --password PASSWORD, -p PASSWORD
                        Password for HuggingFace Hub
  --output OUTPUT, -o OUTPUT
                        Output directory path
  --hub_name HUB_NAME, -hn HUB_NAME
                        Name of the model in the HuggingFace Hub
```

Alternatively, you can also compare it with abstractive summarization. The `models/abstractive_summarization.py` script can be run using the following arguments.

```bash
usage: python abstractive_summarization.py [-h] --input INPUT --output OUTPUT [--poem_length POEM_LENGTH]
                                    [--text_length TEXT_LENGTH] [--split SPLIT] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                                    [--embedding_dim EMBEDDING_DIM] [--hidden_dim HIDDEN_DIM]


optional arguments:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        Path to the training data
  --output OUTPUT, -o OUTPUT
                        Path to store generated results
  --poem_length POEM_LENGTH, -pl POEM_LENGTH
                        Minimum poem length to train on
  --text_length TEXT_LENGTH, -tl TEXT_LENGTH
                        Maximum passage length to train on
  --split SPLIT, -s SPLIT
                        Whether to split the dataset into train/test
  --epochs EPOCHS, -e EPOCHS
                        Number of epochs to train
  --batch_size BATCH_SIZE, -bs BATCH_SIZE
                        Batch size for training
  --embedding_dim EMBEDDING_DIM, -ed EMBEDDING_DIM
                        Dimension of the embedding layer
  --hidden_dim HIDDEN_DIM, -hd HIDDEN_DIM
                        Dimension of the LSTM layers
```
