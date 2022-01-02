# MAPLE - Masking words to generate blackout Poetry using sequence-to-sequence LEarning
Implementation of the paper, **Maple - Masking words to generate blackout Poetry using sequence-to-sequence LEarning**, ICNLSP 2021

The link to the paper will soon be added after the presentation.

If you would like to cite our work, please use the following BiBtex
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

## Authors

[Aditeya Baral](https://github.com/aditeyabaral) <br>
[Himanshu Jain](https://github.com/nhimanshujain) <br>
[Deeksha D](https://github.com/deeksha-d)


# How to use MAPLE

## Setup MAPLE Environment

1. Clone this repository
2. Setup your environment with:
```bash
conda env create -f environment.yml
```

## Create Training Data

Training data can be created by combining Liza Daly's pattern extraction approach with statistically found rules.

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

## Train MAPLE

To train MAPLE, simply run each script with its respective arguments. YOu can view the arguments for each script using thr `-h` flag.

```bash
python script_name_in_models.py -h
```

