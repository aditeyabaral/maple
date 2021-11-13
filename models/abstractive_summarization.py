
import re
import nltk
import json
import argparse
import numpy as np
import pandas as pd
from gensim.models import FastText
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from attention import AttentionLayer
from keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional

np.random.seed(0)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def cleanText(text):
  text = text.lower()
  text = re.sub(" +", " ", text)
  text = re.sub("\n+", " ", text)
  text = re.sub("[^a-zA-Z0-9\n']", " ", text)
  text = re.sub(r'\([^)]*\)', '', text)
  text = re.sub('"','', text)
  text = re.sub(r"'s\b","",text)
  text = re.sub(" +", " ", text)
  text = re.sub("\n+", " ", text)
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

    haikus = list(map(lambda x: 'starttoken ' + haiku + ' endtoken'), haikus)
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


def trainFastTextModel():
    ft_model = FastText(min_count=1)
    ft_model.build_vocab(sentences=X_words+y_words)
    ft_model.train(sentences=X_words+y_words, total_examples=ft_model.corpus_count, epochs=5)
    word2embedding = {word:ft_model.wv.get_vector(word) for word in list(ft_model.wv.vocab)}
    return word2embedding


def fitTokenizer(text, T):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(list(text))
    seq = tokenizer.texts_to_sequences(text)
    seq = pad_sequences(seq,  maxlen=T, padding='post') 
    vocab_size = len(tokenizer.word_index) + 1 
    return tokenizer, seq, vocab_size


def getEmbeddingMatrix(vocab_size, tokenizer):
    embedding_matrix = np.zeros((vocab_size, 100))
    for word, i in tokenizer.word_index.items():
        embedding_vector = word2embedding.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


def decodeSequence(input_seq):
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    target_seq = np.zeros((1,1))
    target_seq[0, 0] = target_word_index['starttoken']
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]
        if(sampled_token!='endtoken'):
            decoded_sentence += ' '+sampled_token
        if (sampled_token == 'endtoken'  or len(decoded_sentence.split()) >= (Ty-1)):
            stop_condition = True
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index
        e_h, e_c = h, c
    return decoded_sentence


def seq2summary(input_seq):
    newString=''
    for i in input_seq:
        if((i!=0 and i!=target_word_index['starttoken']) and i!=target_word_index['endtoken']):
            newString=newString+reverse_target_word_index[i]+' '
    return newString

def seq2text(input_seq):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+reverse_source_word_index[i]+' '
    return newString


parser = argparse.ArgumentParser("Train MAPLE using Abstractive Summarization")
parser.add_argument("--input", "-i", type=str,
                    help="Path to the training data", required=True)
parser.add_argument("--output", "-o", type=str,
                    help="Path to store generated results", required=True)
parser.add_argument("--poem_length", "-pl", type=str,
                    default=5, help="Minimum poem length to train on")
parser.add_argument("--text_length", "-tl", type=str,
                    default=120, help="Maximum passage length to train on")
parser.add_argument("--split", "-s", type=bool, default=True,
                    help="Whether to split the dataset into train/test")
args = parser.parse_args()

input_file = args.input
output_file = args.output
poem_length = int(args.poem_length)
text_length = int(args.text_length)
split = args.split

df = createDataFrame(input_file, clean=True)

X_text = df["poem"].values
y_text = df["haiku"].values
X_words = list(map(word_tokenize, df["poem"].values))
y_words = list(map(word_tokenize, df["haiku"].values))
word2embedding = trainFastTextModel()

Tx = len(max(X_words, key=len))
Ty = len(max(y_words, key=len))

tokenizer_X, X_seq, vocab_size_X = fitTokenizer(X_text, Tx)
tokenizer_y, y_seq, vocab_size_y = fitTokenizer(y_text, Ty)

embedding_matrix_X = getEmbeddingMatrix(vocab_size_X, tokenizer_X)
embedding_matrix_y = getEmbeddingMatrix(vocab_size_y, tokenizer_y)

latent_dim = 1024

encoder_input = Input(shape=(Tx, ))
encoder_embedding =  Embedding(vocab_size_X, 100, weights=[embedding_matrix_X], trainable=False)(encoder_input)

encoder_LSTM_1 = Bidirectional(LSTM(latent_dim, return_state=True, return_sequences=True))
encoder_output, forward_h, forward_c, backward_h, backward_c = encoder_LSTM_1(encoder_embedding)

state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
encoder_states = [state_h, state_c]

decoder_input = Input(shape=(None, ))
decoder_embedding_layer = Embedding(vocab_size_y, 100, weights=[embedding_matrix_y], trainable=False)
decoder_embedding = decoder_embedding_layer(decoder_input)

decoder_LSTM_1 = LSTM(2*latent_dim, return_state=True, return_sequences=True)
decoder_output, decoder_fwd_state, decoder_back_state = decoder_LSTM_1(decoder_embedding, initial_state=encoder_states)

attention_layer = AttentionLayer()
attention_output, attention_states = attention_layer([encoder_output, decoder_output])
decoder_concat = Concatenate(axis=-1)([decoder_output, attention_output])

decoder_dense = TimeDistributed(Dense(vocab_size_y, activation='softmax'))
decoder_output = decoder_dense(decoder_concat)

model = Model([encoder_input, decoder_input], decoder_output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

if split:
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.05, random_state=42)
    history = model.fit(
        [X_train, y_train[:,:-1]], 
        y_train.reshape(y_train.shape[0],y_train.shape[1], 1)[:,1:],
        epochs=4,
        batch_size=128, 
        validation_data=([X_test, y_test[:,:-1]], y_test.reshape(y_test.shape[0],y_test.shape[1], 1)[:,1:])
        )
else:
    history = model.fit(
        [X_seq, y_seq[:,:-1]], 
        y_seq.reshape(y_seq.shape[0],y_seq.shape[1], 1)[:,1:],
        epochs=4,
        batch_size=128
    )

reverse_target_word_index = tokenizer_y.index_word
reverse_source_word_index = tokenizer_X.index_word
target_word_index = tokenizer_y.word_index

encoder_model = Model(inputs=encoder_input, outputs=[encoder_output, state_h, state_c])

decoder_state_input_h = Input(shape=(2 * latent_dim,))
decoder_state_input_c = Input(shape=(2 * latent_dim,))
decoder_hidden_state_input = Input(shape=(Tx, 2 * latent_dim))
dec_emb2 = decoder_embedding_layer(decoder_input) 
decoder_outputs2, state_h2, state_c2 = decoder_LSTM_1(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])
attn_out_inf, attn_states_inf = attention_layer([decoder_hidden_state_input, decoder_outputs2])
decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])
decoder_outputs2 = decoder_dense(decoder_inf_concat) 
decoder_model = Model(
    [decoder_input] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2] + [state_h2, state_c2])

with open(output_file, "w") as f:
    data = list()
    if split:
        for i in range(len(X_test)):
            passage = seq2text(X_test[i])
            poem = decodeSequence(X_test[i].reshape(1, Tx))
            data.append({"passage": passage, "poem": poem})
    else:
        for i in range(len(X_seq)):
            passage = seq2text(X_seq[i])
            poem = decodeSequence(X_seq[i].reshape(1, Tx))
            data.append({"passage": passage, "poem": poem})
    json.dump(data, f)