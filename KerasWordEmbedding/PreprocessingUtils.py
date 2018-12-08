from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


from numpy import argmax
from numpy import array

from nltk.translate.bleu_score import corpus_bleu



class PreprocessingUtils:

    def __init__(self):
        pass

    #encode docs using keras one_hot() function
    def hot_ecode_docs(self, docs, vocab_size, toLower=True):
        encoded_docs = [one_hot(d, vocab_size, lower=toLower) for d in docs]
        return encoded_docs

    #pad docs to a given maxLen using Keras pad_squences() function
    def pad_sequences(self, docs):
        padded_docs = pad_sequences(docs, maxlen=self.get_maxLen(docs), padding='post')
        return padded_docs

    def get_maxLen(self, docs):
        return max(len(line) for line in docs)


    # fit a tokenizer
    def create_tokenizer(self, lines):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines)
        return tokenizer

    def get_vocab_size(self, docs):
        tokenizedDocs = self.create_tokenizer(docs)

        return len(tokenizedDocs.word_index)