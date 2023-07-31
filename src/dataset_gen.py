"""
Script creates train-test data
"""

# Import relevant libraries
from datasets import load_dataset  # HuggingFace library
from pathlib import Path


class TextDatasetLoad:
    """
    Class relating to the loading in of a text corpus
    """

    def __init__(self, hugging_face_dataset='tiny_shakespeare', filepath=None):
        self.filepath = filepath
        self.dataset = hugging_face_dataset

    def load_textfile(self):
        with open(self.filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        return text

    def load_huggingface_dataset(self):
        dataset = load_dataset(self.dataset)
        train_data, val_data, test_data = dataset['train'], dataset['validation'], dataset['test']
        return train_data, val_data, test_data



class TextTokenizer:
    """
    Class to tokenize a given text corpus
    """
    def __init__(self, raw_data, token_level='char', tokenizer='int'):
        if type(raw_data) == str:
            self.raw_data = raw_data  # TODO: for large corpus not very memory efficient?
        else:  # experimenting with using hugging face dataset type TODO: implement explicit check
            self.raw_data = raw_data['text'][0]
        self.token_level = token_level
        self.tokenizer = tokenizer
        self.vocab = self.vocab_gen()
        self.vocab_len = len(self.vocab)

    def vocab_gen(self):
        """
        Method returns the vocabulary set of the text corpus under consideration
        """
        if self.token_level == 'char':
            return sorted(list(set(self.raw_data)))
        else:
            raise ValueError("Currently only character level tokenisation supported")

    def token_encoder(self, text_data):
        """
        Method returns numerical representation for vocabulary elements for given tokenisation scheme
        """
        if self.tokenizer == 'int':
            # Mapping vocab element to integers
            s2i = {v: i for i, v in enumerate(self.vocab)}
        else:
            raise ValueError("Currently only integer mapped tokenisation supported")

        encoded = [s2i.get(v, -1) for v in text_data]  # to deal with out-of-vocabulary elements encode -1
        return encoded

    def token_decoder(self, tokenised_data):
        """
        Method returns text representation from numerical tokenised representation for given tokenisation scheme
        """
        if self.tokenizer == 'int':
            # Mapping from integers to vocab elements
            i2s = {i: v for i, v in enumerate(self.vocab)}
        else:
            raise ValueError("Currently only integer mapped tokenisation supported")

        decoded = [i2s.get(i, '*') for i in tokenised_data]  # asterisks represents out-of-vocab encodings
        return decoded


