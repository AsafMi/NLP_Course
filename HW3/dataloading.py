import torch
from torch.utils.data import Dataset
import pandas as pd
import gensim

from consts import *


class TweetDataset(Dataset):
    def __init__(self, data_args, file_path, vocab=None):
        self.data_args = data_args
        self.file_path = file_path

        # Load data to dataframe
        self.df = pd.read_csv(file_path)
        text = ''.join(self.df.text.to_list())
        # Get vocab
        if vocab is None:
            # Tokenize all of the text using gensim.utils.tokenize(text, lowercase=True)
            tokenized_text = gensim.utils.tokenize(text, lowercase=True)
            # Create a set of all the unique tokens in the text
            self.vocab = set(tokenized_text)
        else:
            self.vocab = vocab


        # Set the vocab size
        self.vocab_size = len(self.vocab)
        # Add the UNK token to the vocab
        self.unk_token = self.vocab_size
        # Create a dictionary mapping tokens to indices
        self.token2id = {value: index for index, value in enumerate(self.vocab)}
        self.id2token = {index: value for index, value in enumerate(self.vocab)}

        # Tokenize data using the tokenize function
        self.df[INPUT_IDS] = self.df.apply(lambda row: self.tokenize(row.text), axis=1)

    def __len__(self):
        # Return the length of the dataset
        return len(self.df)

    def __getitem__(self, idx):
        # Get the row at idx
        row = self.df.iloc[idx]
        input_ids = self.tokenize(row.text)
        label = row.label
        # return the input_ids and the label as tensors, make sure to convert the label type to a long
        return torch.tensor(input_ids), torch.tensor(label, dtype=torch.long)

    def tokenize(self, text):
        tokenized_text = list(gensim.utils.tokenize(text, lowercase=True))
        input_ids = []
        # Tokenize the text using gensim.utils.tokenize(text, lowercase=True)
        for idx, word in enumerate(text.split()):
            # Make sure to trim sequences to max_seq_length
            if idx <= self.data_args['max_seq_length']:
                # Gets the token id, if unknown returns self.unk_token
                try:
                    input_ids.append(self.token2id[word])
                except:
                    input_ids.append(self.unk_token)

        return input_ids
