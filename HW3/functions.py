from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
from torch.nn.utils.rnn import pack_padded_sequence
import unicodedata
import torch
import torch.nn as nn
import random
import time
import math




def findFiles(path): return glob.glob(path)

def unicodeToAscii(s, all_letters):
    # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename, all_letters):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line, all_letters) for line in lines]

def letterToIndex(letter, all_letters):
    # Find letter index from all_letters, e.g. "a" = 0
    return all_letters.find(letter)

def letterToTensor(letter, n_letters):
    # Just for demonstration, turn a letter into a <1 x n_letters> Tensor
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


def lineToTensor(line, n_letters, all_letters):
    # Turn a line into a <line_length x 1 x n_letters>,
    # or an array of one-hot letter vectors
    tensor = torch.zeros(len(line), 1, n_letters)
    for i, letter in enumerate(line):
        tensor[i][0][letterToIndex(letter, all_letters)] = 1
    return tensor

def categoryFromOutput(output, all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i



def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample(all_categories, category_lines, n_letters, all_letters):
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line, n_letters, all_letters)
    return category, line, category_tensor, line_tensor

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# Just return an output given a line
def evaluate(line_tensor, model):
    # Evaluating the Results
    hidden = model.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)

    return output