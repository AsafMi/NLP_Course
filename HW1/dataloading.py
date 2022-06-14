from consts import *
from scipy.sparse import csr_matrix


def load_raw_dataset(txtFilePath):
    with open(txtFilePath) as f:
        lines = f.readlines()
    dataset = [line.strip().split(sep="\t") for line in lines]
    return [item[0] for item in dataset], [item[1] for item in dataset]


def convert_raw_to_features():
    pass


def get_dataset():
    pass
