# from consts import *
from scipy.sparse import csr_matrix


def load_raw_dataset(txtFilePath):
    with open(txtFilePath) as f:
        lines = f.readlines()
    dataset = [line.strip().split(sep=" ") for line in lines if not(line in ['\n', '. . O\n'])]
    return [item[0] for item in dataset], [item[1] for item in dataset], [item[2] for item in dataset]


def convert_raw_to_features(lWords):
    return [word[0].isupper() for word in lWords]


def get_dataset():
    pass
