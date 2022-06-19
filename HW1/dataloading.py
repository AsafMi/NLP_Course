# from consts import *
from scipy.sparse import csr_matrix


def load_raw_dataset(txtFilePath):
    with open(txtFilePath) as f:
        lines = f.readlines()
    dataset = [line.strip().split(sep=" ") for line in lines if not(line in ['\n', '. . O\n'])]
    return [item[0] for item in dataset], [item[1] for item in dataset], [[item[2]] for item in dataset]


def InListFeature(x_train):
    days_months = ['Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'Monday', 'Tuesday']
    days_months.extend(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
    return [word in days_months for word in x_train]