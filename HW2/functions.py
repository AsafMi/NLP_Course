import numpy as np
import pandas as pd
import torch
from bs4 import BeautifulSoup
import re
from sklearn.preprocessing import scale


# Using random seed as demanded
np.random.seed(23)
torch.manual_seed(23)


def dataPreprocess(path):
    df = pd.read_csv(r"./data/IMDB_train.csv")
    lReviews = df.review.to_list()
    lSentiment = df.sentiment.to_list()
    y = [1. if s == "positive" else 0. for s in lSentiment]
    y = np.asarray(y, dtype=np.float32)
    lLines = []
    for review in lReviews:
        text = BeautifulSoup(review, features="html.parser").get_text()  # -- remove <br> and HTML
        lWords = re.sub("[^a-zA-Z]", " ", text).lower().split()  # -- keep lower case letters
        lLines.append(lWords)
    return lLines, y


def line2vec(lLines, oWord2Vec):
    x = []
    d = oWord2Vec.vector_size
    for line in lLines:
        vec = np.zeros(d).reshape((1, d))
        count = 0
        for word in line:
            try:
                vec += oWord2Vec.wv[word].reshape((1, d))
                count += 1.
            except KeyError:
                continue
        if count != 0:
            vec /= count
        x.append(vec)

    x = scale(np.concatenate(x))
    return x


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag.squeeze() == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

