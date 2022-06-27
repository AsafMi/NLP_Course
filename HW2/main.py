import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re
from gensim.models import word2vec

df = pd.read_csv(r"data\IMDB_train.csv")
lReviews = df.review.to_list()
lSentiment = df.sentiment.to_list()
lLines = []
for review in lReviews:
    text   = BeautifulSoup(review, features="html.parser").get_text()                 #-- remove <br> and HTML
    lWords = re.sub("[^a-zA-Z]", " ", text).lower().split() #-- keep lower case letters
    lLines.append(lWords)
# Train Word2Vec model
d            = 300
minWordCount = 40
contextWin   = 5

oWord2Vec    = word2vec.Word2Vec(lLines, workers=4, vector_size=d, min_count=minWordCount, window=contextWin)

x_train = []
for line in lLines:
    vec = np.zeros(d).reshape((1, d))
    count = 0
    for word in line:
        try:
            vec += oWord2Vec[word].reshape((1, d))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    x_train.append(vec)

a=1