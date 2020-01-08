import pandas as pd
import numpy as np
import csv
from sklearn import model_selection, naive_bayes, svm
from sklearn.feature_extraction.text import TfidfVectorizer

np.random.seed(100)

data_words = []
with open(r'D:\Python\FatAcceptance\Texts.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        data_words.append(row)
tfidf = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
X = tfidf.fit_transform(data_words)
print(X.shape)
