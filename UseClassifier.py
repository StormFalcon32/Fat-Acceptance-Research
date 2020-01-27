import csv

import numpy as np
import pandas as pd
from joblib import load
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# io
import InputOutput as io
import pickle

data_words = io.csvIn(r'Overall\TextsDup.csv', False)
vocab = pickle.load(open(r'D:\Python\FatAcceptance\vocab.pkl', 'rb'))
tfidf = TfidfVectorizer(preprocessor=lambda x: x,
                        tokenizer=lambda x: x, vocabulary=vocab)
X = tfidf.fit_transform(data_words)
labels = []
estimator = load(r'D:\Python\FatAcceptance\model.joblib')
labels = estimator.predict(X).tolist()
io.csvOutSingle(r'Overall\Labels.csv', labels)
