import pandas as pd
import numpy as np
from scipy import sparse
import csv

# pickle
from joblib import load

# preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

withdate = []
data_words = []
labels = []
with open(r'D:\Python\FatAcceptance\Overall\TextsDup.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        data_words.append(row)
tfidf = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
x = tfidf.fit_transform(data_words)
lsa = TruncatedSVD(n_components=100)
x = lsa.fit_transform(x)
svm = load(r'D:\Python\FatAcceptance\svm.joblib')
labels = svm.predict(x).tolist()
with open(r'D:\Python\FatAcceptance\Overall\Labels.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(labels)
