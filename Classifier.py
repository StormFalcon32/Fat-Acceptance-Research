from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import csv
from sklearn.decomposition import TruncatedSVD
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score

np.random.seed(100)

data_words = []
labels = []
with open(r'D:\Python\FatAcceptance\Texts0.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        data_words.append(row)
with open(r'D:\Python\FatAcceptance\Labels.csv') as f:
    reader = csv.reader(f)
    row = next(reader)
    for j in range(0, len(row)):
        labels.append(int(row[j]))
tfidf = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
x = tfidf.fit_transform(data_words)
y = np.array(labels)
lsa = TruncatedSVD(n_components=100, n_iter=10)
x = lsa.fit_transform(x)
x_comb, x_val, y_comb, y_val = model_selection.train_test_split(
    x, y, test_size=0.2, stratify=y)
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x_comb, y_comb, test_size=0.2, stratify=y_comb)

SVM = svm.SVC(C=10.0, kernel='poly', degree=3, gamma='scale')
SVM.fit(x_train, y_train)
predictions_SVM = SVM.predict(x_test)
print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, y_test) * 100)
