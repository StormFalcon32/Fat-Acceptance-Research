from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import csv
from sklearn.decomposition import TruncatedSVD
from sklearn import model_selection, svm
from scipy import sparse


def score(prediction, labels, test_mat, inv):
    correct = 0
    for i in range(0, len(labels)):
        if prediction[i] != labels[i]:
            print("Wrong - ind: %s pred: %s actual: %s" %
                  (i, prediction[i], labels[i]))
        else:
            print("Right - ind: %s pred: %s" % (i, prediction[i]))
            correct += 1
    print("Score %s" % (correct * 100 / len(labels)))
    words = [[] for i in range(test_mat.shape[0])]
    for i, j in zip(test_mat.row, test_mat.col):
        words[i].append(inv[j])
    with open(r'D:\Python\FatAcceptance\Training\TestWords10.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(words)
    with open(r'D:\Python\FatAcceptance\Training\TestLabels10.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(labels)


np.random.seed(500)

data_words = []
labels = []
with open(r'D:\Python\FatAcceptance\Training\Texts10.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        data_words.append(row)
with open(r'D:\Python\FatAcceptance\Training\Labels.csv') as f:
    reader = csv.reader(f)
    row = next(reader)
    for j in range(0, len(row)):
        labels.append(int(row[j]))
tfidf = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
x = tfidf.fit_transform(data_words)
inv_vocab = {v: k for k, v in tfidf.vocabulary_.items()}
lsa = TruncatedSVD(n_components=20, n_iter=100)
x_reduced = lsa.fit_transform(x)
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, labels, test_size=0.2, stratify=labels)
cx = sparse.coo_matrix(x_test)
SVM = svm.SVC(C=3, kernel='poly', degree=1, gamma='scale')
SVM.fit(x_train, y_train)
predictions = SVM.predict(x_test)
score(predictions, y_test, cx, inv_vocab)
