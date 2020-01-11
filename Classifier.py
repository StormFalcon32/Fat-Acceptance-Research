import pandas as pd
import numpy as np
from scipy import sparse
import csv

# classifiers
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import model_selection

# scoring
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def debugScore(prediction, labels, test_mat, inv):
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


def model_score_df(model_dict):
    model_name, ac_score_list, p_score_list, r_score_list, f1_score_list = [], [], [], [], []
    for k, v in model_dict.items():
        model_name.append(k)
        v.fit(x_train, y_train)
        y_pred = v.predict(x_test)
        ac_score_list.append(accuracy_score(y_test, y_pred))
        p_score_list.append(precision_score(y_test, y_pred, average='macro'))
        r_score_list.append(recall_score(y_test, y_pred, average='macro'))
        f1_score_list.append(f1_score(y_test, y_pred, average='macro'))
        model_comparison_df = pd.DataFrame(
            [model_name, ac_score_list, p_score_list, r_score_list, f1_score_list]).T
        model_comparison_df.columns = [
            'model_name', 'accuracy_score', 'precision_score', 'recall_score', 'f1_score']
        model_comparison_df = model_comparison_df.sort_values(
            by='f1_score', ascending=False)
    return model_comparison_df


np.random.seed(1000)

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
lsa = TruncatedSVD(n_components=40, n_iter=100)
x_reduced = lsa.fit_transform(x)
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, labels, test_size=0.2, stratify=labels)
cx = sparse.coo_matrix(x_test)
models = {'Dummy': DummyClassifier(),
          'SGD': SGDClassifier(loss='log'),
          'RF': RandomForestClassifier(),
          'DT': DecisionTreeClassifier(),
          'AB': AdaBoostClassifier(),
          'KNN': KNeighborsClassifier(),
          'MNB': MultinomialNB(),
          'SVM': SVC(),
          'LR': LogisticRegression()
          }

scores = model_score_df(models)
print(scores)
