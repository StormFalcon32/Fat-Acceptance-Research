import pandas as pd
import numpy as np
from scipy import sparse
import csv
from joblib import dump

# preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import model_selection

# classifiers
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# kfold and pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# scoring
from sklearn.metrics import accuracy_score, f1_score


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


def score_models(models):
    name = []
    accuracy = []
    f1 = []
    train = []
    for k, v in models.items():
        name.append(k)
        v.fit(x_train, y_train)
        y_pred = v.predict(x_test)
        y_pred_train = v.predict(x_train)
        accuracy.append(accuracy_score(y_true=y_test, y_pred=y_pred))
        f1.append(f1_score(y_true=y_test, y_pred=y_pred, average='macro'))
        train.append(
            f1_score(y_true=y_train, y_pred=y_pred_train, average='macro'))
    compare = pd.DataFrame([name, accuracy, f1, train])
    compare = compare.T
    compare.columns = ['name', 'accuracy', 'f1', 'train']
    compare = compare.sort_values(by='f1')
    return compare


np.random.seed(1)

data_words = []
labels = []
with open(r'D:\Python\FatAcceptance\Training\Final\TrainingTexts.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        data_words.append(row)
with open(r'D:\Python\FatAcceptance\Training\Final\Labels.csv') as f:
    reader = csv.reader(f)
    row = next(reader)
    for j in range(0, len(row)):
        labels.append(int(row[j]))
tfidf = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
x = tfidf.fit_transform(data_words)
inv_vocab = {v: k for k, v in tfidf.vocabulary_.items()}
# lsa = TruncatedSVD(n_components=100)
# x = lsa.fit_transform(x)
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, labels, test_size=0.2, stratify=labels)
cx = sparse.coo_matrix(x_test)
# models = {'Stratified': DummyClassifier(strategy='stratified'),
#           'Frequent': DummyClassifier(strategy='most_frequent'),
#           'Prior': DummyClassifier(strategy='prior'),
#           'Uniform': DummyClassifier(strategy='uniform'),
#           'SGD': SGDClassifier(loss='log'),
#           'RF': RandomForestClassifier(),
#           'DT': DecisionTreeClassifier(),
#           'AB': AdaBoostClassifier(),
#           'KNN': KNeighborsClassifier(),
#           'SVM': SVC(),
#           'LR': LogisticRegression()
#           }
steps = [('SVD', TruncatedSVD()), ('SVM', SVC())]
pipeline = Pipeline(steps)
param_grid = {'SVD__n_components': [50, 100, 500, 1000, 1500, 2000, 2500],
              'SVM__C': [0.1, 1, 10, 100, 1000],
              'SVM__gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'scale'],
              'SVM__kernel': ['rbf', 'linear']
              }
grid = GridSearchCV(pipeline,
                    param_grid,
                    verbose=3,
                    cv=5,
                    scoring='f1_macro'
                    )
grid.fit(x_train, y_train)
svm = SVC()
svm.fit(x_train, y_train)
print(score_models({'Grid': grid.best_estimator_, 'Default': svm}))
print(grid.best_estimator_)
dump(grid.best_estimator_, r'D:\Python\FatAcceptance\svm.joblib')
