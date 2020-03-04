import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump, load
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# io
import InputOutput as io


def score_models(models):
    name = []
    accuracy = []
    f1 = []
    for k, v in models.items():
        name.append(k)
        v.fit(X_train, y_train)
        pred = v.predict(X_test)
        f1.append(f1_score(y_true=y_test, y_pred=pred, average='macro'))
        accuracy.append(accuracy_score(y_true=y_test, y_pred=pred))
    compare = pd.DataFrame([name, accuracy, f1])
    compare = compare.T
    compare.columns = ['Name', 'Accuracy', 'F1 (macro)']
    compare = compare.sort_values(by='F1 (macro)')
    return compare


def select_models():
    models = {'Stratified': DummyClassifier(strategy='stratified'),
              'Frequent': DummyClassifier(strategy='most_frequent'),
              'Prior': DummyClassifier(strategy='prior'),
              'Uniform': DummyClassifier(strategy='uniform'),
              'SGD': SGDClassifier(loss='log'),
              'RF': RandomForestClassifier(),
              'DT': DecisionTreeClassifier(),
              'AB': AdaBoostClassifier(),
              'KNN': KNeighborsClassifier(),
              'SVM (RBF)': SVC(),
              'LR': LogisticRegression()
              }
    score_models(models).to_csv(
        r'D:\Python\NLP\FatAcceptance\Overall\Models.csv', index=False)


def select_hyperparameters():
    param_grid = {
              'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'scale'],
              'kernel': ['rbf', 'linear']
              }
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    grid = GridSearchCV(SVC(),
                        param_grid,
                        verbose=3,
                        n_jobs=-1,
                        cv=cv,
                        scoring='f1_macro',
                        refit=True
                        )
    grid.fit(X_train, y_train)
    print(grid.best_estimator_)
    dump(grid.best_estimator_, r'D:\Python\NLP\FatAcceptance\model.joblib')
    print(grid.best_score_)
    print()
    score()


def score():
    estimator = load(r'D:\Python\NLP\FatAcceptance\model.joblib')
    pred = estimator.predict(X_test)
    print(f1_score(y_true=y_test, y_pred=pred, average='macro'))
    print(accuracy_score(y_true=y_test, y_pred=pred))
    conf_mat = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(conf_mat, annot=True, fmt='d')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


np.random.seed(1000)

path = Path(r'D:/Python/NLP/FatAcceptance/Training/Final/ULMFiT')
train = pd.read_csv(path / 'train.csv', encoding='utf-8')
val = pd.read_csv(path / 'val.csv', encoding='utf-8')
test = pd.read_csv(path / 'test.csv', encoding='utf-8')
train = pd.concat([train, val])
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=1, encoding='utf-8', ngram_range=(1, 2), lowercase=True)
X_train = tfidf.fit_transform(train['text'])
y_train = train['label']
X_test = tfidf.transform(test['text'])
y_test = test['label']
print(X_train.shape)
print(X_test.shape)
select_models()
# select_hyperparameters()
