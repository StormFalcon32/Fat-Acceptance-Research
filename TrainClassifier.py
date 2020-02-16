import statistics
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

# io
import InputOutput as io
import pickle
from joblib import dump, load


def debugScore(prediction, labels, X_test):
    inv = {v: k for k, v in tfidf.vocabulary_.items()}
    test_mat = sparse.coo_matrix(X_test)
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
    io.csvOut(r'Training\TestWords.csv', None, words)
    io.csvOut(r'Training\TestLabels.csv', None, labels)


def score_models(models):
    name = []
    accuracy = []
    f1 = []
    k_fold = StratifiedKFold(n_splits=5, shuffle=True)
    for k, v in models.items():
        name.append(k)
        accuracy.append(statistics.mean(
            cross_val_score(v, X_train, y_train, cv=k_fold, scoring='accuracy')))
        f1.append(statistics.mean(
            cross_val_score(v, X_train, y_train, cv=k_fold, scoring='f1_macro')))
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
        'alpha': [1e-4, 1e-3, 1e-2]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    grid = GridSearchCV(SGDClassifier(),
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


np.random.seed(100)

data_words = io.csvIn(r'Training\Final\TrainingTexts.csv', False)
labels = io.csvInSingle(r'Training\Final\Labels.csv')
vocab = pickle.load(open(r'D:\Python\NLP\FatAcceptance\vocab.pkl', 'rb'))
tfidf = TfidfVectorizer(preprocessor=lambda x: x,
                        tokenizer=lambda x: x, vocabulary=vocab)
X = tfidf.fit_transform(data_words)
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, stratify=labels)
select_hyperparameters()
