import matplotlib.pyplot as plt
import numpy as np
from joblib import load
from sklearn.datasets import load_digits
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import ShuffleSplit, learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import InputOutput as io
import pickle


def plot_learning_curve(estimator, title, X, y, axes, ylim, cv):
    train_sizes = np.linspace(.1, 1.0, 5)
    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    io.csvOut(r'Overall\LearningCurve.csv', cols=None,
              data=[train_sizes, train_scores_mean, test_scores_mean])
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1,
                      color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1,
                      color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
              label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
              label="Cross-validation score")
    axes.legend(loc="best")

    return plt


_, axes = plt.subplots(1, 1)

X = io.csvIn(r'Training\Final\TrainingTexts.csv', False)
y = io.csvInSingle(r'Training\Final\Labels.csv')
vocab = pickle.load(open(r'D:\Python\NLP\FatAcceptance\vocab.pkl', 'rb'))

tfidf = TfidfVectorizer(preprocessor=lambda x: x,
                        tokenizer=lambda x: x, vocabulary=vocab)
X = tfidf.fit_transform(X)
title = 'Learning Curves'
estimator = load(r'D:\Python\NLP\FatAcceptance\model.joblib')
plot_learning_curve(estimator, title, X, y,
                    axes=axes, ylim=(0, 1.01), cv=5)
plt.show()
