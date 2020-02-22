import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fastai.text import *
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

np.random.seed(100)

def clean(s): return ''.join(i for i in s if ord(i) < 128)

def score():
    path = Path(r'D:/Python/NLP/FatAcceptance/Training/Final/')
    learn = load_learner(path / 'models', 'trained_model.pkl')

    test = pd.read_csv(path / 'test.csv', encoding='utf-8')
    predictions = []
    for _, row in test.iterrows():
        predictions.append(learn.predict(row['text'])[0].obj)
    test['pred'] = predictions
    # Output to a text file for comparison with the gold reference
    test.to_csv(path / 'pred.csv', index=False)
    pred = test['pred']
    true = test['final_label']
    print(f1_score(y_true=true, y_pred=pred, average='macro'))
    print(accuracy_score(y_true=true, y_pred=pred))
    conf_mat = confusion_matrix(true, pred)
    plt.subplots(figsize=(10, 10))
    sns.heatmap(conf_mat, annot=True, fmt='d')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(path / 'confusion.jpg', dpi=1000, bbox_inches='tight')

def predict(text):
    path = Path(r'D:/Python/NLP/FatAcceptance/Training/Final/')
    learn = load_learner(path / 'models', 'trained_model.pkl')
    print(learn.predict(text)[0].obj)

def predict_lm(text, n_words):
    path = Path(r'D:/Python/NLP/FatAcceptance/Training/Final/')
    learn = load_learner(path / 'models', 'lm_model.pkl')
    print(learn.predict(text, n_words))

def train_lm(learning_rates=False):
    # file directory
    path = Path(r'D:/Python/NLP/FatAcceptance/Training/Final/')
    # unlabeled set of ~80K tweetes to train unsupervised language model
    data_lm = TextLMDataBunch.from_csv(path, 'unlabeled.csv', min_freq=1, bs=16)
    
    # language model learner
    learn = language_model_learner(data_lm, arch=AWD_LSTM, drop_mult=0.5)
    if learning_rates:
        # graph learning rates
        learn.lr_find(start_lr=1e-8, end_lr=1e2)
        lr_fig_1 = learn.recorder.plot(return_fig=True)
        lr_fig_1.savefig(path / 'lr_fig_1.jpg', dpi=1000, bbox_inches='tight')

    # Gradual unfreezing of lm
    learn.freeze()
    learn.fit_one_cycle(cyc_len=1, max_lr=1e-2, moms=(0.8, 0.7))

    learn.freeze_to(-2)
    learn.fit_one_cycle(cyc_len=1, max_lr=1e-2, moms=(0.8, 0.7))

    learn.freeze_to(-3)
    learn.fit_one_cycle(cyc_len=1, max_lr=1e-2, moms=(0.8, 0.7))

    learn.unfreeze()
    learn.fit_one_cycle(cyc_len=10, max_lr=1e-3, moms=(0.8, 0.7))
    # plot losses
    losses_lm_fig = learn.recorder.plot_losses(return_fig=True)
    losses_lm_fig.savefig(path / 'losses_lm_fig.jpg', dpi=1000, bbox_inches='tight')
    # Save the fine-tuned encoder
    learn.save_encoder('ft_enc')
    learn.export(path / 'models' / 'lm_model.pkl')
    data_lm.save(path / 'models' / 'data_lm.pkl')

    

def train_clas(learning_rates=False):
    # file directory
    path = Path(r'D:/Python/NLP/FatAcceptance/Training/Final/')
    
    # unlabeled set of ~80K tweetes to train unsupervised language model

    # Load labeled data for classifier
    data_lm = load_data(path / 'models', 'data_lm.pkl', bs=16)
    data_clas = TextClasDataBunch.from_csv(path, 'train.csv',
                vocab=data_lm.train_ds.vocab, min_freq=1, bs=32)

    # classifier learner
    learn = text_classifier_learner(data_clas, arch=AWD_LSTM, drop_mult=0.5)
    # load encoder
    learn.load_encoder('ft_enc')
    if learning_rates:
        # graph learning rates
        learn.freeze()
        learn.lr_find(start_lr=1e-8, end_lr=1e2)
        lr_fig_2 = learn.recorder.plot(return_fig=True)
        lr_fig_2.savefig(path / 'lr_fig_2.jpg', dpi=1000, bbox_inches='tight')

    # gradual unfreezing
    learn.freeze()
    learn.fit_one_cycle(cyc_len=1, max_lr=1e-3, moms=(0.8, 0.7))

    learn.freeze_to(-2)
    learn.fit_one_cycle(1, slice(1e-4,1e-2), moms=(0.8,0.7))

    learn.freeze_to(-3)
    learn.fit_one_cycle(1, slice(1e-5,5e-3), moms=(0.8,0.7))

    learn.unfreeze()
    learn.fit_one_cycle(8, slice(1e-5,1e-3), moms=(0.8,0.7))
    # plot losses
    losses_clas_fig = learn.recorder.plot_losses(return_fig=True)
    losses_clas_fig.savefig(path / 'losses_clas_fig.jpg', dpi=1000, bbox_inches='tight')
    learn.export(path / 'models' / 'trained_model.pkl')


def load_files(first_time=False):
    path = Path(r'D:/Python/NLP/FatAcceptance/Training/Final/')
    path_overall = Path(r'D:/Python/NLP/FatAcceptance/Overall/')
    labeled_data = pd.read_csv(path / '1000Selected0Final.csv', encoding='utf-8')
    unlabeled_data = pd.read_csv(path_overall / 'NoDups.csv', encoding='utf-8')
    if first_time:
        for i, row in labeled_data.iterrows():
            if math.isnan(row['final_label']):
                labeled_data.at[i,'final_label'] = row['label1']
            else:
                labeled_data.at[i,'final_label'] = row['final_label']
        labeled_data.final_label = labeled_data.final_label.astype(int)
        labeled_data.to_csv(path / '1000Selected0Final.csv', index=False)
    unlabeled_data['label'] = 0
    unlabeled_data = pd.concat([unlabeled_data['label'], unlabeled_data['text'].apply(clean)], axis=1)
    data = pd.concat([labeled_data['final_label'], labeled_data['text'].apply(clean)], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['final_label'], test_size=0.2, stratify=data['final_label'])
    train = pd.concat([y_train, X_train], axis=1)
    test = pd.concat([y_test, X_test], axis=1)
    test['pred'] = ''
    train.to_csv(path / 'train.csv', index=False)
    test.to_csv(path / 'test.csv', index=False)
    unlabeled_data.to_csv(path / 'unlabeled.csv', index=False)


if __name__ == '__main__':
    train_lm(learning_rates=True)
    train_clas(learning_rates=True)
    score()
    predict_lm('fat acceptance is the only movement', 2)
    predict("Dear #fatshaming #trolls ... Let me save you some time. I KNOW I'M FAT. I'm good with it. #effyourbeautystandards #fatacceptance #JustSayingpic.twitter.com/LIyshWsrLG")
