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
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(conf_mat, annot=True, fmt='d')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(path / 'confusion.jpg', dpi=1000, bbox_inches='tight')

def predict(text):
    path = Path(r'D:/Python/NLP/FatAcceptance/Training/Final/')
    learn = load_learner(path / 'models', 'trained_model.pkl')
    print(learn.predict(text)[0].obj)

def predict_LM(text, n_words):
    path = Path(r'D:/Python/NLP/FatAcceptance/Training/Final/')
    learn = load_learner(path / 'models', 'LM_model.pkl')
    print(learn.predict(text, n_words))

def train_model(learning_rates=False):
    # file directory
    path = Path(r'D:/Python/NLP/FatAcceptance/Training/Final/')
    # unlabeled set of ~80K tweetes to train unsupervised language model
    data_lm = TextLMDataBunch.from_csv(path, 'unlabeled.csv', min_freq=1, bs=16)
    
    # language model learner
    learn = language_model_learner(data_lm, arch=AWD_LSTM, drop_mult=0.5)
    if learning_rates:
        # graph learning rates
        learn.lr_find(start_lr=1e-8, end_lr=1e2)
        fig1 = learn.recorder.plot(return_fig=True)
        fig1.savefig(path / 'fig1.jpg', dpi=1000, bbox_inches='tight')

    # Gradual unfreezing of LM
    learn.freeze()
    learn.fit_one_cycle(cyc_len=1, max_lr=1e-2, moms=(0.8, 0.7))

    learn.freeze_to(-2)
    learn.fit_one_cycle(cyc_len=1, max_lr=1e-2, moms=(0.8, 0.7))

    learn.freeze_to(-3)
    learn.fit_one_cycle(cyc_len=1, max_lr=1e-2, moms=(0.8, 0.7))

    learn.unfreeze()
    learn.fit_one_cycle(cyc_len=8, max_lr=1e-3, moms=(0.8, 0.7))
    # Save the fine-tuned encoder
    learn.save_encoder('ft_enc')
    learn.export(path / 'models' / 'LM_model.pkl')

    # Load labeled data for classifier
    data_clas = TextClasDataBunch.from_csv(path, 'train.csv',
                vocab=data_lm.train_ds.vocab, min_freq=1, bs=32)

    # classifier learner
    learn = text_classifier_learner(data_clas, arch=AWD_LSTM, drop_mult=0.5)
    if learning_rates:
        # graph learning rates
        learn.load_encoder('ft_enc')
        learn.freeze()
        learn.lr_find(start_lr=1e-8, end_lr=1e2)
        fig2 = learn.recorder.plot(return_fig=True)
        fig2.savefig(path / 'fig2.jpg', dpi=1000, bbox_inches='tight')

    # load encoder
    learn.load_encoder('ft_enc')
    # gradual unfreezing
    learn.freeze()
    learn.fit_one_cycle(cyc_len=1, max_lr=1e-3, moms=(0.8, 0.7))

    learn.freeze_to(-2)
    learn.fit_one_cycle(1, slice(1e-4,1e-2), moms=(0.8,0.7))

    learn.freeze_to(-3)
    learn.fit_one_cycle(1, slice(1e-5,5e-3), moms=(0.8,0.7))

    learn.unfreeze()
    learn.fit_one_cycle(8, slice(1e-5,1e-3), moms=(0.8,0.7))
    
    learn.export(path / 'models' / 'trained_model.pkl')


def load_data(first_time=False):
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
    # train_model()
    # score()
    predict_LM('fat acceptance is the only movement', 2)
    predict("Dear #fatshaming #trolls ... Let me save you some time. I KNOW I'M FAT. I'm good with it. #effyourbeautystandards #fatacceptance #JustSayingpic.twitter.com/LIyshWsrLG")
