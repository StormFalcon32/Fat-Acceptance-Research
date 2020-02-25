import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fastai import callbacks
from fastai.text import *
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore', message='Tensor is int32: upgrading to int64; for better performance use int64 input')

@dataclass
class F1(Callback):
    def __init__(self):
        super().__init__()
        self.name = 'f1'
    
    def on_epoch_begin(self, **kwargs):
        self.y_pred = torch.tensor([]).cuda()
        self.y_true = torch.tensor([]).cuda()

    def on_batch_end(self, last_output, last_target, **kwargs):
        self.y_pred = torch.cat((self.y_pred, last_output.argmax(dim=1).float()))
        self.y_true = torch.cat((self.y_true, last_target.float()))

    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, f1_score(self.y_true.cpu(), self.y_pred.cpu(), average='macro'))

def random_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True


def clean(s): return ''.join(i for i in s if ord(i) < 128)

def check_model():
    path = Path(r'D:/Python/NLP/FatAcceptance/Training/Final/ULMFiT')
    learn = load_learner(path / 'models', 'trained_model.pkl')
    print(learn)

def score():
    path = Path(r'D:/Python/NLP/FatAcceptance/Training/Final/ULMFiT')
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
    path = Path(r'D:/Python/NLP/FatAcceptance/Training/Final/ULMFiT')
    learn = load_learner(path / 'models', 'trained_model.pkl')
    print(learn.predict(text)[0].obj)

def predict_lm(text, n_words):
    path = Path(r'D:/Python/NLP/FatAcceptance/Training/Final/ULMFiT')
    learn = load_learner(path / 'models', 'lm_model.pkl')
    print(learn.predict(text, n_words))

def train_lm(learning_rates=False):
    # file directory
    path = Path(r'D:/Python/NLP/FatAcceptance/Training/Final/ULMFiT')
    # unlabeled set of ~80K tweetes to train unsupervised language model
    data_lm = TextLMDataBunch.from_csv(path, 'unlabeled.csv', min_freq=1, bs=16, num_workers=0)
    # language model learner
    learn = language_model_learner(data_lm, arch=AWD_LSTM, drop_mult=0.7, wd=0.1, metrics=[accuracy], pretrained=True)
    learn.freeze()
    if learning_rates:
        # graph learning rates
        learn.lr_find(start_lr=1e-8, end_lr=1e2)
        lr_fig_1 = learn.recorder.plot(return_fig=True, suggestion=True)
        lr_fig_1.savefig(path / 'figs' / 'lr_fig_1.jpg', dpi=1000, bbox_inches='tight')
    print(learn.loss_func)
    # Gradual unfreezing of lm
    learn.fit_one_cycle(cyc_len=1, max_lr=1e-3, moms=(0.8, 0.7))

    learn.unfreeze()
    learn.fit_one_cycle(cyc_len=8, max_lr=1e-3, moms=(0.8, 0.7), callbacks=[callbacks.SaveModelCallback(learn, monitor='valid_loss', name='lm_model')])
    # plot losses
    losses_lm_fig = learn.recorder.plot_losses(return_fig=True)
    losses_lm_fig.savefig(path / 'figs' / 'losses_lm_fig.jpg', dpi=1000, bbox_inches='tight')
    # Save the fine-tuned encoder
    learn.save_encoder('ft_enc')
    learn.export(path / 'models' / 'lm_model.pkl')
    data_lm.save(path / 'models' / 'data_lm.pkl')


def train_clas(learning_rates=False):
    # file directory
    path = Path(r'D:/Python/NLP/FatAcceptance/Training/Final/ULMFiT')
    
    # unlabeled set of ~80K tweets to train unsupervised language model

    # Load labeled data for classifier
    data_lm = load_data(path / 'models', 'data_lm.pkl', num_workers=0)
    bs = 16
    train = pd.read_csv(path / 'train.csv', encoding='utf-8')
    val = pd.read_csv(path / 'val.csv', encoding='utf-8')
    data_clas = TextClasDataBunch.from_df(path, train_df=train, valid_df=val, vocab=data_lm.train_ds.vocab, min_freq=1, bs=bs, num_workers=0)
    # classifier learner
    learn = text_classifier_learner(data_clas, arch=AWD_LSTM, drop_mult=0.7, wd=0.1, metrics=[accuracy, F1()], pretrained=True)
    print(learn.loss_func)
    print(bs)
    # load encoder
    learn.load_encoder('ft_enc')
    learn.freeze()
    if learning_rates:
        # graph learning rates
        learn.lr_find(start_lr=1e-8, end_lr=1e2)
        lr_fig_2 = learn.recorder.plot(return_fig=True, suggestion=True)
        lr_fig_2.savefig(path / 'figs' / 'lr_fig_2.jpg', dpi=1000, bbox_inches='tight')

    # gradual unfreezing
    learn.fit_one_cycle(cyc_len=2, max_lr=1e-3, moms=(0.8, 0.7))

    learn.freeze_to(-2)
    learn.fit_one_cycle(2, slice(1e-2 / (2.6 ** 4), 1e-2), moms=(0.8, 0.7))

    learn.freeze_to(-3)
    learn.fit_one_cycle(2, slice(5e-3 / (2.6 ** 4), 5e-3), moms=(0.8, 0.7))

    learn.unfreeze()
    learn.fit_one_cycle(8, slice(1e-3 / (2.6 ** 4), 1e-3), moms=(0.8, 0.7), callbacks=[callbacks.SaveModelCallback(learn, monitor='f1', name='model')])
    # plot losses
    losses_clas_fig = learn.recorder.plot_losses(return_fig=True)
    losses_clas_fig.savefig(path  / 'figs' / 'losses_clas_fig.jpg', dpi=1000, bbox_inches='tight')
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
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['final_label'], test_size=0.3, stratify=data['final_label'])
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test)
    train = pd.concat([y_train, X_train], axis=1)
    test = pd.concat([y_test, X_test], axis=1)
    val = pd.concat([y_val, X_val], axis=1)
    test['pred'] = ''
    train.to_csv(path / 'ULMFiT' / 'train.csv', index=False)
    test.to_csv(path /  'ULMFiT' / 'test.csv', index=False)
    val.to_csv(path /  'ULMFiT' / 'val.csv', index=False)
    unlabeled_data.to_csv(path /  'ULMFiT' / 'unlabeled.csv', index=False)


if __name__ == '__main__':
    random_seed(100)
    # load_files()
    # train_lm(False)
    train_clas(False)
    # check_model()
    score()
    # predict_lm('fat acceptance is the only movement', 2)
    # predict("Dear #fatshaming #trolls ... Let me save you some time. I KNOW I'M FAT. I'm good with it. #effyourbeautystandards #fatacceptance #JustSayingpic.twitter.com/LIyshWsrLG")
