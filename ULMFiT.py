import math
import statistics
import warnings
import webbrowser
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fastai import callbacks
from fastai.text import *
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

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

def random_seed(seed_value=100000):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True


def clean(s): return ''.join(i for i in s if ord(i) < 128)

def score(test):
    path = Path(r'D:/Python/NLP/FatAcceptance/Training/Final/ULMFiT')
    learn = load_learner(path / 'models', 'trained_model.pkl')
    learn.data.add_test(test['text'])
    predictions = learn.get_preds(ds_type=DatasetType.Test)[0].argmax(dim=1)
    test['pred'] = predictions
    # Output to a text file for comparison with the gold reference
    test.to_csv(path / 'pred.csv', index=False)
    pred = test['pred']
    true = test['label']
    print(accuracy_score(y_true=true, y_pred=pred))
    print(f1_score(y_true=true, y_pred=pred, average='macro'))
    print(precision_score(y_true=true, y_pred=pred, average='macro'))
    print(recall_score(y_true=true, y_pred=pred, average='macro'))
    conf_mat = confusion_matrix(true, pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(conf_mat, annot=True, fmt='d')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    ax.xaxis.set_ticklabels(['Support', 'Oppose', 'Unclear'])
    ax.yaxis.set_ticklabels(['Support', 'Oppose', 'Unclear'])
    plt.savefig(path / 'figs' / 'confusion.jpg', dpi=1000, bbox_inches='tight')

def create_bootstrap(data):
    path = Path(r'D:/Python/NLP/FatAcceptance/Training/Final/ULMFiT')
    n_iterations = 60
    train_batched = pd.DataFrame()
    test_batched = pd.DataFrame()
    # run bootstrap
    for i in range(n_iterations):
        X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.15, stratify=data['label'])
        train_batched = pd.concat([train_batched, pd.concat([y_train, X_train], axis=1)])
        test_batched = pd.concat([test_batched, pd.concat([y_test, X_test], axis=1)])
    train_batched.to_csv(path / 'trainbootstrap.csv', index=False)
    test_batched.to_csv(path / 'testbootstrap.csv', index=False)


def calc_bootstrap(start=0, first_time=True):
    path = Path(r'D:/Python/NLP/FatAcceptance/Training/Final/ULMFiT')
    if first_time:
        with open(path / 'resample.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['pred', 'label'])
    n_iterations = 60
    train_batched = pd.read_csv(path / 'trainbootstrap.csv', encoding='utf-8')
    test_batched = pd.read_csv(path / 'testbootstrap.csv', encoding='utf-8')
    # run bootstrap
    for i in range(n_iterations):
        print(i)
        if i < start:
            continue
        train_iter = train_batched.iloc[i * 1700 : (i + 1) * 1700]
        test_iter = test_batched.iloc[i * 300 : (i + 1) * 300]
        data_lm = load_data(path / 'models', 'data_lm.pkl', num_workers=0)
        bs = 8
        data_clas = TextClasDataBunch.from_df(path, train_df=train_iter, valid_df=test_iter, vocab=data_lm.train_ds.vocab, min_freq=1, bs=bs, num_workers=0)
        learn = train_clas(data_clas, False, True)
        learn.data.add_test(test_iter['text'])
        preds = learn.get_preds(ds_type=DatasetType.Test)[0].argmax(dim=1).tolist()
        with open(path / 'resample.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(list(zip(preds, test_iter['label'].tolist())))

def score_bootstrap():
    path = Path(r'C:/Data/Python/NLP/FatAcceptance/Training/Final/ULMFiT')
    n_iterations = 60
    resampled = pd.read_csv(path / 'resample.csv', encoding='utf-8')
    true = resampled['label']
    pred = resampled['pred']
    stats = [[], [], [], []]
    support_stats = [[], [], []]
    oppose_stats = [[], [], []]
    neutral_stats = [[], [], []]
    for i in range(n_iterations):
        true_batch = true[i * 300 : (i + 1) * 300]
        pred_batch = pred[i * 300 : (i + 1) * 300]
        conf_mat = confusion_matrix(true_batch, pred_batch)
        true_pos_support = conf_mat[0][0]
        false_pos_support = conf_mat[1][0] + conf_mat[2][0]
        false_neg_support = conf_mat[0][1] + conf_mat[0][2]
        true_pos_oppose = conf_mat[1][1]
        false_pos_oppose = conf_mat[0][1] + conf_mat[2][1]
        false_neg_oppose = conf_mat[1][0] + conf_mat[1][2]
        true_pos_neutral = conf_mat[2][2]
        false_pos_neutral = conf_mat[0][2] + conf_mat[1][2]
        false_neg_neutral = conf_mat[2][0] + conf_mat[2][1]
        precision_support = true_pos_support / (true_pos_support + false_pos_support)
        recall_support = true_pos_support / (true_pos_support + false_neg_support)
        f1_support = 2 * precision_support * recall_support / (precision_support + recall_support)
        precision_oppose = true_pos_oppose / (true_pos_oppose + false_pos_oppose)
        recall_oppose = true_pos_oppose / (true_pos_oppose + false_neg_oppose)
        f1_oppose = 2 * precision_oppose * recall_oppose / (precision_oppose + recall_oppose)
        precision_neutral = true_pos_neutral / (true_pos_neutral + false_pos_neutral)
        recall_neutral = true_pos_neutral / (true_pos_neutral + false_neg_neutral)
        f1_neutral = 2 * precision_neutral * recall_neutral / (precision_neutral + recall_neutral)
        stats[0].append(accuracy_score(y_true=true_batch, y_pred=pred_batch))
        stats[1].append((precision_support + precision_oppose + precision_neutral) / 3)
        stats[2].append((recall_support + recall_oppose + recall_neutral) / 3)
        stats[3].append((f1_support + f1_oppose + f1_neutral) / 3)
        support_stats[0].append(precision_support)
        support_stats[1].append(recall_support)
        support_stats[2].append(f1_support)
        oppose_stats[0].append(precision_oppose)
        oppose_stats[1].append(recall_oppose)
        oppose_stats[2].append(f1_oppose)
        neutral_stats[0].append(precision_neutral)
        neutral_stats[1].append(recall_neutral)
        neutral_stats[2].append(f1_neutral)
    for i in stats:
        plot_distribution(i)
    for i in support_stats:
        plot_distribution(i)
    for i in oppose_stats:
        plot_distribution(i)
    for i in neutral_stats:
        plot_distribution(i)
    

def plot_distribution(i):
    plt.hist(i)
    plt.show()
    # confidence intervals
    alpha = 0.95
    p = ((1 - alpha) / 2) * 100
    lower = max(0, np.percentile(i, p))
    p = (alpha + ((1 - alpha) / 2)) * 100
    upper = min(1, np.percentile(i, p))
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha * 100, lower * 100, upper * 100))
    print(statistics.mean(i))

def use():
    path = Path(r'D:/Python/NLP/FatAcceptance/Training/Final/ULMFiT')
    path_overall = Path(r'D:/Python/NLP/FatAcceptance/Overall')
    learn = load_learner(path / 'models', 'trained_model.pkl')
    test = pd.read_csv(path_overall / 'WithRetweets.csv', encoding='utf-8')
    learn.data.add_test(test['text'])
    predictions = learn.get_preds(ds_type=DatasetType.Test)[0].argmax(dim=1)
    test['pred'] = predictions
    test.to_csv(path_overall / 'WithRetweets.csv', encoding='utf-8', index=False)

def predict(text):
    random_seed()
    path = Path(r'D:/Python/NLP/FatAcceptance/Training/Final/ULMFiT')
    learn = load_learner(path / 'models', 'trained_model.pkl')
    test = pd.DataFrame([text], columns=['text'])
    learn.data.add_test(test['text'])
    preds, y, losses = learn.get_preds(ds_type=DatasetType.Test, with_loss=True)
    classes = ['Support', 'Oppose', 'Unclear']
    print(classes[preds.argmax(dim=1)[0].tolist()])
    interp = TextClassificationInterpretation(learn, preds, y, losses)
    html = interp.html_intrinsic_attention(text)
    path = os.path.abspath(path / 'interp.html')
    url = 'file://' + path
    with open(path, 'w') as f:
        f.write(html)
    webbrowser.open(url)

def predict_lm(text, n_words):
    path = Path(r'D:/Python/NLP/FatAcceptance/Training/Final/ULMFiT')
    learn = load_learner(path / 'models', 'lm_model.pkl')
    
    print(learn.predict(text, n_words))

def train_lm(learning_rates=False):
    random_seed()
    # file directory
    path = Path(r'D:/Python/NLP/FatAcceptance/Training/Final/ULMFiT')
    # unlabeled set of ~40K tweetes to train unsupervised language model
    data_lm = TextLMDataBunch.from_csv(path, 'unlabeled.csv', min_freq=1, bs=8, num_workers=0)
    # language model learner
    learn = language_model_learner(data_lm, arch=AWD_LSTM, drop_mult=0.4, wd=0.1, metrics=[accuracy], pretrained=True)
    random_seed()
    learn.freeze()
    if learning_rates:
        # graph learning rates
        learn.lr_find(start_lr=1e-8, end_lr=1e2)
        lr_fig_1 = learn.recorder.plot(return_fig=True, suggestion=True)
        lr_fig_1.savefig(path / 'figs' / 'lr_fig_1.jpg', dpi=1000, bbox_inches='tight')
    print(learn.loss_func)
        # Gradual unfreezing of lm
    learn.fit_one_cycle(cyc_len=1, max_lr=1e-2, moms=(0.8, 0.7))

    learn.unfreeze()
    learn.fit_one_cycle(cyc_len=10, max_lr=1e-3, moms=(0.8, 0.7), callbacks=[callbacks.SaveModelCallback(learn, monitor='valid_loss', name='lm_model')])
    # plot losses
    losses_lm_fig = learn.recorder.plot_losses(return_fig=True)
    losses_lm_fig.savefig(path / 'figs' / 'losses_lm_fig.jpg', dpi=1000, bbox_inches='tight')
    # Save the fine-tuned encoder
    learn.save_encoder('ft_enc')
    learn.export(path / 'models' / 'lm_model.pkl')
    data_lm.save(path / 'models' / 'data_lm.pkl')

def train_clas(data_clas, learning_rates=False, bootstrap=False):
    random_seed()
    # file directory
    path = Path(r'D:/Python/NLP/FatAcceptance/Training/Final/ULMFiT')
    # print(data_clas)
    # classifier learner
    learn = text_classifier_learner(data_clas, arch=AWD_LSTM, drop_mult=0.9, wd=0.1, metrics=[accuracy, F1()], pretrained=True)
    print(learn.loss_func)
    random_seed()
    # load encoder
    learn.load_encoder('ft_enc')
    learn.freeze()
    if learning_rates:
        # graph learning rates
        learn.lr_find(start_lr=1e-8, end_lr=1e2)
        lr_fig_2 = learn.recorder.plot(return_fig=True, suggestion=True)
        lr_fig_2.savefig(path / 'figs' / 'lr_fig_2.jpg', dpi=1000, bbox_inches='tight')
    # gradual unfreezing
    learn.fit_one_cycle(2)

    learn.freeze_to(-2)
    learn.fit_one_cycle(2)

    learn.freeze_to(-3)
    learn.fit_one_cycle(2)

    learn.unfreeze()
    learn.fit_one_cycle(2)
    if not bootstrap:
        # plot losses
        losses_clas_fig = learn.recorder.plot_losses(return_fig=True)
        losses_clas_fig.savefig(path  / 'figs' / 'losses_clas_fig.jpg', dpi=1000, bbox_inches='tight')
        learn.export(path / 'models' / 'trained_model.pkl')
    return learn


def load_files():
    path = Path(r'D:/Python/NLP/FatAcceptance/Training/Final/')
    path_overall = Path(r'D:/Python/NLP/FatAcceptance/Overall/')
    labeled_data = pd.read_csv(path / 'Labeled.csv', encoding='utf-8')
    unlabeled_data = pd.read_csv(path_overall / 'WithoutRetweets.csv', encoding='utf-8')
    unlabeled_data['label'] = 0
    unlabeled_data = pd.concat([unlabeled_data['label'], unlabeled_data['text'].apply(clean)], axis=1)
    data = pd.concat([labeled_data['label'], labeled_data['text'].apply(clean)], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.3, stratify=data['label'])
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test)
    train = pd.concat([y_train, X_train], axis=1)
    test = pd.concat([y_test, X_test], axis=1)
    val = pd.concat([y_val, X_val], axis=1)
    test['pred'] = ''
    train.to_csv(path / 'ULMFiT' / 'train.csv', index=False)
    test.to_csv(path /  'ULMFiT' / 'test.csv', index=False)
    val.to_csv(path /  'ULMFiT' / 'val.csv', index=False)
    unlabeled_data.to_csv(path /  'ULMFiT' / 'unlabeled.csv', index=False)
    print(len(train))
    print(len(unlabeled_data))


if __name__ == '__main__':
    path = Path(r'D:/Python/NLP/FatAcceptance/Training/Final/ULMFiT')
    random_seed()
    # load_files()
    # train_lm(False)
    # train_orig = pd.read_csv(path / 'train.csv', encoding='utf-8')
    # val_orig = pd.read_csv(path / 'val.csv', encoding='utf-8')
    # test_orig = pd.read_csv(path / 'test.csv', encoding='utf-8')
    # train_combined = pd.concat([train_orig, val_orig])
    # data_lm = load_data(path / 'models', 'data_lm.pkl', num_workers=0)
    # bs = 8
    # data_clas = TextClasDataBunch.from_df(path, train_df=train_combined, valid_df=test_orig, vocab=data_lm.train_ds.vocab, min_freq=1, bs=bs, num_workers=0)
    # train_clas(data_clas)
    # score(test_orig)
    # train_combined = pd.concat([train_orig, val_orig, test_orig])
    # create_bootstrap(train_combined)
    # calc_bootstrap()
    score_bootstrap()
    # use()
