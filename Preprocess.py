import re
from pathlib import Path

import pandas as pd


def remove_non_ascii(text): return ''.join(i for i in text if ord(i) < 128)

def remove_unicode(text):
    """ Removes unicode strings like "\u002c" and "x96" """
    text = re.sub(r'(\\u[0-9A-Fa-f]+)',r'', text)       
    text = re.sub(r'[^\x00-\x7f]',r'',text)
    return text

def replace_at_user(text):
    """ Replaces "@user" with "atUser" """
    text = re.sub('@[^\s]+','atUser',text)
    return text

def remove_hashtag_in_front(text):
    """ Removes hastag in front of a word """
    text = re.sub(r'#([^\s]+)', r'\1', text)
    return text

""" Replaces contractions from a string to their equivalents """
contraction_patterns = [ (r'won\'t', 'will not'), (r'can\'t', 'cannot'), (r'i\'m', 'i am'), (r'ain\'t', 'is not'), (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)n\'t', '\g<1> not'), (r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'s', '\g<1> is'), (r'(\w+)\'re', '\g<1> are'), (r'(\w+)\'d', '\g<1> would'), (r'&', 'and'), (r'dammit', 'damn it'), (r'dont', 'do not'), (r'wont', 'will not') ]
def replace_contractions(text):
    patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]
    for (pattern, repl) in patterns:
        (text, count) = re.subn(pattern, repl, text)
    return text

def remove_numbers(text):
    """ Removes integers """
    text = ''.join([i for i in text if not i.isdigit()])         
    return text

def replace_multi_exclamation(text):
    """ Replaces repetitions of exlamation marks """
    text = re.sub(r"(\!)\1+", ' multiExclamation ', text)
    return text

def replace_multi_question(text):
    """ Replaces repetitions of question marks """
    text = re.sub(r"(\?)\1+", ' multiQuestion ', text)
    return text

def replace_multi_stop(text):
    """ Replaces repetitions of stop marks """
    text = re.sub(r"(\.)\1+", ' multiStop ', text)
    return text

def preprocess_df(df):
    df['text'] = df['text'].apply(remove_non_ascii)
    df['text'] = df['text'].apply(remove_unicode)
    df['text'] = df['text'].apply(replace_at_user)
    df['text'] = df['text'].apply(remove_hashtag_in_front)
    df['text'] = df['text'].apply(replace_contractions)
    df['text'] = df['text'].apply(remove_numbers)
    df['text'] = df['text'].apply(replace_multi_exclamation)
    df['text'] = df['text'].apply(replace_multi_question)
    df['text'] = df['text'].apply(replace_multi_stop)

path = Path(r'D:/Python/NLP/FatAcceptance/Training/Final/ULMFiT')
train = pd.read_csv(path / 'train.csv', encoding='utf-8')
val = pd.read_csv(path / 'val.csv', encoding='utf-8')
test = pd.read_csv(path / 'test.csv', encoding='utf-8')
preprocess_df(train)
preprocess_df(val)
preprocess_df(test)
path_new = Path(r'D:/Python/NLP/FatAcceptance/Training/Final/Traditional')
train.to_csv(path_new / 'train.csv', encoding='utf-8', index=False)
val.to_csv(path_new / 'val.csv', encoding='utf-8', index=False)
test.to_csv(path_new / 'test.csv', encoding='utf-8', index=False)
