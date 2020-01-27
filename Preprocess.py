from nltk.corpus import stopwords
import re
import string
import csv
import pandas as pd
import gensim
import spacy
import math
import InputOutput as io


def clean(text):
    text = text.lower()
    # remove URLs
    text = re.sub(
        r'([\d\w]+?:\/\/)?([\w\d\.\-]+)(\.\w+)(:\d{1,5})?(\/\S*)?', ' ', text)
    # remove non ascii characters
    text = remove_non_ascii(text)
    # remove leading and trailing whitespace and merge extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_non_ascii(s): return ''.join(i for i in s if ord(i) < 128)


def to_words(sentences):
    # tokenize and remove punctuation
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(sentence, deacc=True))


def remove_stopwords(tokenized_sentences, stop_words):
    for sentence in tokenized_sentences:
        filtered = [w for w in sentence if not w in stop_words]
        yield(filtered)


def lemmatize(tokenized_sentences, nlp):
    for sentence in tokenized_sentences:
        doc = nlp(' '.join(sentence))
        lemmatized = [w.lemma_ for w in doc]
        yield(lemmatized)


def main(entire):
    # nlp = spacy.load('en', disable=['parser', 'ner'])
    # stop_words = set(stopwords.words('english'))
    csvIn = None
    fileDir = None
    if entire:
        fileDir = r'D:\Python\FatAcceptance\Overall\NoDups.csv'
    else:
        fileDir = r'D:\Python\FatAcceptance\Training\Final\1000Selected0Final.csv'
    csvIn = pd.read_csv(fileDir, encoding='utf-8')
    df = csvIn.to_dict('index')
    new = []
    dupretweets = []
    labels = []
    uncleaned = []
    for i in range(0, len(df)):
        num_retweets = df[i]['retweets']
        cleaned = clean(df[i]['text'])
        for _ in range(0, int(num_retweets + 1)):
            dupretweets.append(cleaned)
            uncleaned.append([df[i]['date'], df[i]['text']])
        new.append(cleaned)
        if not entire:
            if not math.isnan(df[i]['final_label']):
                labels.append(int(df[i]['final_label']))
            else:
                labels.append(df[i]['label1'])
    # tokenize, remove stopwords, and lemmatize
    if not entire:
        data_words = list(to_words(new))
        # data_words = list(remove_stopwords(data_words, stop_words))
        # data_words = list(lemmatize(data_words, nlp))
        io.csvOut(r'Training\Final\Lemmatized.csv', cols=None, data=data_words)
        io.csvOutSingle(r'Training\Final\Labels.csv', labels)
    else:
        data_words = list(to_words(dupretweets))
        # data_words = list(remove_stopwords(data_words, stop_words))
        # data_words = list(lemmatize(data_words, nlp))
        io.csvOut(r'Overall\LemmatizedDup.csv',
                  cols=None, data=data_words)
        io.csvOut(r'Overall\WithRetweets.csv', cols=[
                  'date', 'text'], data=uncleaned)


if __name__ == '__main__':
    main(False)
