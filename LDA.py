import gensim
import csv
import logging
from gensim.test.utils import datapath
import pickle


def main(start, end, increment):
    # configure logging
    logging.basicConfig(filename='lda_model.log',
                        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    data_words = []
    with open(r'D:\Python\GamingDisorder\Texts.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            data_words.append(row)
    # create dictionary, corpus, and tdm
    id2word = gensim.corpora.Dictionary(data_words)
    texts = data_words
    corpus = [id2word.doc2bow(text) for text in texts]
    # view corpus
    # print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])
    for k in range(start, end, increment):
        # train model
        lda_model = gensim.models.ldamulticore.LdaMulticore(
            corpus=corpus, id2word=id2word, num_topics=k, passes=10, workers=2, per_word_topics=True)
        # save model
        file = datapath(r'D:\Python\GamingDisorder\Models\Model%s' % k)
        lda_model.save(file)
        pickle.dump(corpus, open(
            r'D:\Python\GamingDisorder\Models\Corp%s.pkl' % k, 'wb'))


if __name__ == '__main__':
    main(2, 10, 1)
