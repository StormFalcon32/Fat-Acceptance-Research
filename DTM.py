import pickle
import gensim
import InputOutput as io

data_words = io.csvIn(r'Overall\TextsDup.csv', False)
id2word = gensim.corpora.Dictionary(data_words)
pickle.dump(id2word.token2id, open(
    r'D:\Python\NLP\FatAcceptance\vocab.pkl', 'wb'))
