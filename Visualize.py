import gensim
from gensim.test.utils import datapath
import csv
import pickle
import matplotlib.pyplot as plt


def main(start, end, increment):
    coherence_values_cv = []
    coherence_values_umass = []
    for k in range(start, end, increment):
        # load model
        file = datapath(r'D:\Python\GamingDisorder\Models\Model%s' % k)
        lda_model = gensim.models.ldamodel.LdaModel.load(file)
        data_words = []
        with open(r'D:\Python\GamingDisorder\Texts.csv') as f:
            reader = csv.reader(f)
            for row in reader:
                data_words.append(row)
        corpus = pickle.load(
            open(r'D:\Python\GamingDisorder\Models\Corp%s.pkl' % k, 'rb'))
        # coherence score
        coherence_model_cv = gensim.models.coherencemodel.CoherenceModel(
            model=lda_model, texts=data_words, dictionary=lda_model.id2word, coherence='c_v')
        coherence_model_umass = gensim.models.coherencemodel.CoherenceModel(
            model=lda_model, corpus=corpus, dictionary=lda_model.id2word, coherence='u_mass')
        coherence_cv = coherence_model_cv.get_coherence()
        coherence_umass = coherence_model_umass.get_coherence()
        coherence_values_cv.append(coherence_cv)
        coherence_values_umass.append(coherence_umass)

    x = range(start, end, increment)
    plt.figure(1)
    plt.plot(x, coherence_values_cv)
    plt.xlabel('Num Topics')
    plt.ylabel('Coherence score')
    plt.legend(('coherence_values'), loc='best')
    plt.figure(2)
    plt.plot(x, coherence_values_umass)
    plt.xlabel('Num Topics')
    plt.ylabel('Coherence score')
    plt.legend(('coherence_values'), loc='best')
    plt.show()
    plt.show()


if __name__ == '__main__':
    main(2, 10, 1)
