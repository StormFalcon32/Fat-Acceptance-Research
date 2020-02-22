import gensim
import csv
import InputOutput as io


def make_bigrams(tokenized_sentences, bigram_mod):
    for sentence in tokenized_sentences:
        yield(bigram_mod[sentence])


def main():
    data_words = io.csvIn(r'Training\Final\Lemmatized.csv', skip_first=False)
    overall = io.csvIn(r'Overall\LemmatizedDup.csv', skip_first=False)
    # create bigrams
    bigram = gensim.models.Phrases(
        data_words, scoring='npmi', threshold=0.7)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    data_words = list(make_bigrams(data_words, bigram_mod))
    overall = list(make_bigrams(overall, bigram_mod))
    io.csvOut(r'Training\Final\TrainingTexts.csv', cols=None, data=data_words)
    io.csvOut(r'Overall\TextsDup.csv', cols=None, data=overall)


if __name__ == '__main__':
    main()
