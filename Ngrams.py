import gensim
import csv


def make_trigrams(tokenized_sentences, trigram_mod, bigram_mod):
    for sentence in tokenized_sentences:
        yield(trigram_mod[bigram_mod[sentence]])


def main():
    data_words = []
    with open(r'D:\Python\FatAcceptance\Lemmatized.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            data_words.append(row)
    # create bigrams and trigrams
    bigram = gensim.models.Phrases(
        data_words, scoring='npmi', min_count=10000, threshold=0.2)
    trigram = gensim.models.Phrases(
        bigram[data_words], scoring='npmi', min_count=1000, threshold=0.2)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    data_words = list(make_trigrams(data_words, trigram_mod, bigram_mod))
    with open(r'D:\Python\FatAcceptance\Texts.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data_words)


if __name__ == '__main__':
    main()
