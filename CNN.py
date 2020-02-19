import tensorflow as tf
from tensorflow import keras
import numpy as np
import re


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


def review_encode(s):
    encoded = [1]
    for word in s:
        if word in word_index:
            encoded.append(word_index[word])
        else:
            encoded.append(2)
    return encoded


# load data
imdb = keras.datasets.imdb
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=88000)
# preprocessing
word_index = imdb.get_word_index()

word_index = {k: (v+3) for k, v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

train = False

if train:

    max_len = 250
    # for review in X_train:
    #     max_len = max(max_len, len(review))
    # for review in X_test:
    #     max_len = max(max_len, len(review))

    X_train = keras.preprocessing.sequence.pad_sequences(X_train, value=word_index['<PAD>'], padding='post', maxlen=max_len)
    X_test = keras.preprocessing.sequence.pad_sequences(X_test, value=word_index['<PAD>'], padding='post', maxlen=max_len)

    # model
    model = keras.Sequential()
    model.add(keras.layers.Embedding(88000, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    X_val = X_train[:3000]
    X_train = X_train[3000:]

    y_val = y_train[:3000]
    y_train = y_train[3000:]

    model.fit(X_train, y_train, epochs=40, batch_size=8192, validation_data=(X_val, y_val), verbose=1)

    # model.save('textNN.h5')

model = keras.models.load_model(r'D:\Python\textNN.h5')
with open(r'D:/Python/test.txt', encoding='utf-8') as f:
    for line in f:
        nline = re.sub(r'([^\s\w]|_)+', '', line).lower()
        nline = nline.strip().split(' ')
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index['<PAD>'], padding='post', maxlen=250)  # make the data 250 words long
        predict = model.predict(encode)
        print(decode_review(encode[0]))
        print(encode[0])
        print(predict[0])
