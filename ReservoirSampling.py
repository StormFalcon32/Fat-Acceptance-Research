import random
import pandas as pd
import csv
import InputOutput as io


def removePreviousSelected():
    previous = []
    with open(r'D:\Python\NLP\FatAcceptance\Training\Selected1Ben.csv', encoding='utf-8') as f:
        reader = csv.reader(f)
        line = 0
        for row in reader:
            if line != 0:
                previous.append(int(row[0]))
            line += 1
        return previous


def selectKItems(stream, k, n):
    i = 0
    reservoir = [0] * k
    for i in range(k):
        reservoir[i] = stream[i]
    while i < n:
        j = random.randrange(i + 1)
        if j < k:
            reservoir[j] = stream[i]
        i += 1
    return reservoir


def main():
    csvIn = pd.read_csv(r'D:\Python\NLP\FatAcceptance\Overall\NoDups.csv')
    df = csvIn.to_dict('index')
    previous = removePreviousSelected()
    indices = []
    for i in range(0, len(df)):
        if df[i]['id'] in previous:
            continue
        indices.append(i)
    selected_indices = selectKItems(indices, 1000, len(indices))
    selected = []
    for i in range(0, len(selected_indices)):
        subselected = []
        for key in df[selected_indices[i]]:
            subselected.append(df[selected_indices[i]][key])
        subselected.append('')
        selected.append(subselected)
    io.csvOut(r'Training\Selected2.csv', cols=['id', 'user_id', 'date', 'text',
                                               'likes', 'replies', 'retweets', 'label'], data=selected)


if __name__ == '__main__':
    main()
