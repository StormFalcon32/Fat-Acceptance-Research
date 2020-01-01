import random
import pandas as pd
import csv


def removePreviousSelected():
    previous = []
    with open(r'D:\Python\FatAcceptance\Selected.csv') as f:
        reader = csv.reader(f)
        line = 0
        for row in reader:
            if line != 0:
                previous.append(int(row[1]))
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
    csvIn = pd.read_csv(r'D:\Python\FatAcceptance\NoDups.csv')
    df = csvIn.to_dict('index')
    previous = removePreviousSelected()
    indices = []
    for i in range(0, len(df)):
        if i in previous:
            continue
        indices.append(i)
    selected_indices = selectKItems(indices, 100, len(indices))
    selected = []
    for i in range(0, len(selected_indices)):
        subselected = []
        subselected.append(i)
        subselected.append(selected_indices[i])
        subselected.append(df[selected_indices[i]]['id'])
        subselected.append(df[selected_indices[i]]['text'])
        subselected.append('')
        selected.append(subselected)
    with open(r'D:\Python\FatAcceptance\Selected2.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['num', 'orig_ind', 'id', 'text', 'label'])
        writer.writerows(selected)


if __name__ == '__main__':
    main()
