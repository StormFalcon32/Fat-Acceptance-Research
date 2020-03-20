from sklearn.metrics import cohen_kappa_score

import InputOutput as io

data = io.csvIn(r'Training\Final\Labeled.csv', skip_first=True)
labels1 = []
labels2 = []
for row in data:
    labels1.append(row[4])
    labels2.append(row[5])
print(cohen_kappa_score(labels1, labels2))
