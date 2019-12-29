from sklearn.metrics import cohen_kappa_score
import csv

ben = []
with open(r'D:\Python\FatAcceptance\SelectedRound1Ben.csv') as f:
    reader = csv.reader(f)
    line = 0
    for row in reader:
        if line == 0:
            line = 1
            continue
        ben.append(row[4])
sadie = []
with open(r'D:\Python\FatAcceptance\SelectedRound1Sadie.csv') as f:
    reader = csv.reader(f)
    line = 0
    for row in reader:
        if line == 0:
            line = 1
            continue
        sadie.append(row[4])
kappa = cohen_kappa_score(ben, sadie)
print(kappa)
