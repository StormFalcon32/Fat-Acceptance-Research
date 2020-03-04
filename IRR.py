from sklearn.metrics import cohen_kappa_score
import csv
import InputOutput as io

benlabels = []
orig = []
with open(r'D:\Python\NLP\FatAcceptance\Training\1000Selected1.csv', encoding='utf-8') as f:
    reader = csv.reader(f)
    line = 0
    for row in reader:
        if line == 0:
            line = 1
            continue
        benlabels.append(int(row[7]))
        orig.append(row[:-1])
        line += 1
sadielabels = []
with open(r'D:\Python\NLP\FatAcceptance\Training\1000Selected1Sadie.csv', encoding='utf-8') as f:
    reader = csv.reader(f)
    line = 0
    for row in reader:
        if line == 0:
            line = 1
            continue
        sadielabels.append(int(row[7]))
        line += 1
kappa = cohen_kappa_score(benlabels, sadielabels)
print(kappa)
ind = 0
for row in orig:
    row.append(benlabels[ind])
    row.append(sadielabels[ind])
    if benlabels[ind] != sadielabels[ind]:
        row.append("*")
    else:
        row.append("")
    ind += 1
io.csvOut(r'Training\1000Selected1Final.csv', cols=[
          'id', 'user_id', 'date', 'text',
          'likes', 'replies', 'retweets',
          'label1', 'label2', 'label'], data=orig)
