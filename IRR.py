from sklearn.metrics import cohen_kappa_score
import csv

benlabels = []
orig = []
with open(r'D:\Python\FatAcceptance\Training\Final\1000Selected0Ben.csv', encoding='utf-8') as f:
    reader = csv.reader(f)
    line = 0
    for row in reader:
        if line == 0:
            line = 1
            continue
        benlabels.append(int(row[7]))
        orig.append(row[:-1])
sadielabels = []
with open(r'D:\Python\FatAcceptance\Training\Final\1000Selected0Sadie.csv', encoding='utf-8') as f:
    reader = csv.reader(f)
    line = 0
    for row in reader:
        if line == 0:
            line = 1
            continue
        sadielabels.append(int(row[7]))
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
with open(r'D:\Python\FatAcceptance\Training\Final\1000Selected0Final.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'user_id', 'date', 'text',
                     'likes', 'replies', 'retweets',
                     'label1', 'label2', 'final_label'])
    writer.writerows(orig)
