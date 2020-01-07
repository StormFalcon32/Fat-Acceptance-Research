from sklearn.metrics import cohen_kappa_score
import csv

benlabels = []
orig = []
with open(r'D:\Python\FatAcceptance\Selected1Ben.csv', encoding='utf-8') as f:
    reader = csv.reader(f)
    line = 0
    for row in reader:
        if line == 0:
            line = 1
            continue
        benlabels.append(int(row[4]))
        orig.append(row)
sadielabels = []
with open(r'D:\Python\FatAcceptance\Selected1Sadie.csv', encoding='utf-8') as f:
    reader = csv.reader(f)
    line = 0
    for row in reader:
        if line == 0:
            line = 1
            continue
        sadielabels.append(int(row[4]))
kappa = cohen_kappa_score(benlabels, sadielabels)
print(kappa)
final = []
ind = 0
for row in orig:
    final.append([int(row[0]), int(row[1]), int(row[2]),
                  row[3], benlabels[ind], sadielabels[ind], ""])
    ind += 1
with open(r'D:\Python\FatAcceptance\Selected1Final.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['num', 'orig_ind', 'id', 'text',
                     'label1', 'label2', 'final_label'])
    writer.writerows(final)
