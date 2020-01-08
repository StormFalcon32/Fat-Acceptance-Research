import pandas as pd
import csv

csvIn = pd.read_csv(
    r'D:\Python\FatAcceptance\OldDataset\Selected1Ben.csv')
dfS = csvIn.to_dict('index')

csvInOrig = pd.read_csv(r'D:\Python\FatAcceptance\NoDups.csv')
dfO = csvInOrig.to_dict('index')

final = []
for i in range(0, len(dfS)):
    idnum = dfS[i]['id']
    for j in range(0, len(dfO)):
        if dfO[j]['id'] == idnum:
            final.append(dfO[j])
with open(r'D:\Python\FatAcceptance\OldDataset\Selected1Convert.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(final)
