import json
from pathlib import Path

import pandas as pd

import InputOutput as io

path = Path(r'D:/Python/NLP/FatAcceptance/Overall')
df = pd.read_csv(path / 'WithRetweets.csv', encoding='utf-8')
df = df.to_dict('index')
num = 0
numtoind = {}
for ind in df:
    numtoind[df[ind]['num']] = ind
with open(path / 'tweets.json') as f:
    json_data = json.load(f)
for ind in df:
    tweet = json_data[str(df[ind]['num'])]
    if tweet['place']:
        for i in range(4):
            for j in range (2):
                df[ind][('long%s' if j == 0 else 'lat%s') % i] = tweet['place']['bounding_box']['coordinates'][0][i][j]
        num += 1
print(num)
output = pd.DataFrame.from_dict(df, orient='index')
output.to_csv(path /  'Location.csv', index=False)
