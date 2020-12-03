import json
from pathlib import Path

import pandas as pd

path = Path('C:/Data/Python/NLP/FatAcceptance/Overall')

with open(path / 'tweets.json') as f:
    json_data = json.load(f)

total = 0
links = 0
images = 0

df = pd.read_csv(path / 'WithRetweets.csv', encoding='utf-8')
for i, row in df.iterrows():
    print(i)
    tweet = json_data[str(row['num'])]
    if len(tweet['entities']['urls']) > 0:
        links += 1
    if 'media' in tweet['entities']:
        images += 1
    total += 1

print(total)
print(links)
print(images)
