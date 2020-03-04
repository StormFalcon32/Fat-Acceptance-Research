from pathlib import Path

import pandas as pd


path = Path(r'D:/Python/NLP/FatAcceptance/Overall')
without = pd.read_csv(path / 'WithoutRetweets.csv', encoding='utf-8')
without_dict = without.to_dict('index')
retweets_dict = {}
num = 0
for i in range(0, len(without_dict)):
    num_retweets = without_dict[i]['retweets']
    for j in range(0, num_retweets + 1):
        retweets_dict[num] = without_dict[i]
        num += 1
with_retweets = pd.DataFrame.from_dict(retweets_dict, orient='index', columns = without.columns) 
print(with_retweets)
with_retweets.to_csv(path / 'WithRetweets.csv', encoding='utf-8', index=False)