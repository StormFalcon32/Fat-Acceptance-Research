import json
from pathlib import Path

import pandas as pd
from statistics import mean

path = Path('C:/Data/Python/NLP/FatAcceptance/Overall')
with open(path / 'tweets.json') as f:
    json_data = json.load(f)
print(len(json_data))
df = pd.read_csv(path / 'WithRetweets.csv', encoding='utf-8')
print(len(df))

total = 0
cat0_likes = []
cat1_likes = []
cat0_retweets = []
cat1_retweets = []

for _, row in df.iterrows():
    tweet_id = row['id']
    found_tweet = False
    for tweet_ind in json_data:
        if json_data[tweet_ind]['id'] == tweet_id:
            if row['pred'] == 0:
                cat0_likes.append(json_data[tweet_ind]['favorite_count'])
                cat0_retweets.append(json_data[tweet_ind]['retweet_count'])
            elif row['pred'] == 1:
                cat1_likes.append(json_data[tweet_ind]['favorite_count'])
                cat1_retweets.append(json_data[tweet_ind]['retweet_count'])
            found_tweet = True
    if not found_tweet:
        print()
    total += 1
    print(total)

print(total)
print(mean(cat0_likes))
print(mean(cat0_retweets))
print(mean(cat1_likes))
print(mean(cat1_retweets))
