import json
from pathlib import Path

import pandas as pd

path = Path('C:/Data/Python/NLP/FatAcceptance/Overall')

with open(path / 'tweets.json') as f:
    json_data = json.load(f)


def count():
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

def split():
    total = 0
    links_pos = 0
    links_neg = 0
    youtube_pos = 0
    youtube_neg = 0
    tumblr_pos = 0
    tumblr_neg = 0
    media_pos = 0
    media_neg = 0
    df = pd.read_csv(path / 'WithRetweets.csv', encoding='utf-8')
    for i, row in df.iterrows():
        # print(i)
        tweet = json_data[str(row['num'])]
        if len(tweet['entities']['urls']) > 0:
            if row['pred'] == 0:
                links_pos += 1
            if row['pred'] == 1:
                links_neg += 1
            for link in tweet['entities']['urls']:
                if 'youtube' in link['expanded_url']:
                    if row['pred'] == 0:
                        youtube_pos += 1
                    if row['pred'] == 1:
                        youtube_neg += 1
                if 'tmblr' in link['expanded_url'] or 'tumblr' in link['expanded_url']:
                    if row['pred'] == 0:
                        tumblr_pos += 1
                    if row['pred'] == 1:
                        tumblr_neg += 1
        if 'media' in tweet['entities']:
            if row['pred'] == 0:
                media_pos += 1
            if row['pred'] == 1:
                media_neg += 1
        total += 1
    print(total)
    print((links_pos / (links_pos + links_neg)))
    print((media_pos / (media_pos + media_neg)))
    print((youtube_pos / (youtube_pos + youtube_neg)))
    print((tumblr_pos / (tumblr_pos + tumblr_neg)))
    print((tumblr_pos + tumblr_neg))

split()
