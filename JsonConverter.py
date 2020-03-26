import csv
import json
import re
from datetime import datetime, timedelta
from email.utils import parsedate_tz
from pathlib import Path

import InputOutput as io


def to_datetime(datestring):
    time_tuple = parsedate_tz(datestring.strip())
    dt = datetime(*time_tuple[:6])
    return dt - timedelta(seconds=time_tuple[-1])

path = Path(r'D:/Python/NLP/FatAcceptance/Overall')

with open(path / 'tweets.json') as f:
    json_data = json.load(f)

data = []
data_without = []
for ind in json_data:
    tweet = json_data[ind]
    tweet_list = [ind, tweet['id'], to_datetime(tweet['created_at']), tweet['text']]
    if 'extended_tweet' in tweet:
        tweet_list[3] = tweet['extended_tweet']['full_text']
    if 'retweeted_status' in tweet:
        if 'extended_tweet' in tweet['retweeted_status']:
            tweet_list[3] = tweet['retweeted_status']['extended_tweet']['full_text']
    text = re.sub(r'#([^\s]+)', '', tweet_list[3])
    text = re.sub(r'http\S+', '', text)
    tokenized = text.split()
    if len(tokenized) < 5:
        continue
    has_phrase = False
    has_hashtag = False
    text = tweet_list[3]
    cleaned = re.sub('[^0-9a-zA-Z]+', ' ', text).lower().split()
    for j in range(len(cleaned) - 1):
        if cleaned[j] == 'fat' and cleaned[j + 1] == 'acceptance':
            has_phrase = True
    if '#fatacceptance' in text.lower():
        has_hashtag = True
    if (not has_phrase) and (not has_hashtag):
        continue
    tweet_list[3] = tweet_list[3] + ' xxbio ' + (tweet['user']['description'] if tweet['user']['description'] else '')
    if 'retweeted_status' not in tweet:
        data_without.append(tweet_list)
    data.append(tweet_list)

print(len(data))
print(len(data_without))

io.csvOut(r'Overall\WithRetweets.csv', cols=['num', 'id', 'date', 'text'], data=data)
io.csvOut(r'Overall\WithoutRetweets.csv', cols=['num', 'id', 'date', 'text'], data=data_without)
