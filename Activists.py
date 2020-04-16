import json
from datetime import date, datetime, timedelta
from email.utils import parsedate_tz
from pathlib import Path

import pandas as pd


def to_datetime(datestring):
    time_tuple = parsedate_tz(datestring.strip())
    dt = datetime(*time_tuple[:6])
    return dt - timedelta(seconds=time_tuple[-1])

path = Path(r'D:/Python/NLP/FatAcceptance/Overall')
with open(path / 'tweets.json') as f:
    json_data = json.load(f)
df = pd.read_csv(path / 'WithRetweets.csv', encoding='utf-8')
user_ids = []
user_dates = []
for _, row in df.iterrows():
    user_ids.append(json_data[str(row['num'])]['user']['id'])
    user_dates.append(to_datetime(json_data[str(row['num'])]['user']['created_at']))
df['user_id'] = user_ids
df['user_date'] = user_dates
user_dict = {}
for _, row in df.iterrows():
    if row['user_id'] in user_dict:
        user_dict[row['user_id']] += 1
    else:
        user_dict[row['user_id']] = 1
user_nums = []
for _, row in df.iterrows():
    user_nums.append(user_dict[row['user_id']])
df['num_tweets'] = user_nums
df = df.sort_values(by=['user_id', 'date'])
curr_id = 0
curr_num = 0
nums = []
for _, row in df.iterrows():
    if row['user_id'] == curr_id:
        curr_num += 1
        nums.append(curr_num)
    else:
        nums.append(1)
        curr_num = 1
        curr_name = row['user_id']
df['number_by_user'] = nums
start = date(2006, 3, 1)
days = []
user_days = []
post_days = []
tallies = {}
for _, row in df.iterrows():
    postdate = datetime.strptime(row['date'], '%Y-%m-%d %H:%M:%S').date()
    userdate = row['user_date'].date()
    diff = (postdate - userdate).days
    days.append(diff)
    user_days.append((userdate - start).days)
    post_days.append((postdate - start).days)
    diff = int(diff / 365)
    if diff in tallies:
        tallies[diff][row['pred']] += 1
    else:
        tallies[diff] = {'year': diff, 0: 0, 1: 0, 2: 0}
        tallies[diff][row['pred']] += 1

df['user_days'] = user_days
df['post_days'] = post_days
df['diff'] = days
df_tallies = pd.DataFrame.from_dict(tallies, orient='index')
df_tallies = df_tallies.sort_values(by='year')
df.to_csv(path / 'UserAndDates.csv', index=False)
df_tallies.to_csv(path / 'TenureTallies.csv', index=False)
