import InputOutput as io
import pandas as pd

data = io.csvIn(r'Overall\Trend.csv', skip_first=True)
user_dict = {}
for row in data:
    if row[9] in user_dict:
        user_dict[row[9]] += 1
    else:
        user_dict[row[9]] = 1
for row in data:
    row.append(user_dict[row[9]])

df = pd.DataFrame(data, columns=['id', 'date', 'text', 'likes', 'replies', 'retweets', 'is_retweet', 'has_comment', 'user_id', 'user_name', 'user_date', 'bio', 'location', 'lat0', 'long0', 'lat1', 'long1', 'lat2', 'long2', 'lat3', 'long3', 'post_days', 'user_days', 'diff', 'label', 'num_tweets'])
df = df.sort_values(by=['user_name', 'date'])
curr_name = ''
curr_num = 0
nums = []
for _, row in df.iterrows():
    if row['user_name'] == curr_name:
        curr_num += 1
        nums.append(curr_num)
    else:
        nums.append(1)
        curr_num = 1
        curr_name = row['user_name']
df['number_by_user'] = nums
df.to_csv(r'D:\Python\NLP\FatAcceptance\Overall\NumTweets.csv', index=False)
