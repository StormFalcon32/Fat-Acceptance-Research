import json
import statistics
from pathlib import Path

import pandas as pd
from geopy.geocoders import Nominatim
import InputOutput as io

path = Path(r'D:/Python/NLP/FatAcceptance/Overall')
df = pd.read_csv(path / 'WithRetweets.csv', encoding='utf-8')
df = df.to_dict('index')
num = 0
with open(path / 'tweets.json') as f:
    json_data = json.load(f)
new_df = {}
locator = Nominatim(user_agent='myGeocoder', timeout=10)
for ind in df:
    tweet = json_data[str(df[ind]['num'])]
    if tweet['place']:
        new_df[num] = df[ind]
        longs = []
        lats = []
        for i in range(4):
            for j in range (2):
                new_df[num][('long%s' if j == 0 else 'lat%s') % i] = tweet['place']['bounding_box']['coordinates'][0][i][j]
            longs.append(new_df[num]['long%s' % i])
            lats.append(new_df[num]['lat%s' % i])
        long_mean = round(statistics.mean(longs), 2)
        lat_mean = round(statistics.mean(lats), 2)
        coordinates = '%s, %s' % (lat_mean, long_mean)
        location = locator.reverse(coordinates)
        if 'error' in location.raw:
            new_df[num]['state'] = 'Error'
        elif location.raw['address']['country_code'] != 'us':
            new_df[num]['state'] = 'Foreign'
        else:
            new_df[num]['state'] = location.raw['address']['state']
            print(location.raw['address']['state'])
        num += 1
output = pd.DataFrame.from_dict(new_df, orient='index')
output = output.sort_values(by='state')
output.to_csv(path /  'Location.csv', index=False)
