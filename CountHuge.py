from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

path = Path('C:/Data/Python/NLP/FatAcceptance/Overall')
df = pd.read_csv(path / 'WithRetweets.csv', encoding='utf-8')
df = df.sort_values(by='date')

start = date(2010, 1, 1)
end = date(2011, 1, 1)

total = 0
specific_count = 0
general_count = 0

for _, row in df.iterrows():
    currdate = datetime.strptime(row['date'], '%Y-%m-%d %H:%M:%S').date()
    if currdate >= start:
        if currdate <= end:
            if '"huge" marks advance' in row['text'].lower() or 'quot;huge quot; marks advance' in row['text'].lower():
                specific_count += 1
            if 'Huge' in row['text']:
                general_count += 1
            total += 1
        else:
            break

print(total)
print(specific_count)
print(general_count)