from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

import InputOutput as io

path = Path(r'D:/Python/NLP/FatAcceptance/Overall')
df = pd.read_csv(path / 'WithRetweets.csv', encoding='utf-8')
df = df.sort_values(by='date')
df.to_csv(r'D:\Python\NLP\FatAcceptance\Overall\Trend.csv', index=False)
start = date(2010, 1, 1)
end = start + timedelta(weeks=1)
weeks = [{0: 0, 1: 0, 2: 0}]
week = 0
start_year = date(2010, 1, 1)
end_year = date(2011, 1, 1)
years = [{0: 0, 1: 0, 2: 0}]
year = 0
for _, row in df.iterrows():
    currdate = datetime.strptime(row['date'], '%Y-%m-%d %H:%M:%S').date()
    if currdate >= end:
        week += 1
        start += timedelta(weeks=1)
        end += timedelta(weeks=1)
        weeks.append({0: 0, 1: 0, 2: 0})
    if currdate >= end_year:
        year += 1
        start_year = date((2010 + year), 1, 1)
        end_year = date((2010 + year + 1), 1, 1)
        years.append({0: 0, 1: 0, 2: 0})
    a = row['pred']
    years[year][row['pred']] += 1
    weeks[week][row['pred']] += 1
df_weeks = pd.DataFrame(weeks)
df_years = pd.DataFrame(years)
df_weeks.to_csv(
    r'D:\Python\NLP\FatAcceptance\Overall\WeeklyTallies.csv', index=False)
df_years.to_csv(
    r'D:\Python\NLP\FatAcceptance\Overall\YearlyTallies.csv', index=False)
