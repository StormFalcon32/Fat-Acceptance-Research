import re
import InputOutput as io

data_words = io.csvIn(r'Training\Final\1000Selected0Final.csv', True)
trend = io.csvIn(r'Overall\Trend.csv', True)
for row in data_words:
    matches = re.findall(r'https?://\S+', row[3])
    if matches:
        row.append(1)
    else:
        row.append(0)
