import csv
from pathlib import Path

path = Path(r'C:/Data/Python/NLP/FatAcceptance/Training/Final/ULMFiT')
with open(path / 'pred.csv', encoding='utf') as f:
    reader = csv.reader(f)
    line = 0
    num = 0
    for row in reader:
        line += 1
        if line == 1:
            continue
        if row[0] == '0' and row[2] == '1':
            num += 1
            print('%s: %s' % (num, row[1]))