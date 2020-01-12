import csv


def add():
    final = []
    with open(r'D:\Python\FatAcceptance\Training\1000Selected0Blank.csv', encoding='utf-8') as f:
        reader = csv.reader(f)
        line = 0
        for row in reader:
            if line == 0:
                line = 1
                continue
            row.insert(0, line)
            final.append(row)
            line += 1

    with open(r'D:\Python\FatAcceptance\Training\1000Selected0Ind.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['num', 'id', 'user_id', 'date', 'text',
                         'likes', 'replies', 'retweets', 'label'])
        writer.writerows(final)


add()
