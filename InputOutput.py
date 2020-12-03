import csv


def csvOut(file_dir, cols, data):
    file_dir = r'C:\Data\Python\NLP\FatAcceptance\%s' % file_dir
    with open(file_dir, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if cols != None:
            writer.writerow(cols)
        writer.writerows(data)


def csvOutSingle(file_dir, data):
    file_dir = r'C:\Data\Python\NLP\FatAcceptance\%s' % file_dir
    with open(file_dir, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(data)


def csvIn(file_dir, skip_first):
    file_dir = r'C:\Data\Python\NLP\FatAcceptance\%s' % file_dir
    data = []
    with open(file_dir, encoding='utf-8') as f:
        reader = csv.reader(f)
        line = 0
        for row in reader:
            if line == 0:
                line = 1
                if skip_first:
                    continue
            data.append(row)
    return data


def csvInSingle(file_dir):
    data = []
    file_dir = r'C:\Data\Python\NLP\FatAcceptance\%s' % file_dir
    with open(file_dir, encoding='utf-8') as f:
        reader = csv.reader(f)
        row = next(reader)
        for j in range(0, len(row)):
            data.append(int(row[j]))
    return data
