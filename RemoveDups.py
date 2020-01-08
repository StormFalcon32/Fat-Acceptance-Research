import re
import pandas as pd
import gensim
import csv


def remove_hashtag_and_short(text):
    # remove URL
    potential = re.sub(
        r'([\d\w]+?:\/\/)?([\w\d\.\-]+)(\.\w+)(:\d{1,5})?(\/\S*)?', ' ', text)
    # remove hashtag
    potential = re.sub(r'#\S+', ' ', potential)
    # remove non ascii characters
    potential = remove_non_ascii(potential)
    # remove leading and trailing whitespace and merge extra whitespace
    potential = re.sub(r'\s+', ' ', potential).strip()
    tokens = gensim.utils.simple_preprocess(potential, deacc=True)
    if len(tokens) < 5:
        return ''
    return text


def remove_non_ascii(s): return ''.join(i for i in s if ord(i) < 128)


def main():
    data = pd.read_csv(r'D:\Python\FatAcceptance\Raw.csv')
    data.drop_duplicates(subset=['user_id', 'date'], inplace=True)
    df = data.to_dict('index')
    new = []
    for i in df:
        cleaned = remove_hashtag_and_short(df[i]['text'])
        subnew = []
        if cleaned:
            subnew.append(df[i]['id'])
            subnew.append(df[i]['user_id'])
            subnew.append(df[i]['date'])
            subnew.append(cleaned)
            subnew.append(df[i]['likes'])
            subnew.append(df[i]['replies'])
            subnew.append(df[i]['retweets'])
            new.append(subnew)
    with open(r'D:\Python\FatAcceptance\NoDups.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'user_id', 'date', 'text',
                         'likes', 'replies', 'retweets'])
        writer.writerows(new)


if __name__ == '__main__':
    main()
