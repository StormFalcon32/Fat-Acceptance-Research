import re
import pandas as pd
import gensim
import csv
import InputOutput as io


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
    data = io.csvIn(r'Overall\Raw.csv', skip_first=True)
    new = []
    num = 0
    for row in data:
        cleaned = remove_hashtag_and_short(row[2])
        if cleaned:
            new.append(row)
        num += 1
    io.csvOut(r'Overall\NoDups.csv', cols=['id', 'date', 'text', 'likes', 'replies', 'retweets', 'is_retweet', 'has_comment', 'user_id', 'user_name', 'user_date', 'bio', 'location', 'lat0', 'long0', 'lat1', 'long1', 'lat2', 'long2', 'lat3', 'long3', 'post_days', 'user_days', 'diff'], data=new)


if __name__ == '__main__':
    main()
