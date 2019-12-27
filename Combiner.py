import os
import glob
import pandas as pd
os.chdir(r'D:\Python\GamingDisorder\RawScraped')
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
combined_csv.to_csv(r'D:\Python\GamingDisorder\Raw.csv', index=False)
