import wget
import os
import pandas as pd

# create folder to save file
data_rawpath = os.path.join('data', 'raw')
os.makedirs(data_rawpath, exist_ok=True)

url = 'https://github.com/ods-ai-ml4sg/proj_news_viz/releases/download/data/rt.csv.gz'

wget.download(url, out = data_rawpath)

topic_list = ['Спорт', 'Мир', 'Экономика', 'Наука']

df = pd.read_csv('data/raw/rt.csv.gz')

data = df[['url', 'topics', 'title', 'text']]
data = data[data['topics'].isin(topic_list)]
data.to_csv('data/raw/rt_data.csv', index=False)