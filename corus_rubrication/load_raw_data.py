import wget
import os
import pandas as pd

# create folder to save file
data_rawpath = os.path.join('corus_rubrication','data', 'raw')
os.makedirs(data_rawpath, exist_ok=True)

url = 'https://github.com/ods-ai-ml4sg/proj_news_viz/releases/download/data/rt.csv.gz'

print(f'\n загрузка raw данных..')
wget.download(url, out = data_rawpath)

topic_list = ['Спорт', 'Мир', 'Экономика', 'Наука']

df = pd.read_csv(f'{data_rawpath}/rt.csv.gz')

data = df[['url', 'topics', 'title', 'text']]
data = data[data['topics'].isin(topic_list)]
data.to_csv(f'{data_rawpath}/rt_data.csv', index=False)
print(f'\n raw данные загружены в {data_rawpath}/rt_data.csv \n')

os.remove(f'{data_rawpath}/rt.csv.gz')