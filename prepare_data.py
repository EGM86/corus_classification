import pandas as pd
import os
import yaml

from src.natasha_preprocess import Preprocessor

# create folder to save file
data_prepare_path = os.path.join('data', 'prepared')
os.makedirs(data_prepare_path, exist_ok=True)

params = yaml.safe_load(open('params.yaml'))['prepare_data']
quant = params['quant']


# load data
print('load data_parent..')
data_for_model = pd.read_csv('data/raw/rt_data.csv')

# preprocessing text
print('Preprocessor..')
preproc = Preprocessor(remove_tags=True, clean_text=True, lemma=True, stopwords=True)

data_for_model['title'] = data_for_model.title.apply(lambda x: preproc(x) if pd.notnull(x) else x)
data_for_model['text'] = data_for_model.text.apply(lambda x: preproc(x) if pd.notnull(x) else x)

data_for_model['len'] = data_for_model.text.str.split().str.len()

print('filter data by len text..')
data_final = data_for_model[(data_for_model.len >= data_for_model.len.quantile(quant)) & (data_for_model.len <= data_for_model.len.quantile(1 - quant))].copy()
data_final = data_final.reset_index(drop=True)
print('count removed id from data_parent: ', data_for_model.shape[0] - data_final.shape[0])

print('save data_for_parent to file.. data/prepare/data_parent_clean.parquet.gzip')
data_final.to_csv(f'{data_prepare_path}/data_clean.csv', index=False)

