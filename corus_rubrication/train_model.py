import pandas as pd
import os
import yaml
import pickle

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate


# create folder to save file
data_predict_path = os.path.join('corus_rubrication','models')
os.makedirs(data_predict_path, exist_ok=True)

#params = yaml.safe_load(open('params.yaml'))['train_model']
#quant = params['quant']

print('load data..')
data = pd.read_csv('corus_rubrication/data/proceed/data_clean.csv')

Tfidf_vect = TfidfVectorizer(max_features=500)
Tfidf_vect.fit(data['text'])

Encoder = LabelEncoder()
Encoder.fit(data['topics'])

label_category = dict(zip(Encoder.classes_, Encoder.transform(Encoder.classes_)))
print('labels: ', label_category)

X = Tfidf_vect.transform(data['text'])
y = Encoder.fit_transform(data['topics'])



print('cross validate..')
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=42)
clf = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')

scores = cross_validate(clf, X, y, cv=cv, scoring='f1_macro')
print(scores['test_score'])

print('train model..')
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)

with open('corus_rubrication/models/model.pkl', "wb") as fd:
    pickle.dump(model, fd)

target_names = ['Мир', 'Наука', 'Спорт', 'Экономика']
print(classification_report(y_test, y_pred, target_names=target_names))