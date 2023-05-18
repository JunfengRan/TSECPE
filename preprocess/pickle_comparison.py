import pickle
import csv
import pandas as pd

df = pd.read_csv('../data/discourse_zh.csv')
mydict = {}

for i in range(len(df)):
    sec = df['section'][i]
    mydict['{}'.format(sec)] = eval(df['ec_emotion_pos'][i])

with open ('../data/sentimental_clauses_zh.pkl', 'rb') as f:
    data = pickle.load(f)

for i in range(100):
    sec = df['section'][i]
    if '{}'.format(sec) not in data.keys():
        continue
    print(mydict['{}'.format(sec)], data['{}'.format(sec)])