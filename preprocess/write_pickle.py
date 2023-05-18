import csv
import pickle
import pandas as pd

df = pd.read_csv('../data/discourse_en.csv')
mydict = {}

with open ('../data/sentimental_clauses_en.pkl', 'wb') as f:
    for i in range(len(df)):
        sec = df['section'][i]
        mydict["{}".format(sec)] = eval(df['ec_emotion_pos'][i])
    pickle.dump(mydict, f)