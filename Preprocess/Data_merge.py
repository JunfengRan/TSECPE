import pandas as pd
import numpy as np
import csv

df_pairs = pd.read_csv('data/pairs_withconn_super.csv')
df_discourse = pd.read_csv('data/discourse.csv')
connLists = []
header = 0

for i in range(len(df_discourse)):
    discourse_connLists = ''
    for j in range(len(eval(df_discourse['ec_emotion_pos'][i]))):
        emotion_connList = ''
        for k in range(int(df_discourse['doc_len'][i])):
            emotion_connList += df_pairs['conn'][header]
            header += 1
        discourse_connLists += emotion_connList
    for j in range(len(eval(df_discourse['ce_cause_pos'][i]))):
        cause_connList = ''
        for k in range(int(df_discourse['doc_len'][i])):
            cause_connList += df_pairs['conn'][header]
            header += 1
        discourse_connLists += cause_connList
    connLists.append(discourse_connLists)

df_discourse['conn'] = connLists
# df_discourse.to_json('data_out/discourse_withconn.json', orient='records', lines=True, force_ascii=False)
df_discourse.to_csv('data_out/discourse_withconn_super.csv', index=False)
