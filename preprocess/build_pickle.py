import csv
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer
import pickle

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

emotion_words_all = []
emotion_indexes_all = []

with open ('../data/all_data_pair_en.txt', 'r') as f:
    sec = f.readline()  # Read section ID and length

    # For each section
    while sec:

        # Get section length
        num = sec.split(' ')
        section = int(num[0])
        length = int(num[1])
        content = ['' for i in range(length)]
        refined_content = ['' for i in range(length)]

        pairs = f.readline().lstrip().rstrip()  # Get the index of pairs and delete the beginning ' ' and ending '\n'

        emotion_words = []

        # Get the content of section
        # save for cls
        # sum_len = 1
        sum_len = 0
        word_count = 0
        sentence_len = []
        for i in range(length):

            line = f.readline().lstrip().rstrip().split(',')
            
            line[1], line[2] = line[1].lower(), line[2].lower()
            
            if line[1] != 'null' and line[1] != '' and line[1] != ' ':
                if line[1] not in emotion_words:
                    emotion_words.append(line[1])
            if line[2] != 'null' and line[2] != '' and line[2] != ' ':
                if line[2] not in emotion_words:
                    emotion_words.append(line[2])
            
            content[i] = line[3]

            # Get the raw content
            refined_content[i] = content[i].lower()
        
            content_len = len(tokenizer.tokenize(refined_content[i]))
            
            # save for sep
            # sum_len += content_len + 2
            sum_len += content_len
            word_count += content_len
            sentence_len.append(content_len)
        
        # Set Bert_trunk (pass)
        if sum_len > 512:
            sec = f.readline()
            continue
        
        for item in emotion_words:
            if item not in emotion_words_all:
                emotion_words_all.append(item)
        
        sec = f.readline()  # Read following section length


with open ('../data/all_data_pair_en.txt', 'r') as f:
    sec = f.readline()  # Read section ID and length

    # For each section
    while sec:

        # Get section length
        num = sec.split(' ')
        section = int(num[0])
        length = int(num[1])
        content = ['' for i in range(length)]
        refined_content = ['' for i in range(length)]

        pairs = f.readline().lstrip().rstrip()  # Get the index of pairs and delete the beginning ' ' and ending '\n'
        
        emotion_indexes = []

        # Get the content of section
        # save for cls
        # sum_len = 1
        sum_len = 0
        word_count = 0
        sentence_len = []
        for i in range(length):
            
            line = f.readline().lstrip().rstrip().split(',')
            
            content[i] = line[3]

            # Get the raw content
            refined_content[i] = content[i].lower()

            content_len = len(tokenizer.tokenize(refined_content[i]))
            
            # save for sep
            # sum_len += content_len + 2
            sum_len += content_len
            word_count += content_len
            sentence_len.append(content_len)
        
        # Set Bert_trunk (pass)
        if sum_len > 512:
            sec = f.readline()
            continue

        for i in range(length):
            sentence_words = refined_content[i].split(' ')
            for j in range(len(sentence_words)):
                if sentence_words[j] in emotion_words_all:
                    emotion_indexes.append(i + 1)
                    break
        
        emotion_indexes_all.append(emotion_indexes)
        
        sec = f.readline()  # Read following section length
        
df = pd.read_csv('../data/discourse_en.csv')
mydict = {}

with open ('../data/sentimental_clauses_en.pkl', 'wb') as f:
    for i in range(len(df)):
        sec = df['section'][i]
        emotion_indexes_list = []
        for pos in eval(df['ec_emotion_pos'][i]):
            if pos - 1 in emotion_indexes_all[i]:
                if pos - 1 not in emotion_indexes_list:
                    emotion_indexes_list.append(pos - 1)
            if pos not in emotion_indexes_list:
                emotion_indexes_list.append(pos)
            if pos + 1 in emotion_indexes_all[i]:
                if pos + 1 not in emotion_indexes_list:
                    emotion_indexes_list.append(pos + 1)
        mydict["{}".format(sec)] = emotion_indexes_list
    pickle.dump(mydict, f)

# with open ('../data/sentimental_clauses_en.pkl', 'wb') as f:
#     for i in range(len(df)):
#         sec = df['section'][i]
#         for pos in eval(df['ec_emotion_pos'][i]):
#             if pos not in emotion_indexes_all[i]:
#                 emotion_indexes_all[i].append(pos)
#         mydict["{}".format(sec)] = emotion_indexes_all[i]
#     pickle.dump(mydict, f)