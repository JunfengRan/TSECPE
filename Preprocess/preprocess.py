import csv
import torch
import numpy as np
from transformers import BertTokenizer

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# Init param
cause_conn = []  # Set cause_conn

# Load connectives info
with open ('../data/cause_conn_modified.txt', 'r', encoding='utf-8') as f:
    line = f.readline()
    while line:
        for word in line.split(','):
            cause_conn.append(word)
        line = f.readline()

# Init csv for recording result
with open ('../data/pairs.csv', 'w', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['pair_type', 'section', 'clause_index', 'candidate_index', 'clause', 'candidate', 'correctness'])

# Init csv for recording raw
with open ('../data/discourse.csv', 'w', encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['section', 'discourse', 'word_count', 'doc_len', 'clause_len', 'ec_emotion_pos', 'ec_cause_pos', 'ce_cause_pos', 'ce_emotion_pos', 'ec_true_pairs', 'ce_true_pairs'])

with open ('../data/all_data_pair.txt', 'r', encoding='utf-8') as f:  # Encode by utf-8 for Chinese
    sec = f.readline()  # Read section ID and length

    # For each section
    while sec:

        # Get section length
        num = sec.split(' ')
        section = int(num[0])
        # if section > 200:
        #     break
        length = int(num[1])
        content = ['' for i in range(length)]
        refined_content = ['' for i in range(length)]

        pairs = f.readline().lstrip().rstrip()  # Get the index of pairs and delete the beginning ' ' and ending '\n'

        # Get the index of pairs (int)
        pairs_index = []
        for pair in pairs.split(', '):
            pairs_index.append(list(map(int, pair.lstrip('(').rstrip(')').split(','))))

        # Get the content of section
        # save for cls
        # sum_len = 1
        sum_len = 0
        word_count = 0
        sentence_len = []
        for i in range(length):
            content[i] = f.readline().lstrip().rstrip().split(',')[3]

            # Get the raw content
            for word in content[i].split(' '):
                refined_content[i] += word
        
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

        # Get emo_index and cau_index
        ec_emo_index = []
        ec_cau_index = []
        ce_emo_index = []
        ce_cau_index = []
        for pair in pairs_index:
            if pair[0] not in ec_emo_index:
                ec_emo_index.append(pair[0])  # pair[0] = emo_index
                ec_cau_index.append([])
            ec_cau_index[ec_emo_index.index(pair[0])].append(pair[1])  # pair[1] = cau_index
        for pair in pairs_index:
            if pair[1] not in ce_cau_index:
                ce_cau_index.append(pair[1])  # pair[1] = cau_index
                ce_emo_index.append([])
            ce_emo_index[ce_cau_index.index(pair[1])].append(pair[0])  # pair[0] = emo_index

        merged_content = ''
        for item in refined_content:
            merged_content += item
        
        ec_pairs_index = pairs_index
        ce_pairs_index = [[pair_index[1], pair_index[0]] for pair_index in pairs_index]
        
        with open ('../data/discourse.csv', 'a', encoding='utf-8', newline='') as g:
            csv_writer = csv.writer(g)
            csv_writer.writerow([section, merged_content, word_count, length, sentence_len, ec_emo_index, ec_cau_index, ce_cau_index, ce_emo_index, ec_pairs_index, ce_pairs_index])

        # Delete original connectives
        for i in range(length):

            # Delete bigram first
            del_pos = []
            new_content = ''
            for j in range(len(refined_content[i]) - 1):
                if refined_content[i][j:j+2] in cause_conn:
                    del_pos.extend([j,j+1])
            for j in range(len(refined_content[i])):
                if j not in del_pos:
                    new_content = new_content + refined_content[i][j]
            refined_content[i] = new_content

            # Delete unigram later
            del_pos = []
            new_content = ''
            for j in range(len(refined_content[i])):
                if refined_content[i][j] in cause_conn:
                    del_pos.append(j)
            for j in range(len(refined_content[i])):
                if j not in del_pos:
                    new_content = new_content + refined_content[i][j]
            refined_content[i] = new_content

            # Padding
            if refined_content[i] == '':
                refined_content[i] = '[UNK]'

        '''
        # For each emo_clause
        # Construct pairs
        correctness = ''
        for emo_clause_index in emo_index:
            for i in range(length):
                if i + 1 not in emo_index:
                    cau_candidate_index = i + 1
                    # pairs = '[CLS]' + refined_content[emo_clause_index - 1] + '[SEP]' + '[MASK]' + refined_content[i] + '[SEP]'
                    emotion_clause = refined_content[emo_clause_index - 1]
                    cause_candidate = refined_content[i]
                    correctness = 'false'
                    if i + 1 in cau_index[emo_index.index(emo_clause_index)]:
                        correctness = 'true'
                    
                    # Write result in csv
                    with open ('data/test/pairs.csv', 'a', encoding='utf-8', newline='') as g:
                        csv_writer = csv.writer(g)
                        csv_writer.writerow([section, emo_clause_index, cau_candidate_index, emotion_clause, cause_candidate, correctness])

            # For each emo_clause considering itself
            cau_candidate_index = emo_clause_index
            emotion_clause = refined_content[emo_clause_index - 1]
            cause_candidate = refined_content[cau_candidate_index - 1]
            correctness = 'false'
            if cau_candidate_index in cau_index[emo_index.index(emo_clause_index)]:
                correctness = 'true'
            
            # Write result in csv
            with open ('data/test/pairs.csv', 'a', encoding='utf-8', newline='') as g:
                csv_writer = csv.writer(g)
                csv_writer.writerow([section, emo_clause_index, cau_candidate_index, emotion_clause, cause_candidate, correctness])
        '''
        
        # For each emo_clause
        # Construct paris
        correctness = ''
        pair_type = 'ec'
        for emo_clause_index in ec_emo_index:
            for i in range(length):
                cau_candidate_index = i + 1
                # pairs = '[CLS]' + refined_content[emo_clause_index - 1] + '[SEP]' + '[MASK]' + refined_content[i] + '[SEP]'
                emotion_clause = refined_content[emo_clause_index - 1]
                cause_candidate = refined_content[i]
                correctness = 'false'
                if i + 1 in ec_cau_index[ec_emo_index.index(emo_clause_index)]:
                    correctness = 'true'
                    
                # Write result in csv
                with open ('../data/pairs.csv', 'a', encoding='utf-8', newline='') as g:
                    csv_writer = csv.writer(g)
                    csv_writer.writerow([pair_type, section, emo_clause_index, cau_candidate_index, emotion_clause, cause_candidate, correctness])
                    
        # For each cau_clause
        # Construct paris
        correctness = ''
        pair_type = 'ce'
        for cau_clause_index in ce_cau_index:
            for i in range(length):
                emo_candidate_index = i + 1
                # pairs = '[CLS]' + refined_content[cau_clause_index - 1] + '[SEP]' + '[MASK]' + refined_content[i] + '[SEP]'
                cause_clause = refined_content[cau_clause_index - 1]
                emotion_candidate = refined_content[i]
                correctness = 'false'
                if i + 1 in ce_emo_index[ce_cau_index.index(cau_clause_index)]:
                    correctness = 'true'
                    
                # Write result in csv
                with open ('../data/pairs.csv', 'a', encoding='utf-8', newline='') as g:
                    csv_writer = csv.writer(g)
                    csv_writer.writerow([pair_type, section, cau_clause_index, emo_candidate_index, cause_clause, emotion_candidate, correctness])
        
        sec = f.readline()  # Read following section length