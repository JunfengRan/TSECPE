import csv
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM

# Init param
cause_uniconn = []
noncause_uniconn = []
candidate_conn = []
with open ('../data/cause_uniconn_modified.txt', 'r', encoding='utf-8') as f:
    line = f.readline()
    while line:
        for word in line.split(','):
            cause_uniconn.append(word)
        line = f.readline()
with open ('../data/noncause_uniconn.txt', 'r', encoding='utf-8') as f:
    line = f.readline()
    while line:
        for word in line.split(','):
            noncause_uniconn.append(word)
        line = f.readline()
with open ('../data/uniconn_modified.txt', 'r', encoding='utf-8') as f:
    line = f.readline()
    while line:
        for word in line.split(','):
            candidate_conn.append(word)
        line = f.readline()

# Load dataset
df = pd.read_csv('../data/pairs.csv')

# Init csv of result
with open ('../data/pairs_withconn_super.csv', 'w', encoding='utf-8', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['pair_type', 'section', 'clause_index', 'candidate_index', 'clause', 'candidate', 'conn', 'correctness', 'is_cause_conn'])

# Designate device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Set model as bert-base-uncased and use BertForMaskedLM
model = BertForMaskedLM.from_pretrained('bert-base-chinese')
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

for i in range(len(df)):
    pair_type = df['pair_type'][i]
    section = df['section'][i]
    clause_index = df['clause_index'][i]
    candidate_index = df['candidate_index'][i]
    clause = df['clause'][i]
    candidate = df['candidate'][i]
    text = '[CLS]' + str(clause) + '[SEP]' + '[MASK]' + str(candidate) + '[SEP]'
    correctness = df['correctness'][i]

    # Tokenize
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Create the tensors of segments
    # Add [CLS]+[SEP], [MASK]+[SEP] respectively
    segments_ids = [0] * (len(tokenizer.tokenize(clause)) + 2) + [1] * (len(tokenizer.tokenize(candidate)) + 2)

    # Convert tensors to Pytorch tensors
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    segments_tensors = torch.tensor([segments_ids]).to(device)

    # Set mode to evaluation
    model.eval()
    model.to(device)

    # Get MASK index
    # Uniconn
    def get_index1(lst=None, item=''):
        return [index for (index,value) in enumerate(lst) if value == item]
    masked_index = get_index1(tokenized_text, '[MASK]')

    # Get prediction
    with torch.no_grad():
        # [1，14，30522] # [#batch, #word, #vocab]
        # Outputs are the probabilities of words
        outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    predictions = outputs[0]  # [1，14，30522] # [#batch, #word, #vocab]

    '''
    # Probability distribution
    print(torch.exp(predictions[0, masked_index])/torch.sum(torch.exp(predictions[0, masked_index])))
    '''

    # Prediction
    candidate_conn_index = []
    candidate_conn_score = []
    for i in range(len(candidate_conn)):
        candidate_conn_index.append(tokenizer.convert_tokens_to_ids(candidate_conn[i]))
        candidate_conn_score.extend(predictions[0, masked_index, candidate_conn_index[i]].cpu().numpy().tolist())
        
    cause_conn_index = []
    cause_conn_score = []
    for i in range(len(cause_uniconn)):
        cause_conn_index.append(tokenizer.convert_tokens_to_ids(cause_uniconn[i]))
        cause_conn_score.extend(predictions[0, masked_index, cause_conn_index[i]].cpu().numpy().tolist())
    
    noncause_conn_index = []
    noncause_conn_score = []
    for i in range(len(noncause_uniconn)):
        noncause_conn_index.append(tokenizer.convert_tokens_to_ids(noncause_uniconn[i]))
        noncause_conn_score.extend(predictions[0, masked_index, noncause_conn_index[i]].cpu().numpy().tolist())
    
    if correctness == True:
        conn = '因'
        
    else:
        conn_candidate_index = np.argmax(noncause_conn_score)
        conn = noncause_uniconn[conn_candidate_index]

    if conn in cause_uniconn:
        is_cause_conn = 'true'
    else:
        is_cause_conn = 'false'

    # Write result in csv
    with open ('../data/pairs_withconn_super.csv', 'a', encoding='utf-8', newline='') as g:
        csv_writer = csv.writer(g)
        csv_writer.writerow([pair_type, section, clause_index, candidate_index, clause, candidate, conn, correctness, is_cause_conn])