import csv
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM, AutoTokenizer, AutoModelForMaskedLM

# Designate device
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# Set model as bert-base-uncased and use BertForMaskedLM
model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
model_f = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
model_f.load_state_dict(torch.load('../model/bert_finetuned.pth'))
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Init param
cause_uniconn = []
noncause_uniconn = []
candidate_conn = []
cause_uniconn_token = []
noncause_uniconn_token = []
candidate_conn_token = []

df_uniconn = pd.read_csv('../data/uniconn_2level.csv')
df_cause = pd.read_csv('../data/cause_uniconn.csv')
df_noncause = pd.read_csv('../data/noncause_uniconn.csv')

for i in range(len(df_cause)):
    word = df_cause['conn'][i]
    cause_uniconn.append(word)
for i in range(len(df_noncause)):
    word = df_noncause['conn'][i]
    noncause_uniconn.append(word)
for i in range(len(df_uniconn)):
    word = df_uniconn['conn'][i]
    candidate_conn.append(word)

for i in range(len(cause_uniconn)):
    cause_uniconn_token.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cause_uniconn[i])))
for i in range(len(noncause_uniconn)):
    noncause_uniconn_token.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(noncause_uniconn[i])))
for i in range(len(candidate_conn)):
    candidate_conn_token.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(candidate_conn[i])))

emotion_clause = 'the old man was very happy'
cause_candidate = 'a policeman visited the old man with the lost money'
text = '[CLS]' + emotion_clause + '[SEP]' + '[MASK]' + cause_candidate + '[SEP]'

# Tokenize
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Create the tensors of segments
# Add [CLS]+[SEP], [MASK]+[SEP] respectively
segments_ids = [0] * (len(tokenizer.tokenize(emotion_clause)) + 2) + [1] * (len(tokenizer.tokenize(cause_candidate)) + 2)

# Convert tensors to Pytorch tensors
tokens_tensor = torch.tensor([indexed_tokens]).to(device)
segments_tensors = torch.tensor([segments_ids]).to(device)

# Set mode to evaluation
model.eval()
model.to(device)
model_f.eval()
model_f.to(device)

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

# Prediction
candidate_conn_index = []
candidate_conn_score = []
for i in range(len(candidate_conn)):
    candidate_conn_index.append(tokenizer.convert_tokens_to_ids(candidate_conn[i]))
    candidate_conn_score.extend(predictions[0, masked_index, candidate_conn_index[i]].cpu().numpy().tolist())
candidate_index = np.argmax(candidate_conn_score)
conn = candidate_conn[candidate_index]
print(conn)

# Get prediction
with torch.no_grad():
    # [1，14，30522] # [#batch, #word, #vocab]
    # Outputs are the probabilities of words
    outputs = model_f(tokens_tensor, token_type_ids=segments_tensors)
predictions = outputs[0]  # [1，14，30522] # [#batch, #word, #vocab]

# Prediction
candidate_conn_index = []
candidate_conn_score = []
for i in range(len(candidate_conn)):
    candidate_conn_index.append(tokenizer.convert_tokens_to_ids(candidate_conn[i]))
    candidate_conn_score.extend(predictions[0, masked_index, candidate_conn_index[i]].cpu().numpy().tolist())
candidate_index = np.argmax(candidate_conn_score)
conn = candidate_conn[candidate_index]
print(conn)