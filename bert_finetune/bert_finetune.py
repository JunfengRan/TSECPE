import csv
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data import Dataset, DataLoader

def my_collate_fn(batch):
    batch = zip(*batch)
    tokens_tensor, segments_tensor, masked_tensor, masked_index, indexed_conn = batch

    tokens_tensor = torch.tensor([item.tolist() for item in tokens_tensor]).to(torch.int32)
    segments_tensor = torch.tensor([item.tolist() for item in segments_tensor]).to(torch.int32)
    masked_tensor = torch.tensor([item.tolist() for item in masked_tensor]).to(torch.int32)
    masked_index = torch.tensor([item.tolist() for item in masked_index]).to(torch.int32)
    indexed_conn = torch.tensor([item.tolist() for item in indexed_conn]).to(torch.int32)
    
    return tokens_tensor, segments_tensor, masked_tensor, masked_index, indexed_conn

class Pair(Dataset):
    def __init__(self, tokenizer, path, start, end):
        self.tokenizer = tokenizer
        self.tokens_tensors = []
        self.segments_tensors = []
        self.masked_tensors = []
        self.masked_indexes = []
        self.indexed_conns = []
        
        df = pd.read_csv(path)
        
        # Get MASK index
        # Uniconn
        def get_index1(lst=None, item=''):
            return [index for (index,value) in enumerate(lst) if value == item]
        
        for i in range(len(df)):
            if i < start or i >= end:
                continue
            arg1 = str(df['Arg1_RawText'][i]).lower()
            arg2 = str(df['Arg2_RawText'][i]).lower()
            conn = df['conn'][i].lower()
            
            text = '[CLS]' + arg1 + '[SEP]' + '[MASK]' + arg2 + '[SEP]'
        
            # Tokenize
            tokenized_text = tokenizer.tokenize(text)
            if len(tokenized_text) > 512:
                continue
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            tokenized_conn = tokenizer.tokenize(conn)
            indexed_conn = torch.tensor(tokenizer.convert_tokens_to_ids(tokenized_conn))

            # Create the tensors of segments
            # Add [CLS]+[SEP], [MASK]+[SEP] respectively
            segments_ids = [0] * (len(tokenizer.tokenize(arg1)) + 2) + [1] * (510 - len(tokenizer.tokenize(arg1)))
            masked_ids = [1] * len(tokenized_text) + [0] * (512 - len(tokenized_text))

            # Convert tensors to Pytorch tensors
            tokens_tensor = torch.tensor(indexed_tokens)
            tokens_tensor = F.pad(tokens_tensor, (0, 512 - tokens_tensor.size(-1)), 'constant', 0)
            segments_tensor = torch.tensor(segments_ids)
            masked_tensor = torch.tensor(masked_ids)

            # Get MASK index
            masked_index = torch.tensor(get_index1(tokenized_text, '[MASK]'))
            
            self.tokens_tensors.append(tokens_tensor)
            self.segments_tensors.append(segments_tensor)
            self.masked_tensors.append(masked_tensor)
            self.masked_indexes.append(masked_index)
            self.indexed_conns.append(indexed_conn)

    def __getitem__(self, idx):
        return self.tokens_tensors[idx], self.segments_tensors[idx], self.masked_tensors[idx], self.masked_indexes[idx], self.indexed_conns[idx]
    
    def __len__(self):
        return len(self.tokens_tensors)

# Init tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')

dataset_path = '../data/dataset.csv'

dataset_len = len(pd.read_csv(dataset_path))
train_start = int(0 * dataset_len)
train_end = int(0.9 * dataset_len)
train_dataset = Pair(tokenizer, dataset_path, train_start, train_end)
train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=16, collate_fn=my_collate_fn, drop_last=True)
test_start = int(0.9 * dataset_len)
test_end = int(1 * dataset_len)
test_dataset = Pair(tokenizer, dataset_path, test_start, test_end)
test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=16, collate_fn=my_collate_fn, drop_last=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

min_test_loss = None
for epoch in range(50):
    model.train()
    train_loss = 0
    for train_step, batch in enumerate(train_loader):
        
        # Clear gradient
        optimizer.zero_grad()
        
        tokens_tensor, segments_tensor, masked_tensor, masked_index, indexed_conn = batch
        outputs = model(tokens_tensor.to(DEVICE), masked_tensor.to(DEVICE), segments_tensor.to(DEVICE))
        predictions = outputs[0]  # [#batch, #word, #vocab]
        
        # Get prediction
        # Outputs are the probabilities of words
        for i in range(tokens_tensor.size(0)):
            loss = torch.nn.CrossEntropyLoss()
            output = predictions[i, masked_index[i][0]].view(-1)
            target = torch.zeros(output.size())
            target[indexed_conn[i][0]] = 1
            target = target.to(DEVICE)
            loss = loss(output, target)
            train_loss += loss
        # Backward
        loss.requires_grad_(True)
        loss.backward()
        # Update parameters
        optimizer.step()
    
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for test_step, batch in enumerate(test_loader):
            
            tokens_tensor, segments_tensor, masked_tensor, masked_index, indexed_conn = batch
            outputs = model(tokens_tensor.to(DEVICE), masked_tensor.to(DEVICE), segments_tensor.to(DEVICE))
            predictions = outputs[0]  # [#batch, #word, #vocab]
            
            # Get prediction
            # Outputs are the probabilities of words
            for i in range(tokens_tensor.size(0)):
                loss = torch.nn.CrossEntropyLoss()
                output = predictions[i, masked_index[i][0]].view(-1)
                target = torch.zeros(output.size())
                target[indexed_conn[i][0]] = 1
                target = target.to(DEVICE)
                loss = loss(output, target)
                test_loss += loss
            
    if min_test_loss is None or test_loss < min_test_loss:
        early_stop_flag = 1
        min_test_loss = test_loss
        torch.save(model.state_dict(), 'model/bert_finetuned.pth')
    else:
        early_stop_flag += 1
    if early_stop_flag >= 10:
        break
    
    print('epoch: {}, train_loss: {}, test_loss: {}'.format(epoch, train_loss, test_loss))