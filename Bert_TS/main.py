import os
import csv
from config import *
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer, BertForSequenceClassification
from model_shared_linear import Network
import datetime
import numpy as np
import pandas as pd
import pickle
from utils.utils import *
from accelerate import Accelerator


'''
batch size = 1 for experiment
'''


# dataset
class Discourse(Dataset):
    def __init__(self, tokenizer, path):
        self.tokenizer = tokenizer
        self.data_path = path
        self.discourses_list = []
        df = pd.read_csv(self.data_path)
        for i in range(len(df)):
            section = int(df['section'][i])
            discourse = torch.Tensor(self.tokenizer(df['discourse'][i],padding='max_length',max_length=512)['input_ids']).to(torch.int32)
            word_count = int(df['word_count'][i])
            doc_len = int(df['doc_len'][i])
            clause_len = df['clause_len'][i]
            ec_emotion_pos = df['ec_emotion_pos'][i]
            ec_cause_pos = df['ec_cause_pos'][i]
            ce_cause_pos = df['ce_cause_pos'][i]
            ce_emotion_pos = df['ce_emotion_pos'][i]
            ec_true_pairs = df['ec_true_pairs'][i]
            ce_true_pairs = df['ce_true_pairs'][i]
            conn = torch.Tensor(self.tokenizer(df['conn'][i],padding='max_length',max_length=512)['input_ids']).to(torch.int32)
            self.discourses_list.append([section, discourse, word_count, doc_len, clause_len, ec_emotion_pos, ec_cause_pos, ce_cause_pos, ce_emotion_pos, ec_true_pairs, ce_true_pairs, conn])

    def __getitem__(self, item):
        item = self.discourses_list[item]
        return item

    def __len__(self):
        return len(self.discourses_list)


# evaluate one batch
def evaluate_one_batch(configs, batch, model, tokenizer):
    # 1 doc has 3 emotion clauses and 4 cause clauses at most, respectively
    # 1 emotion clause has 3 corresponding cause clauses at most, 1 cause clause has only 1 corresponding emotion clause
    with open('data/sentimental_clauses.pkl', 'rb') as f:
        emo_dictionary = pickle.load(f)

    section, discourse, word_count, doc_len, clause_len, ec_emotion_pos, ec_cause_pos, ce_cause_pos, ce_emotion_pos, ec_true_pairs, ce_true_pairs, conn = batch

    ec_emotion_pos = eval(ec_emotion_pos[0])
    ec_cause_pos = eval(ec_cause_pos[0])
    ce_cause_pos = eval(ce_cause_pos[0])
    ce_emotion_pos = eval(ce_emotion_pos[0])
    ec_true_pairs = eval(ec_true_pairs[0])
    ce_true_pairs = eval(ce_true_pairs[0])
    
    discourse_mask = torch.Tensor([1] * word_count + [0] * (512 - word_count)).to(torch.int32)
    segment_mask = torch.Tensor([0] * 512).to(torch.int32)

    query_len = 0
    clause_len = eval(clause_len[0])
    discourse_adj = torch.ones([doc_len, doc_len])  # batch size = 1
    
    # eval by emotion cause
    emo_ans = torch.zeros(doc_len)
    for pos in ec_emotion_pos:
        emo_ans[int(pos) - 1] = 1
    emo_ans_mask = torch.ones(doc_len)  # batch size = 1

    pair_count = len(ec_emotion_pos) * (doc_len - len(ec_emotion_pos) + 1)

    emo_cau_ans = torch.zeros(len(ec_emotion_pos) * doc_len)
    for i in range(len(ec_emotion_pos)):
        for j in range(len(ec_cause_pos[i])):
            emo_cau_ans[int(doc_len) * i + ec_cause_pos[i][j] - 1] = 1
    emo_cau_ans_mask = torch.ones(len(ec_emotion_pos) * doc_len)

    # due to batch = 1, dim0 = 1
    true_emo = ec_emotion_pos
    true_cau = []
    for emo in ec_cause_pos:
        for cau in emo:
            if cau not in true_cau:
                true_cau.append(cau)

    # Init
    pred_emo = []
    pred_cau = []
    pred_pair = []
    pred_pair_pro = []
    pred_emo_single = []
    pred_cau_single = []
    
    section = str(section.item())

    # step 1
    emo_pred = model(discourse, discourse_mask.unsqueeze(0), segment_mask.unsqueeze(0), query_len, clause_len, ec_emotion_pos, ec_cause_pos, ce_cause_pos, ce_emotion_pos, doc_len, discourse_adj, conn, 'ec_emo')
    emo_ans_mask = emo_ans_mask.to(DEVICE)
    temp_emo_prob = emo_pred.masked_select(emo_ans_mask.bool()).cpu().numpy().tolist()
    for idx in range(len(temp_emo_prob)):
        if temp_emo_prob[idx] > 0.9 or (temp_emo_prob[idx] > 0.5 and idx + 1 in emo_dictionary[section]):
            pred_emo.append(idx)
            pred_emo_single.append(idx + 1)
    ec_pred_emotion_pos = pred_emo
    if ec_pred_emotion_pos == []:
        ec_pred_emotion_pos = [0]

    # step 2
    cau_pred = model(discourse, discourse_mask.unsqueeze(0), segment_mask.unsqueeze(0), query_len, clause_len, ec_pred_emotion_pos, ec_cause_pos, ce_cause_pos, ce_emotion_pos, doc_len, discourse_adj, conn, 'ec_emo_cau')  
    temp_cau_prob = cau_pred[0].cpu().numpy().tolist()
    for idx_emo in pred_emo:
        for idx_cau in range(doc_len):
            if temp_cau_prob[pred_emo.index(idx_emo) * doc_len + idx_cau] > 0.5 and abs(idx_emo - idx_cau) <= 11:
                if idx_cau + 1 not in pred_cau_single:
                    pred_cau.append(idx_cau)
                    pred_cau_single.append(idx_cau + 1)
                prob_t = temp_emo_prob[idx_emo] * temp_cau_prob[idx_cau]
                if idx_cau - idx_emo >= 0 and idx_cau - idx_emo <= 2:
                    pass
                else:
                    prob_t *= 0.9
                pred_pair_pro.append(prob_t)
                pred_pair.append([idx_emo + 1, idx_cau + 1])

    pred_emo_final = []
    pred_cau_final = []
    pred_pair_final = []

    for i, pair in enumerate(pred_pair):
        if pred_pair_pro[i] > 0.5:
            pred_pair_final.append(pair)

    for pair in pred_pair_final:
        if pair[0] not in pred_emo_final:
            pred_emo_final.append(pair[0])
        if pair[1] not in pred_cau_final:
            pred_cau_final.append(pair[1])

    metric_e, metric_c, metric_p = \
        cal_metric(pred_emo_final, true_emo, pred_cau_final, true_cau, pred_pair_final, ec_true_pairs, doc_len)
    return metric_e, metric_c, metric_p


# evaluate step
def evaluate(configs, test_loader, model, tokenizer):
    model.eval()
    all_emo, all_cau, all_pair = [0, 0, 0], [0, 0, 0], [0, 0, 0]
    for batch in test_loader:
        emo, cau, pair = evaluate_one_batch(configs, batch, model, tokenizer)
        for i in range(3):
            all_emo[i] += emo[i]
            all_cau[i] += cau[i]
            all_pair[i] += pair[i]

    eval_emo = eval_func(all_emo)
    eval_cau = eval_func(all_cau)
    eval_pair = eval_func(all_pair)
    return eval_emo, eval_cau, eval_pair


def main(configs, train_loader, test_loader, tokenizer):
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True

    # model
    model = Network(configs).to(DEVICE)
    # optimizer
    params = list(model.named_parameters())
    optimizer_grouped_params = [
        {'params': [p for n, p in params if '_bert' in n], 'weight_decay': 0.01},
        {'params': [p for n, p in params if '_bert' not in n], 'lr': configs.lr, 'weight_decay': 0.01}
    ]
    optimizer = AdamW(params=optimizer_grouped_params, lr=configs.tuning_bert_rate)

    # scheduler
    training_steps = configs.epochs * len(train_loader) // configs.gradient_accumulation_steps
    warmup_steps = int(training_steps * configs.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=training_steps)

    # training
    model.zero_grad()
    max_result_pair, max_result_emo, max_result_cau = None, None, None
    max_result_emos, max_result_caus = None, None
    early_stop_flag = 0

    for epoch in range(1, configs.epochs+1):
        for train_step, batch in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            
            section, discourse, word_count, doc_len, clause_len, ec_emotion_pos, ec_cause_pos, ce_cause_pos, ce_emotion_pos, ec_true_pairs, ce_true_pairs, conn = batch

            ec_emotion_pos = eval(ec_emotion_pos[0])
            ec_cause_pos = eval(ec_cause_pos[0])
            ce_cause_pos = eval(ce_cause_pos[0])
            ce_emotion_pos = eval(ce_emotion_pos[0])
            
            discourse_mask = torch.Tensor([1] * word_count + [0] * (512 - word_count)).to(torch.int32)
            segment_mask = torch.Tensor([0] * 512).to(torch.int32)

            query_len = 0
            clause_len = eval(clause_len[0])
            discourse_adj = torch.ones([doc_len, doc_len])  # batch size = 1
            
            # emotion cause
            ec_emo_ans = torch.zeros(doc_len)
            for pos in ec_emotion_pos:
                ec_emo_ans[int(pos) - 1] = 1
            ec_emo_ans_mask = torch.ones(doc_len)  # batch size = 1

            ec_pair_count = len(ec_emotion_pos) * doc_len

            ec_emo_cau_ans = torch.zeros(len(ec_emotion_pos) * doc_len)
            for i in range(len(ec_emotion_pos)):
                for j in range(len(ec_cause_pos[i])):
                    ec_emo_cau_ans[int(doc_len) * i + ec_cause_pos[i][j] - 1] = 1
            ec_emo_cau_ans_mask = torch.ones(len(ec_emotion_pos) * doc_len)

            # cause emotion
            ce_cau_ans = torch.zeros(doc_len)
            for pos in ce_cause_pos:
                ce_cau_ans[int(pos) - 1] = 1
            ce_cau_ans_mask = torch.ones(doc_len)  # batch size = 1

            ce_pair_count = len(ce_cause_pos) * doc_len

            ce_cau_emo_ans = torch.zeros(len(ce_cause_pos) * doc_len)
            for i in range(len(ce_cause_pos)):
                for j in range(len(ce_emotion_pos[i])):
                    ce_cau_emo_ans[int(doc_len) * i + ce_emotion_pos[i][j] - 1] = 1
            ce_cau_emo_ans_mask = torch.ones(len(ce_cause_pos) * doc_len)

            ec_emo_pred = model(discourse, discourse_mask.unsqueeze(0), segment_mask.unsqueeze(0), query_len, clause_len, ec_emotion_pos, ec_cause_pos, ce_cause_pos, ce_emotion_pos, doc_len, discourse_adj, conn, 'ec_emo')
            ec_emo_cau_pred = model(discourse, discourse_mask.unsqueeze(0), segment_mask.unsqueeze(0), query_len, clause_len, ec_emotion_pos, ec_cause_pos, ce_cause_pos, ce_emotion_pos, doc_len, discourse_adj, conn, 'ec_emo_cau')
            ce_cau_pred = model(discourse, discourse_mask.unsqueeze(0), segment_mask.unsqueeze(0), query_len, clause_len, ec_emotion_pos, ec_cause_pos, ce_cause_pos, ce_emotion_pos, doc_len, discourse_adj, conn, 'ce_cau')
            ce_cau_emo_pred = model(discourse, discourse_mask.unsqueeze(0), segment_mask.unsqueeze(0), query_len, clause_len, ec_emotion_pos, ec_cause_pos, ce_cause_pos, ce_emotion_pos, doc_len, discourse_adj, conn, 'ce_cau_emo')
            
            loss_ec_emo = model.pre_loss(ec_emo_pred, ec_emo_ans, ec_emo_ans_mask)
            loss_ec_emo_cau = model.pre_loss(ec_emo_cau_pred, ec_emo_cau_ans, ec_emo_cau_ans_mask)
            loss_ce_cau = model.pre_loss(ce_cau_pred, ce_cau_ans, ce_cau_ans_mask)
            loss_ce_cau_emo = model.pre_loss(ce_cau_emo_pred, ce_cau_emo_ans, ce_cau_emo_ans_mask)
            
            loss = loss_ec_emo + loss_ec_emo_cau + loss_ce_cau + loss_ce_cau_emo
            loss.backward()

            if train_step % configs.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            if train_step % 1 == 0:
                print('epoch: {}, step: {}, loss_emo: {}, loss_emo_cau: {}, loss: {}'
                    .format(epoch, train_step, loss_ec_emo, loss_ec_emo_cau, loss))
        
        with torch.no_grad():
            eval_emo, eval_cau, eval_pair = evaluate(configs, test_loader, model, tokenizer)
            
            if max_result_pair is None or eval_pair[0] > max_result_pair[0]:
                early_stomax_result_pairp_flag = 1
                max_result_emo = eval_emo
                max_result_cau = eval_cau
                max_result_pair = eval_pair
    
                state_dict = {'model': model.state_dict(), 'result': max_result_pair}
                torch.save(state_dict, 'model/model.pth')
            else:
                early_stop_flag += 1
        if epoch > configs.epochs / 2 and early_stop_flag >= 7:
            break

    return max_result_emo, max_result_cau, max_result_pair


if __name__ == '__main__':
    configs = Config()
    device = DEVICE
    tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
    model = Network(configs).to(DEVICE)
    
    train_dataset = Discourse(tokenizer, configs.train_dataset_path)
    train_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=1,drop_last=True)
    test_dataset = Discourse(tokenizer, configs.test_dataset_path)
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1)
    
    max_result_emo, max_result_cau, max_result_pair = main(configs, train_loader, test_loader, tokenizer)
    print('max_result_emo: {}, max_result_cau: {}, max_result_pair: {}'.format(max_result_emo, max_result_cau, max_result_pair))