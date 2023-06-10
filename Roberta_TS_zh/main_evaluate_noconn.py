import os
import csv
from config import *
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer, AutoTokenizer
from model_noconn import Network
import datetime
import numpy as np
import pandas as pd
import pickle
from utils.utils import *


# dataset
class Discourse(Dataset):
    def __init__(self, tokenizer, path, start1, end1, start2=0, end2=0):
        self.tokenizer = tokenizer
        self.data_path = path
        self.section = []
        self.discourse = []
        self.word_count = []
        self.doc_len = []
        self.clause_len = []
        self.ec_emotion_pos = []
        self.ec_cause_pos = []
        self.ce_cause_pos = []
        self.ce_emotion_pos = []
        self.ec_true_pairs = []
        self.ce_true_pairs = []
        
        df = pd.read_csv(self.data_path)
        
        for i in range(len(df)):
            if (int(df['section'][i]) >= start1 and int(df['section'][i]) < end1) or (int(df['section'][i]) >= start2 and int(df['section'][i]) < end2):
                pass
            else:
                continue
            self.section.append(int(df['section'][i]))
            self.discourse.append(torch.Tensor(self.tokenizer(df['discourse'][i],padding='max_length',max_length=512)['input_ids']).to(torch.int32))
            self.word_count.append(int(df['word_count'][i]))
            self.doc_len.append(int(df['doc_len'][i]))
            self.clause_len.append(eval(df['clause_len'][i]))
            self.ec_emotion_pos.append(eval(df['ec_emotion_pos'][i]))
            self.ec_cause_pos.append(eval(df['ec_cause_pos'][i]))
            self.ce_cause_pos.append(eval(df['ce_cause_pos'][i]))
            self.ce_emotion_pos.append(eval(df['ce_emotion_pos'][i]))
            self.ec_true_pairs.append(eval(df['ec_true_pairs'][i]))
            self.ce_true_pairs.append(eval(df['ce_true_pairs'][i]))
            
    def __getitem__(self, index):
        return self.section[index], self.discourse[index], self.word_count[index], self.doc_len[index], self.clause_len[index], self.ec_emotion_pos[index], self.ec_cause_pos[index], self.ce_cause_pos[index], self.ce_emotion_pos[index], self.ec_true_pairs[index], self.ce_true_pairs[index]

    def __len__(self):
        return len(self.section)


# evaluate one batch
def evaluate_one_batch(configs, batch, model, tokenizer):
    # 1 doc has 3 emotion clauses and 4 cause clauses at most, respectively
    # 1 emotion clause has 3 corresponding cause clauses at most, 1 cause clause has only 1 corresponding emotion clause
    # set emotion slot to 8 for padding
    
    section, discourse, word_count, doc_len, clause_len, ec_emotion_pos, ec_cause_pos, ce_cause_pos, ce_emotion_pos, ec_true_pairs, ce_true_pairs, discourse_mask, segment_mask, query_len, ec_emo_ans, ec_emo_ans_mask, ec_emo_cau_ans, ec_emo_cau_ans_mask, ce_cau_ans, ce_cau_ans_mask, ce_cau_emo_ans, ce_cau_emo_ans_mask, ec_pair_count, ce_pair_count, discourse_adj = batch
    
    # change batch_size to len(section) for unification
    batch_size = len(section)
    
    # load emo_dictionary
    with open('../data/sentimental_clauses_zh.pkl', 'rb') as f:
        emo_dictionary = pickle.load(f)

    pred_emos = []
    pred_caus = []
    pred_pairs = []
    pred_pairs_pro = []
    pred_emos_single = []
    pred_caus_single = []

    true_emos = []
    true_caus = []
    true_pairs = ec_true_pairs
    for i in range(batch_size):
        true_emos.append(ec_emotion_pos[i])
        cause_pos = ec_cause_pos[i]
        cau_list = []
        for emo_index in range(len(cause_pos)):
            for cau in cause_pos[emo_index]:
                if cau not in cau_list:
                    cau_list.append(cau)
        true_caus.append(cau_list)

    # emotion step
    emo_pred = model(discourse, discourse_mask, segment_mask, query_len, clause_len, ec_emotion_pos, ec_cause_pos, ce_cause_pos, ce_emotion_pos, doc_len, discourse_adj, 'emo')
    ec_emo_ans_mask = ec_emo_ans_mask.to(DEVICE)
    temp_emos_prob = []
    for i in range(batch_size):
        temp_emos_prob.append(emo_pred[i].masked_select(ec_emo_ans_mask[i].bool()).cpu().numpy().tolist())
    pred_emotion_pos = []
    for i in range(batch_size):
        pred_emo = []
        pred_emo_single = []
        for idx in range(len(temp_emos_prob[i])):
            if temp_emos_prob[i][idx] > 0.99 or (temp_emos_prob[i][idx] > 0.5 and idx + 1 in emo_dictionary[str(section[i])]):
                pred_emo.append(idx)
                pred_emo_single.append(idx + 1)

        pred_emo_pos = pred_emo_single
        if pred_emo_pos == []:
            pred_emo_pos = [1]
        pred_emos.append(pred_emo)
        pred_emos_single.append(pred_emo_single)
        pred_emotion_pos.append(pred_emo_pos)

    print(section)
    print(pred_emos)

    # emotion cause step
    cau_pred = model(discourse, discourse_mask, segment_mask, query_len, clause_len, pred_emotion_pos, ec_cause_pos, ce_cause_pos, ce_emotion_pos, doc_len, discourse_adj, 'emo_cau')  
    temp_caus_prob = cau_pred.cpu().numpy().tolist()
    for i in range(batch_size):
        pred_cau = []
        pred_cau_single = []
        pred_pair = []
        pred_pair_pro = []
        for idx_emo in pred_emos[i]:
            for idx_cau in range(doc_len[i]):
                if temp_caus_prob[i][pred_emos[i].index(idx_emo) * doc_len[i] + idx_cau] > 0.5 and abs(idx_emo - idx_cau) <= 11:
                    if idx_cau + 1 not in pred_cau_single:
                        pred_cau.append(idx_cau)
                        pred_cau_single.append(idx_cau + 1)
                    prob_t = temp_emos_prob[i][idx_emo] * temp_caus_prob[i][pred_emos[i].index(idx_emo) * doc_len[i] + idx_cau]
                    if idx_cau - idx_emo >= 0 and idx_cau - idx_emo <= 2:
                        pass
                    else:
                        prob_t *= 0.9
                    pred_pair_pro.append(prob_t)
                    pred_pair.append([idx_emo + 1, idx_cau + 1])
        
        pred_caus.append(pred_cau)
        pred_caus_single.append(pred_cau_single)
        pred_pairs.append(pred_pair)
        pred_pairs_pro.append(pred_pair_pro)

    pred_emos_final = []
    pred_caus_final = []
    pred_pairs_final = []

    for i in range(batch_size):
        pred_pair_final = []
        for idx, pair in enumerate(pred_pairs[i]):
            if pred_pairs_pro[i][idx] > 0.5:
                pred_pair_final.append(pair)
        pred_pairs_final.append(pred_pair_final)
    
    print(pred_pairs_final)
    
    for i in range(batch_size):
        pred_emo_final = []
        pred_cau_final = []
        for pair in pred_pairs_final[i]:
            if pair[0] not in pred_emo_final:
                pred_emo_final.append(pair[0])
            if pair[1] not in pred_cau_final:
                pred_cau_final.append(pair[1])
        pred_emos_final.append(pred_emo_final)
        pred_caus_final.append(pred_cau_final)

    metric_e, metric_c, metric_p = [0, 0, 0], [0, 0, 0], [0, 0, 0]
    
    for i in range(batch_size):
        e, c, p = cal_metric(pred_emos_final[i], true_emos[i], pred_caus_final[i], true_caus[i], pred_pairs_final[i], true_pairs[i], doc_len[i])
        metric_e, metric_c, metric_p = [item1 + item2 for item1, item2 in zip(metric_e, e)], [item1 + item2 for item1, item2 in zip(metric_c, c)], [item1 + item2 for item1, item2 in zip(metric_p, p)]
    return metric_e, metric_c, metric_p


# evaluate step
def evaluate(configs, test_loader, model, tokenizer):
    model.eval()
    all_emo, all_cau, all_pair = [0, 0, 0], [0, 0, 0], [0, 0, 0]
    for batch in test_loader:
        emo, cau, pair = evaluate_one_batch(configs, batch, model, tokenizer)
        print(emo, cau, pair)
        for i in range(3):
            all_emo[i] += emo[i]
            all_cau[i] += cau[i]
            all_pair[i] += pair[i]

    print(all_emo, all_cau, all_pair)
    
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
    model.load_state_dict(torch.load('model/model_noconn.pth')['model'])

    # evaluate
    model.eval()
    with torch.no_grad():
        eval_emo, eval_cau, eval_pair = evaluate(configs, test_loader, model, tokenizer)
        
    print(eval_emo, eval_cau, eval_pair)
            
    return None


def my_collate_fn(batch):
    configs = Config()
    
    # unzip batch
    batch = zip(*batch)
    section, discourse, word_count, doc_len, clause_len, ec_emotion_pos, ec_cause_pos, ce_cause_pos, ce_emotion_pos, ec_true_pairs, ce_true_pairs = batch
    
    # change batch_size to len(section) for unification
    batch_size = len(section)

    # save for possible query
    query_len = (0,) * batch_size
    
    max_doc_len = max(doc_len)
    
    # mask for real words
    discourse_mask = torch.zeros(batch_size, 512)
    for i in range(batch_size):
        discourse_mask[i,:] =  torch.Tensor([1] * word_count[i] + [0] * (512 - word_count[i])).to(torch.int32)
    
    # mask for segment
    segment_mask = torch.zeros(batch_size, 512)
    for i in range(batch_size):
        segment_mask[i,:] =  torch.Tensor([0] * query_len[i] + [1] * (512 - query_len[i])).to(torch.int32)
    
    # mask for real clauses
    discourse_adj = torch.zeros(batch_size, max_doc_len, max_doc_len)
    for i in range(batch_size):
        discourse_adj[i, :doc_len[i], :doc_len[i]] =  1

    # emotion cause
    # 1 doc has 3 emotion clauses and 4 cause clauses at most, respectively
    # 1 emotion clause has 3 corresponding cause clauses at most, 1 cause clause has only 1 corresponding emotion clause
    # set emotion slot to 8 for padding
    
    # emotion answer and mask
    ec_emo_ans = []
    ec_emo_ans_mask = []
    for i in range(batch_size):
        ec_emo_ans.append(torch.zeros(max_doc_len))
        ec_emo_ans_mask.append(torch.zeros(max_doc_len))
        ec_emo_ans_mask[i][:doc_len[i]] = 1
        for pos in ec_emotion_pos[i]:
            ec_emo_ans[i][pos - 1] = 1

    # count ec_pairs
    ec_pair_count = []
    for i in range(batch_size):
        ec_pair_count.append(len(ec_emotion_pos[i]) * doc_len[i])

    # emotion cause answer and mask
    # note: ec_emo_cau_ans and ans_mask change during evaluate and need another initialization
    ec_emo_cau_ans = []
    ec_emo_cau_ans_mask = []
    for i in range(batch_size):
        ec_emo_cau_ans.append(torch.zeros(8 * max_doc_len))
        ec_emo_cau_ans_mask.append(torch.zeros(8 * max_doc_len))
        ec_emo_cau_ans_mask[i][:len(ec_emotion_pos[i]) * doc_len[i]] = 1
        for emo_index in range(len(ec_cause_pos[i])):
            for pos in ec_cause_pos[i][emo_index]:
                ec_emo_cau_ans[i][doc_len[i] * emo_index + pos - 1] = 1

    # cause emotion
    # 1 doc has 3 emotion clauses and 4 cause clauses at most, respectively
    # 1 emotion clause has 3 corresponding cause clauses at most, 1 cause clause has only 1 corresponding emotion clause
    # set cause slot to 8 for padding
    
    # cause answer and mask
    ce_cau_ans = []
    ce_cau_ans_mask = []
    for i in range(batch_size):
        ce_cau_ans.append(torch.zeros(max_doc_len))
        ce_cau_ans_mask.append(torch.zeros(max_doc_len))
        ce_cau_ans_mask[i][:doc_len[i]] = 1
        for pos in ce_cause_pos[i]:
            ce_cau_ans[i][pos - 1] = 1

    # count ce_pairs
    ce_pair_count = []
    for i in range(batch_size):
        ce_pair_count.append(len(ce_cause_pos[i]) * doc_len[i])

    # cause emotion answer and mask
    # note: ce_cau_emo_ans and ans_mask change during evaluate and need another initialization
    ce_cau_emo_ans = []
    ce_cau_emo_ans_mask = []
    for i in range(batch_size):
        ce_cau_emo_ans.append(torch.zeros(8 * max_doc_len))
        ce_cau_emo_ans_mask.append(torch.zeros(8 * max_doc_len))
        ce_cau_emo_ans_mask[i][:len(ce_cause_pos[i]) * doc_len[i]] = 1
        for cau_index in range(len(ce_emotion_pos[i])):
            for pos in ce_emotion_pos[i][cau_index]:
                ce_cau_emo_ans[i][doc_len[i] * cau_index + pos - 1] = 1
    
    discourse = torch.tensor([item.tolist() for item in discourse]).to(torch.int32)
    discourse_mask = torch.tensor([item.tolist() for item in discourse_mask]).to(torch.int32)
    segment_mask = torch.tensor([item.tolist() for item in segment_mask]).to(torch.int32)
    ec_emo_ans = torch.tensor([item.tolist() for item in ec_emo_ans]).to(torch.int32)
    ec_emo_ans_mask = torch.tensor([item.tolist() for item in ec_emo_ans_mask]).to(torch.int32)
    ec_emo_cau_ans = torch.tensor([item.tolist() for item in ec_emo_cau_ans]).to(torch.int32)
    ec_emo_cau_ans_mask = torch.tensor([item.tolist() for item in ec_emo_cau_ans_mask]).to(torch.int32)
    ce_cau_ans = torch.tensor([item.tolist() for item in ce_cau_ans]).to(torch.int32)
    ce_cau_ans_mask = torch.tensor([item.tolist() for item in ce_cau_ans_mask]).to(torch.int32)
    ce_cau_emo_ans = torch.tensor([item.tolist() for item in ce_cau_emo_ans]).to(torch.int32)
    ce_cau_emo_ans_mask = torch.tensor([item.tolist() for item in ce_cau_emo_ans_mask]).to(torch.int32)
    discourse_adj = torch.tensor([item.tolist() for item in discourse_adj])

    return section, discourse, word_count, doc_len, clause_len, ec_emotion_pos, ec_cause_pos, ce_cause_pos, ce_emotion_pos, ec_true_pairs, ce_true_pairs, discourse_mask, segment_mask, query_len, ec_emo_ans, ec_emo_ans_mask, ec_emo_cau_ans, ec_emo_cau_ans_mask, ce_cau_ans, ce_cau_ans_mask, ce_cau_emo_ans, ce_cau_emo_ans_mask, ec_pair_count, ce_pair_count, discourse_adj


if __name__ == '__main__':
    configs = Config()
    device = DEVICE
    
    tokenizer = AutoTokenizer.from_pretrained(configs.roberta_cache_path)
    
    fold = configs.fold
    i = configs.fold_id
    dataset_len = configs.dataset_len
    
    train_loader = None
    
    test_start = int(i / fold * dataset_len) + 1
    test_end = int((i + 1) / fold * dataset_len) + 1
    test_dataset = Discourse(tokenizer, configs.dataset_path, test_start, test_end)
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=configs.batch_size, collate_fn=my_collate_fn, drop_last=True)
    
    main(configs, train_loader, test_loader, tokenizer)