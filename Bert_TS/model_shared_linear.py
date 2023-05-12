from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from config import DEVICE
from gnn_layer import GraphAttentionLayer
from config import *

# Init param
cause_uniconn = []
noncause_uniconn = []
candidate_conn = []
with open ('data/cause_uniconn_modified.txt', 'r', encoding='utf-8') as f:
    line = f.readline()
    while line:
        for word in line.split(','):
            cause_uniconn.append(word)
        line = f.readline()
with open ('data/noncause_uniconn.txt', 'r', encoding='utf-8') as f:
    line = f.readline()
    while line:
        for word in line.split(','):
            noncause_uniconn.append(word)
        line = f.readline()
with open ('data/uniconn_modified.txt', 'r', encoding='utf-8') as f:
    line = f.readline()
    while line:
        for word in line.split(','):
            candidate_conn.append(word)
        line = f.readline()

class Network(nn.Module):
    def __init__(self, configs):
        super(Network, self).__init__()
        self.bert_encoder = BertEncoder(configs)
        self.gnn = GraphNN(configs)
        self.ec_pred_emo = Predictions_ec_emo(configs)
        self.ec_pred_emo_cau = Predictions_ec_emo_cau(configs)
        self.ce_pred_cau = Predictions_ce_cau(configs)
        self.ce_pred_cau_emo = Predictions_ce_cau_emo(configs)

    def forward(self, query, query_mask, query_seg, query_len, clause_len, ec_emotion_pos, ec_cause_pos, ce_cause_pos, ce_emotion_pos, doc_len, adj, conn, q_type):
        # shape: batch_size, max_doc_len, 1024
        doc_sents_h = self.bert_encoder(query, query_mask, query_seg, query_len, clause_len, doc_len)
        doc_sents_h = self.gnn(doc_sents_h, doc_len, adj)
        ec_pair_count = len(ec_emotion_pos) * doc_len
        ce_pair_count = len(ce_cause_pos) * doc_len
        pred_ec_emo = self.ec_pred_emo(doc_sents_h)
        pred_ec_emo_cau = self.ec_pred_emo_cau(doc_sents_h, query, ec_emotion_pos, doc_len, clause_len)
        pred_ce_cau = self.ce_pred_cau(doc_sents_h)
        pred_ce_cau_emo = self.ce_pred_cau_emo(doc_sents_h, query, ce_cause_pos, doc_len, clause_len)
        if q_type == 'ec_emo':
            return pred_ec_emo
        if q_type == 'ec_emo_cau':
            return pred_ec_emo_cau
        if q_type == 'ce_cau':
            return pred_ce_cau
        if q_type == 'ce_cau_emo':
            return pred_ce_cau_emo
        return None

    def pre_loss(self, pred, true, mask):
        true = torch.FloatTensor(true.float()).to(DEVICE)  # shape: batch_size, seq_len
        mask = torch.BoolTensor(mask.bool()).to(DEVICE)
        pred = pred.masked_select(mask)
        true = true.masked_select(mask)
        # weight = torch.where(true > 0.5, 2, 1)
        criterion = nn.BCELoss()
        return criterion(pred, true)


class BertEncoder(nn.Module):
    def __init__(self, configs):
        super(BertEncoder, self).__init__()
        hidden_size = configs.feat_dim
        self.bert = BertModel.from_pretrained(configs.bert_cache_path).to(DEVICE)
        self.tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
        self.fc = nn.Linear(768, 1)

    def forward(self, discourse, discourse_mask, segment_mask, query_len, clause_len, doc_len):
        hidden_states = self.bert(input_ids=discourse.to(DEVICE),
                                  attention_mask=discourse_mask.to(DEVICE),
                                  token_type_ids=segment_mask.to(DEVICE))[0]
        hidden_states, mask_doc = self.get_sentence_state(hidden_states, query_len, clause_len, doc_len)

        alpha = self.fc(hidden_states).squeeze(-1)  # bs, max_doc_len, max_seq_len
        mask_doc = 1 - mask_doc # bs, max_doc_len, max_seq_len
        alpha.data.masked_fill_(mask_doc.bool(), -9e5)
        alpha = F.softmax(alpha, dim=-1).unsqueeze(-1).repeat(1, 1, 1, hidden_states.size(-1))
        hidden_states = torch.sum(alpha * hidden_states, dim=2) # bs, max_doc_len, 768

        return hidden_states.to(DEVICE)

    def get_sentence_state(self, hidden_states, query_lens, clause_lens, doc_len):
        # 对文档的每个句子的token做注意力，得到每个句子的向量表示
        sentence_state_all = []
        mask_all = []
        max_clause_len = 0
        clause_lens = [clause_lens]

        for clause_len in clause_lens: # 找出最长的一句话包含多少token
            for l in clause_len:
                max_clause_len = max(max_clause_len, l)

        max_doc_len = max(doc_len) # 最长的文档包含多少句子
        for i in range(hidden_states.size(0)):  # 对每个batch
            # 对文档sentence
            mask = []
            begin = 0
            sentence_state = []
            for clause_len in clause_lens[i]:
                sentence = hidden_states[i, begin: begin + clause_len]
                begin += clause_len
                if sentence.size(0) < max_clause_len:
                    sentence = torch.cat([sentence, torch.zeros((max_clause_len - clause_len, sentence.size(-1))).to(DEVICE)],
                                         dim=0)
                sentence_state.append(sentence.unsqueeze(0))
                mask.append([1] * clause_len + [0] * (max_clause_len - clause_len))
            sentence_state = torch.cat(sentence_state, dim=0).to(DEVICE)
            if sentence_state.size(0) < max_doc_len:
                mask.extend([[0] * max_clause_len] * (max_doc_len - sentence_state.size(0)))
                padding = torch.zeros(
                    (max_doc_len - sentence_state.size(0), sentence_state.size(-2), sentence_state.size(-1)))
                sentence_state = torch.cat([sentence_state, padding.to(DEVICE)], dim=0)
            sentence_state_all.append(sentence_state.unsqueeze(0))
            mask_all.append(mask)
        sentence_state_all = torch.cat(sentence_state_all, dim=0).to(DEVICE)
        mask_all = torch.tensor(mask_all).to(DEVICE)
        return sentence_state_all, mask_all


class GraphNN(nn.Module):
    def __init__(self, configs):
        super(GraphNN, self).__init__()
        in_dim = configs.feat_dim
        self.gnn_dims = [in_dim] + [int(dim) for dim in configs.gnn_dims.strip().split(',')]  # [1024, 256]

        self.gnn_layers = len(self.gnn_dims) - 1
        self.att_heads = [int(att_head) for att_head in configs.att_heads.strip().split(',')] # [4]
        self.gnn_layer_stack = nn.ModuleList()
        for i in range(self.gnn_layers):
            in_dim = self.gnn_dims[i] * self.att_heads[i - 1] if i != 0 else self.gnn_dims[i]
            self.gnn_layer_stack.append(
                GraphAttentionLayer(self.att_heads[i], in_dim, self.gnn_dims[i + 1], configs.dp)
            )

    def forward(self, doc_sents_h, doc_len, adj):
        batch, max_doc_len, _ = doc_sents_h.size()
        assert max(doc_len) == max_doc_len
        for i, gnn_layer in enumerate(self.gnn_layer_stack):
            doc_sents_h = gnn_layer(doc_sents_h, adj)
        return doc_sents_h


class Predictions_ec_emo(nn.Module):
    def __init__(self, configs):
        super(Predictions_ec_emo, self).__init__()
        self.feat_dim = 768
        self.out_emo = nn.Linear(768, 1).to(DEVICE)

    def forward(self, doc_sents_h):
        pred_emo = self.out_emo(doc_sents_h).squeeze(-1)  # bs, max_doc_len, 1
        pred_emo = torch.sigmoid(pred_emo)
        return pred_emo # shape: bs ,max_doc_len


class Predictions_ec_emo_cau(nn.Module):
    def __init__(self, configs):
        super(Predictions_ec_emo_cau, self).__init__()
        self.feat_dim = 768
        self.linear_layer = nn.Linear(3, 1)
        self.out_cau = nn.Linear(768, 1).to(DEVICE)
        self.tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
        self.bert = BertModel.from_pretrained(configs.bert_cache_path)
        self.bert.eval()
        self.bert.to(DEVICE)
 
    def forward(self, doc_sents_h, discourse, emotion_pos, doc_len, clause_len):
        doc_sents_h_2d = doc_sents_h.squeeze(0)  # shape: batch_size=1, max_doc_len, 768, squeeze dim0

        # Init pairs_h
        pairs_h = torch.tensor([]).to(DEVICE)
        clause_start = [0]
        for i in range(len(clause_len)):
            clause_start.append(clause_start[i] + clause_len[i])
        for i in range(len(emotion_pos)):
            for j in range(doc_len):
                arg1 = discourse[0][clause_start[emotion_pos[i] - 1]:clause_start[emotion_pos[i]]]
                arg2 = discourse[0][clause_start[j]:clause_start[j + 1]]
                len1 = clause_len[emotion_pos[i] - 1]
                len2 = clause_len[j]
                sep_token = torch.tensor([102])
                mask_token = torch.tensor([103])
                inputs = torch.cat([arg1, sep_token, mask_token, arg2])
                inputs = F.pad(inputs, (0, 512 - inputs.size(-1)), 'constant', 0).unsqueeze(0)
                mask = torch.tensor([1] * (len1 + 2 + len2)+ [0] * (510 - len1 - len2)).unsqueeze(0)
                segement = torch.tensor([0] * (len1 + 1) + [1] * (511 - len1)).unsqueeze(0)
        
                # Get connective embedding
                with torch.no_grad():
                    conn_embedding = self.bert(inputs.to(DEVICE), mask.to(DEVICE), segement.to(DEVICE))[0][0][len1 + 1]
                
                # Stack three embeddings for one pair presentation
                pair_h = torch.stack([doc_sents_h_2d[emotion_pos[i] - 1], conn_embedding, doc_sents_h_2d[j]], dim=-1)
                pair_h = self.linear_layer(pair_h).unsqueeze(-1)
                
                # Concatenate pairs for whole doc answer
                if pairs_h == torch.Size([]):
                    pairs_h = pair_h
                else:
                    pairs_h = torch.concatenate([pairs_h, pair_h], dim=-1)
        
        pairs_h = torch.permute(pairs_h, (1,2,0))
        pred_emo_cau = self.out_cau(pairs_h)  # bs, max_doc_len, 1
        pred_emo_cau = pred_emo_cau.squeeze(-1)
        pred_emo_cau = torch.sigmoid(pred_emo_cau)
        return pred_emo_cau # shape: bs , emo_num * max_doc_len


class Predictions_ce_cau(nn.Module):
    def __init__(self, configs):
        super(Predictions_ce_cau, self).__init__()
        self.feat_dim = 768
        self.out_cau = nn.Linear(768, 1).to(DEVICE)

    def forward(self, doc_sents_h):
        pred_cau = self.out_cau(doc_sents_h).squeeze(-1)  # bs, max_doc_len, 1
        pred_cau = torch.sigmoid(pred_cau)
        return pred_cau # shape: bs ,max_doc_len

class Predictions_ce_cau_emo(nn.Module):
    def __init__(self, configs):
        super(Predictions_ce_cau_emo, self).__init__()
        self.feat_dim = 768
        self.linear_layer = nn.Linear(3, 1)
        self.out_cau = nn.Linear(768, 1).to(DEVICE)
        self.tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
        self.bert = BertModel.from_pretrained(configs.bert_cache_path)
        self.bert.eval()
        self.bert.to(DEVICE)
 
    def forward(self, doc_sents_h, discourse, cause_pos, doc_len, clause_len):
        doc_sents_h_2d = doc_sents_h.squeeze(0)  # shape: batch_size=1, max_doc_len, 768, squeeze dim0

        # Init pairs_h
        pairs_h = torch.tensor([]).to(DEVICE)
        clause_start = [0]
        for i in range(len(clause_len)):
            clause_start.append(clause_start[i] + clause_len[i])
        for i in range(len(cause_pos)):
            for j in range(doc_len):
                arg1 = discourse[0][clause_start[cause_pos[i] - 1]:clause_start[cause_pos[i]]]
                arg2 = discourse[0][clause_start[j]:clause_start[j + 1]]
                len1 = clause_len[cause_pos[i] - 1]
                len2 = clause_len[j]
                sep_token = torch.tensor([102])
                mask_token = torch.tensor([103])
                inputs = torch.cat([arg1, sep_token, mask_token, arg2])
                inputs = F.pad(inputs, (0, 512 - inputs.size(-1)), 'constant', 0).unsqueeze(0)
                mask = torch.tensor([1] * (len1 + 2 + len2)+ [0] * (510 - len1 - len2)).unsqueeze(0)
                segement = torch.tensor([0] * (len1 + 1) + [1] * (511 - len1)).unsqueeze(0)
        
                # Get connective embedding
                with torch.no_grad():
                    conn_embedding = self.bert(inputs.to(DEVICE), mask.to(DEVICE), segement.to(DEVICE))[0][0][len1 + 1]
                
                # Stack three embeddings for one pair presentation
                pair_h = torch.stack([doc_sents_h_2d[cause_pos[i] - 1], conn_embedding, doc_sents_h_2d[j]], dim=-1)
                pair_h = self.linear_layer(pair_h).unsqueeze(-1)
                
                # Concatenate pairs for whole doc answer
                if pairs_h == torch.Size([]):
                    pairs_h = pair_h
                else:
                    pairs_h = torch.concatenate([pairs_h, pair_h], dim=-1)
        
        pairs_h = torch.permute(pairs_h, (1,2,0))
        pred_cau_emo = self.out_cau(pairs_h)  # bs, max_doc_len, 1
        pred_cau_emo = pred_cau_emo.squeeze(-1)
        pred_cau_emo = torch.sigmoid(pred_cau_emo)
        return pred_cau_emo # shape: bs , emo_num * max_doc_len