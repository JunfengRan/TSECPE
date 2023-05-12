from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from config import DEVICE
from gnn_layer import GraphAttentionLayer


class Network(nn.Module):
    def __init__(self, configs):
        super(Network, self).__init__()
        self.configs = configs
        self.feat_dim = self.configs.feat_dim
        self.batch_size = self.configs.batch_size
        
        self.bert_train = BertModel.from_pretrained(self.configs.bert_cache_path)
        self.bert_eval = BertModel.from_pretrained(self.configs.bert_cache_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.configs.bert_cache_path)
        
        # bert_encoder
        self.fc = nn.Linear(self.feat_dim, 1)
        
        # graph_gnn
        in_dim = self.feat_dim
        self.gnn_dims = [in_dim] + [int(dim) for dim in self.configs.gnn_dims.strip().split(',')]  # [1024, 256]

        self.gnn_layers = len(self.gnn_dims) - 1
        self.att_heads = [int(att_head) for att_head in self.configs.att_heads.strip().split(',')] # [4]
        self.gnn_layer_stack = nn.ModuleList()
        for i in range(self.gnn_layers):
            in_dim = self.gnn_dims[i] * self.att_heads[i - 1] if i != 0 else self.gnn_dims[i]
            self.gnn_layer_stack.append(
                GraphAttentionLayer(self.att_heads[i], in_dim, self.gnn_dims[i + 1], self.configs.dp)
            )
    
        # pred_emo
        self.out_emo = nn.Linear(self.feat_dim, 1)
        
        # pred_emo_cau
        self.out_emo_cau = nn.Linear(self.feat_dim * 3, 1)
        self.bert_eval.eval()
        self.bert_eval
    
    def forward(self, query, query_mask, query_seg, query_len, clause_len, emotion_pos, cause_pos, doc_len, adj, q_type):
        # shape: batch_size, max_doc_len, 512
        doc_sents_h = self.bert_encoder(query, query_mask, query_seg, query_len, clause_len, doc_len)
        doc_sents_h = self.graph_gnn(doc_sents_h, doc_len, adj)
        pred_emo = self.pred_emo(doc_sents_h)
        pred_emo_cau = self.pred_emo_cau(doc_sents_h, query, emotion_pos, doc_len, clause_len)
        if q_type == 'emo':
            return pred_emo
        if q_type == 'emo_cau':
            return pred_emo_cau
        
        return None

    def bert_encoder(self, discourse, discourse_mask, segment_mask, query_len, clause_len, doc_len):
        
        hidden_states = self.bert_train(input_ids=discourse.to(DEVICE),
                                  attention_mask=discourse_mask.to(DEVICE),
                                  token_type_ids=segment_mask.to(DEVICE))[0]
        hidden_states, mask_doc = self.get_sentence_state(hidden_states, query_len, clause_len, doc_len)

        alpha = self.fc(hidden_states).squeeze(-1)  # shape: batch_size, max_doc_len, max_seq_len
        mask_doc = 1 - mask_doc # shape: batch_size, max_doc_len, max_seq_len
        alpha.data.masked_fill_(mask_doc.bool(), -9e5)
        alpha = F.softmax(alpha, dim=-1).unsqueeze(-1).repeat(1, 1, 1, hidden_states.size(-1))
        hidden_states = torch.sum(alpha * hidden_states, dim=2) # shape: batch_size, max_doc_len, feat_dim

        return hidden_states.to(DEVICE)       

    def get_sentence_state(self, hidden_states, query_lens, clause_lens, doc_len):
        # get attention for tokens in each sentences
        # get sentence state
        sentence_state_all = []
        mask_all = []
        max_clause_len = 0

        for clause_len in clause_lens:  # max token number in a sentence
            for l in clause_len:
                max_clause_len = max(max_clause_len, l)

        max_doc_len = max(doc_len)  # max sentence number in a document
        for i in range(hidden_states.size(0)):  # for each batch
            # for each sentence
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

    def graph_gnn(self, doc_sents_h, doc_len, adj):
        batch, max_doc_len, _ = doc_sents_h.size()
        assert max(doc_len) == max_doc_len
        for i, gnn_layer in enumerate(self.gnn_layer_stack):
            doc_sents_h = gnn_layer(doc_sents_h, adj)
        return doc_sents_h

    def pred_emo(self, doc_sents_h):
        pred_emo = self.out_emo(doc_sents_h).squeeze(-1)  # shape: batch_size, max_doc_len, 1
        pred_emo = torch.sigmoid(pred_emo)
        return pred_emo # shape: batch_size, max_doc_len

    def pred_emo_cau(self, doc_sents_h, discourse, emotion_pos, doc_len, clause_len):
        # For each item in batch
        # shape: batch_size, max_doc_len, feat_dim
        pairs_hs = torch.tensor([]).to(DEVICE)
        max_doc_len = max(doc_len)
        for i in range(doc_sents_h.size(0)):
            # Init pairs_h
            pairs_h = torch.tensor([]).to(DEVICE)
            clause_start = [0]
            for j in range(len(clause_len[i])):
                clause_start.append(clause_start[j] + clause_len[i][j])
            for j in range(len(emotion_pos[i])):
                for k in range(doc_len[i]):
                    arg1_start = min([clause_start[min([emotion_pos[i][j] - 1, doc_len[i]])], len(discourse[i]) - 1])
                    arg1_end = min([clause_start[min([emotion_pos[i][j], doc_len[i]])], len(discourse[i]) - 1])
                    arg2_start = min([clause_start[k], len(discourse[i]) - 1])
                    arg2_end = min([clause_start[k + 1], len(discourse[i]) - 1])
                    arg1 = discourse[i][arg1_start: arg1_end]
                    arg2 = discourse[i][arg2_start: arg2_end]
                    len1 = len(arg1)
                    len2 = len(arg2)
                    sep_token = torch.tensor([102])
                    mask_token = torch.tensor([103])
                    inputs = torch.cat([arg1, sep_token, mask_token, arg2])
                    inputs = F.pad(inputs, (0, 512 - inputs.size(-1)), 'constant', 0).unsqueeze(0)
                    mask = torch.tensor([1] * (len1 + 2 + len2)+ [0] * (510 - len1 - len2)).unsqueeze(0)
                    segement = torch.tensor([0] * (len1 + 1) + [1] * (511 - len1)).unsqueeze(0)
            
                    # Get connective embedding
                    with torch.no_grad():
                        conn_embedding = self.bert_eval(inputs.to(DEVICE), mask.to(DEVICE), segement.to(DEVICE))[0][0][len1 + 1]
                    
                    # Stack three embeddings for one pair presentation
                    pair_h = torch.stack([doc_sents_h[i][emotion_pos[i][j] - 1], conn_embedding, doc_sents_h[i][k]], dim=-1)  # shape: feat_dim, 3
                    pair_h = pair_h.reshape([3 * self.feat_dim])  # shape: feat_dim * 3
                    
                    # Concatenate pairs for whole doc answer
                    if pairs_h.shape[-1] == 0:
                        pairs_h = pair_h
                    else:
                        pairs_h = torch.vstack([pairs_h, pair_h])  # shape: clause_len * emotion_num, feat_dim * 3
            
            # Pad pairs_h to 8 * max_doc_len
            while pairs_h.size(0) < 8 * max_doc_len:
                pairs_h = torch.vstack([pairs_h, torch.zeros(([3 * self.feat_dim])).to(DEVICE)])  # shape: 8 * max_doc_len, feat_dim * 3
            
        # Concatenate pairs for whole batch answer
            if pairs_hs.shape[-1] == 0:
                pairs_hs = pairs_h
            else:
                pairs_hs = torch.stack([pairs_hs, pairs_h], dim=0)  # shape: batch_size, 8 * max_doc_len, 3 * feat_dim

        pred_emo_cau = self.out_emo_cau(pairs_hs).squeeze(-1)  # shape: batch_size, 8 * max_doc_len
        pred_emo_cau = torch.sigmoid(pred_emo_cau)
        return pred_emo_cau # shape: batch_size, 8 * max_doc_len

    def loss_pre_emo(self, pred, true, mask):
        true = torch.FloatTensor(true.float()).to(DEVICE)  # shape: batch_size, seq_len
        mask = torch.BoolTensor(mask.bool()).to(DEVICE)
        pred = pred.masked_select(mask)
        true = true.masked_select(mask)
        # weight = torch.where(true > 0.5, 2, 1)
        criterion = nn.BCELoss()
        return criterion(pred, true)

    def loss_pre_emo_cau(self, pred, true, mask):
        true = torch.FloatTensor(true.float()).to(DEVICE)  # shape: batch_size, seq_len
        mask = torch.BoolTensor(mask.bool()).to(DEVICE)
        pred = pred.masked_select(mask)
        true = true.masked_select(mask)
        # weight = torch.where(true > 0.5, 2, 1)
        criterion = nn.BCELoss()
        return criterion(pred, true)