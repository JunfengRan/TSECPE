from config import *
from transformers import BertTokenizer, BertModel, BertForMaskedLM, AutoModel, AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from config import DEVICE
from gnn_layer import GraphAttentionLayer


configs = Config()

# Init tokenizer
tokenizer = AutoTokenizer.from_pretrained(configs.roberta_cache_path)

# Init param
cause_uniconn = []
noncause_uniconn = []
candidate_conn = []
cause_uniconn_token = []
noncause_uniconn_token = []
candidate_conn_token = []
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
for i in range(len(cause_uniconn)):
    cause_uniconn_token.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cause_uniconn[i])))
for i in range(len(noncause_uniconn)):
    noncause_uniconn_token.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(noncause_uniconn[i])))
for i in range(len(candidate_conn)):
    candidate_conn_token.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(candidate_conn[i])))


class Network(nn.Module):
    def __init__(self, configs):
        super(Network, self).__init__()
        self.configs = configs
        self.feat_dim = self.configs.feat_dim
        self.batch_size = self.configs.batch_size
        
        self.roberta_train = AutoModel.from_pretrained(self.configs.roberta_cache_path)
        self.roberta_mask = AutoModelForMaskedLM.from_pretrained(self.configs.roberta_cache_path)
        self.roberta_eval = AutoModel.from_pretrained(self.configs.roberta_cache_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.configs.roberta_cache_path)
        
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
        
        # pred_cau
        self.out_cau = nn.Linear(self.feat_dim, 1)
        
        # pred_pair
        self.out_pair1 = nn.Linear(self.feat_dim * 3, 1)
        self.out_pair2 = nn.Linear(self.feat_dim * 3, 1)
        self.roberta_mask.eval()
        self.roberta_eval.eval()
    
    def forward(self, query, query_mask, query_seg, query_len, clause_len, ec_emotion_pos, ec_cause_pos, ce_cause_pos, ce_emotion_pos, doc_len, adj, q_type):
        # shape: batch_size, max_doc_len, 512
        doc_sents_h = self.bert_encoder(query, query_mask, query_seg, query_len, clause_len, doc_len)
        doc_sents_h = self.graph_gnn(doc_sents_h, doc_len, adj)
        pred_emo = self.pred_emo(doc_sents_h)
        pred_emo_cau = self.pred_emo_cau(doc_sents_h, query, ec_emotion_pos, doc_len, clause_len)
        pred_cau = self.pred_cau(doc_sents_h)
        pred_cau_emo = self.pred_cau_emo(doc_sents_h, query, ce_cause_pos, doc_len, clause_len)
        if q_type == 'emo':
            return pred_emo
        if q_type == 'emo_cau':
            return pred_emo_cau
        if q_type == 'cau':
            return pred_cau
        if q_type == 'cau_emo':
            return pred_cau_emo
        
        return None

    def bert_encoder(self, discourse, discourse_mask, segment_mask, query_len, clause_len, doc_len):
        
        hidden_states = self.roberta_train(input_ids=discourse.to(DEVICE),
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

    def pred_cau(self, doc_sents_h):
        pred_cau = self.out_cau(doc_sents_h).squeeze(-1)  # shape: batch_size, max_doc_len, 1
        pred_cau = torch.sigmoid(pred_cau)
        return pred_cau # shape: batch_size, max_doc_len

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
                    arg1_start = clause_start[emotion_pos[i][j] - 1]
                    arg1_end = clause_start[emotion_pos[i][j]]
                    arg2_start = clause_start[k]
                    arg2_end = clause_start[k + 1]
                    arg1 = discourse[i][arg1_start: arg1_end]
                    arg2 = discourse[i][arg2_start: arg2_end]
                    len1 = len(arg1)
                    len2 = len(arg2)
                    
                    # rule for adding connectives
                    # 1. Search the candidate clauses from the beginning of the clause to the end, requiring a continuous sequence of connectives, otherwise stop.
                    # For example, "but because" is a continuous sequence of connectives.

                    # 2. If there is a connective sequence, we delete the connective sequence and predict the connective with Bert or Roberta.
                    # If there is no connective sequence, select the connective directly with Bert or Roberta.

                    # 3. If the clause itself forms a pair with itself, we delete the first sequence of connectives and extract sequences of connectives from any other position in the clause (excluding the sequence at the beginning).
                    # If there is a single causal connective in the sequence of connectives, directly extract the single-causal connective and choose it as our desired connectives.
                    # If there are multiple choice, we choose the first one; If there is no single causal connective, select the connective directly with Bert or Roberta.
                    
                    if arg1_start != arg2_start:
                        # step 1
                        conn = None
                        conn_seq = []
                        for pos in range(len2):
                            if arg2[pos] in candidate_conn_token:
                                conn_seq.append(arg2[pos])
                            else:
                                break
                        # step 2
                        if conn_seq != []:
                            arg2 = arg2[len(conn_seq):]
                            len2 = len2 - len(conn_seq)
                        sep_token = torch.tensor([self.tokenizer.convert_tokens_to_ids('[SEP]')])
                        mask_token = torch.tensor([self.tokenizer.convert_tokens_to_ids('[MASK]')])
                        inputs = torch.cat([arg1, sep_token, mask_token, arg2])
                        inputs = F.pad(inputs, (0, 512 - inputs.size(-1)), 'constant', 0).unsqueeze(0)
                        mask = torch.tensor([1] * (len1 + 2 + len2)+ [0] * (510 - len1 - len2)).unsqueeze(0)
                        segement = torch.tensor([0] * (len1 + 1) + [1] * (511 - len1)).unsqueeze(0)
                    
                        # Get mask distribution
                        with torch.no_grad():
                            mask_distribution = self.roberta_mask(inputs.to(DEVICE), mask.to(DEVICE), segement.to(DEVICE))[0][0][len1 + 1]
                        
                        # Get conn
                        candidate_conn_score = []
                        for idx in range(len(candidate_conn_token)):
                            candidate_conn_score.extend(mask_distribution[candidate_conn_token[idx]].cpu())
                        max_index = torch.argmax(torch.tensor(candidate_conn_score))
                        conn = candidate_conn_token[max_index]

                    # step 3
                    else:
                        conn = None
                        conn_seq = []
                        conn_seqs = []
                        for pos in range(len2):
                            if arg2[pos] in candidate_conn_token:
                                conn_seq.append(arg2[pos])
                            else:
                                break
                        arg2 = arg2[len(conn_seq):]
                        len2 = len2 - len(conn_seq)
                        pos = 0
                        while pos < len2:
                            if arg2[pos] in candidate_conn_token:
                                conn_seq = []
                                while pos < len2:
                                    if arg2[pos] in candidate_conn_token:
                                        conn_seq.append(arg2[pos])
                                        pos += 1
                                    else:
                                        break
                                conn_seqs.append(conn_seq)
                            pos += 1
                        if conn_seqs != []:
                            for idx in range(len(conn_seqs)):
                                for item in conn_seqs[idx]:
                                    if item in cause_uniconn_token:
                                        conn = item
                                        break
                                if conn != None:
                                    break
                        if conn == None:
                            sep_token = torch.tensor([self.tokenizer.convert_tokens_to_ids('[SEP]')])
                            mask_token = torch.tensor([self.tokenizer.convert_tokens_to_ids('[MASK]')])
                            inputs = torch.cat([arg1, sep_token, mask_token, arg2])
                            inputs = F.pad(inputs, (0, 512 - inputs.size(-1)), 'constant', 0).unsqueeze(0)
                            mask = torch.tensor([1] * (len1 + 2 + len2)+ [0] * (510 - len1 - len2)).unsqueeze(0)
                            segement = torch.tensor([0] * (len1 + 1) + [1] * (511 - len1)).unsqueeze(0)
                        
                            # Get mask distribution
                            with torch.no_grad():
                                mask_distribution = self.roberta_mask(inputs.to(DEVICE), mask.to(DEVICE), segement.to(DEVICE))[0][0][len1 + 1]
                            
                            # Get conn
                            candidate_conn_score = []
                            for idx in range(len(candidate_conn_token)):
                                candidate_conn_score.extend(mask_distribution[candidate_conn_token[idx]].cpu())
                            max_index = torch.argmax(torch.tensor(candidate_conn_score))
                            conn = candidate_conn_token[max_index]
                    
                    # Get conn embedding
                    conn = torch.tensor(conn)
                    sep_token = torch.tensor([self.tokenizer.convert_tokens_to_ids('[SEP]')])
                    inputs = torch.cat([arg1, sep_token, conn, arg2])
                    inputs = F.pad(inputs, (0, 512 - inputs.size(-1)), 'constant', 0).unsqueeze(0)
                    mask = torch.tensor([1] * (len1 + 2 + len2)+ [0] * (510 - len1 - len2)).unsqueeze(0)
                    segement = torch.tensor([0] * (len1 + 1) + [1] * (511 - len1)).unsqueeze(0)

                    with torch.no_grad():
                        conn_embedding = self.roberta_eval(inputs.to(DEVICE), mask.to(DEVICE), segement.to(DEVICE))[0][0][len1 + 1]
                    
                    # Stack three embeddings for one pair presentation
                    pair_h = torch.cat([doc_sents_h[i][emotion_pos[i][j] - 1], conn_embedding, doc_sents_h[i][k]], dim=-1)  # shape: 3 * feat_dim
                    
                    # Concatenate pairs for whole doc answer
                    if pairs_h.size(-1) == 0:
                        pairs_h = pair_h
                    else:
                        pairs_h = torch.vstack([pairs_h, pair_h])  # shape: doc_len * emotion_num, feat_dim * 3
            
            # Pad pairs_h to 8 * max_doc_len
            while pairs_h.size(0) < 8 * max_doc_len:
                pairs_h = torch.vstack([pairs_h, torch.zeros(([3 * self.feat_dim])).to(DEVICE)])  # shape: 8 * max_doc_len, feat_dim * 3
            
            # Concatenate pairs for whole batch answer
            if pairs_hs.size(-1) == 0:
                pairs_hs = pairs_h.unsqueeze(0)
            else:
                pairs_hs = torch.cat([pairs_hs, pairs_h.unsqueeze(0)], dim=0)  # shape: batch_size, 8 * max_doc_len, 3 * feat_dim

        pred_cau = self.out_cau(pairs_hs[:, :, 2 * self.feat_dim:]).squeeze(-1)  # shape: batch_size, 8 * max_doc_len
        pred_emo_cau = self.out_pair1(pairs_hs).squeeze(-1)  # shape: batch_size, 8 * max_doc_len
        pred_emo_cau = torch.sigmoid(pred_cau + pred_emo_cau)
        return pred_emo_cau # shape: batch_size, 8 * max_doc_len

    def pred_cau_emo(self, doc_sents_h, discourse, cause_pos, doc_len, clause_len):
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
            for j in range(len(cause_pos[i])):
                for k in range(doc_len[i]):
                    arg1_start = clause_start[cause_pos[i][j] - 1]
                    arg1_end = clause_start[cause_pos[i][j]]
                    arg2_start = clause_start[k]
                    arg2_end = clause_start[k + 1]
                    arg1 = discourse[i][arg1_start: arg1_end]
                    arg2 = discourse[i][arg2_start: arg2_end]
                    len1 = len(arg1)
                    len2 = len(arg2)
                    
                    # rule for adding connectives
                    # 1. Search the candidate clauses from the beginning of the clause to the end, requiring a continuous sequence of connectives, otherwise stop.
                    # For example, "but because" is a continuous sequence of connectives.

                    # 2. If there is a connective sequence, we delete the connective sequence and predict the connective with Bert or Roberta.
                    # If there is no connective sequence, select the connective directly with Bert or Roberta.

                    # 3. If the clause itself forms a pair with itself, we delete the first sequence of connectives and extract sequences of connectives from any other position in the clause (excluding the sequence at the beginning).
                    # If there is a single causal connective in the sequence of connectives, directly extract the single-causal connective and choose it as our desired connectives.
                    # If there are multiple choice, we choose the first one; If there is no single causal connective, select the connective directly with Bert or Roberta.
                    
                    if arg1_start != arg2_start:
                        # step 1
                        conn = None
                        conn_seq = []
                        for pos in range(len2):
                            if arg2[pos] in candidate_conn_token:
                                conn_seq.append(arg2[pos])
                            else:
                                break
                        # step 2
                        if conn_seq != []:
                            arg2 = arg2[len(conn_seq):]
                            len2 = len2 - len(conn_seq)
                        sep_token = torch.tensor([self.tokenizer.convert_tokens_to_ids('[SEP]')])
                        mask_token = torch.tensor([self.tokenizer.convert_tokens_to_ids('[MASK]')])
                        inputs = torch.cat([arg1, sep_token, mask_token, arg2])
                        inputs = F.pad(inputs, (0, 512 - inputs.size(-1)), 'constant', 0).unsqueeze(0)
                        mask = torch.tensor([1] * (len1 + 2 + len2)+ [0] * (510 - len1 - len2)).unsqueeze(0)
                        segement = torch.tensor([0] * (len1 + 1) + [1] * (511 - len1)).unsqueeze(0)
                    
                        # Get mask distribution
                        with torch.no_grad():
                            mask_distribution = self.roberta_mask(inputs.to(DEVICE), mask.to(DEVICE), segement.to(DEVICE))[0][0][len1 + 1]
                        
                        # Get conn
                        candidate_conn_score = []
                        for idx in range(len(candidate_conn_token)):
                            candidate_conn_score.extend(mask_distribution[candidate_conn_token[idx]].cpu())
                        max_index = torch.argmax(torch.tensor(candidate_conn_score))
                        conn = candidate_conn_token[max_index]

                    # step 3
                    else:
                        conn = None
                        conn_seq = []
                        conn_seqs = []
                        for pos in range(len2):
                            if arg2[pos] in candidate_conn_token:
                                conn_seq.append(arg2[pos])
                            else:
                                break
                        arg2 = arg2[len(conn_seq):]
                        len2 = len2 - len(conn_seq)
                        pos = 0
                        while pos < len2:
                            if arg2[pos] in candidate_conn_token:
                                conn_seq = []
                                while pos < len2:
                                    if arg2[pos] in candidate_conn_token:
                                        conn_seq.append(arg2[pos])
                                        pos += 1
                                    else:
                                        break
                                conn_seqs.append(conn_seq)
                            pos += 1
                        if conn_seqs != []:
                            for idx in range(len(conn_seqs)):
                                for item in conn_seqs[idx]:
                                    if item in cause_uniconn_token:
                                        conn = item
                                        break
                                if conn != None:
                                    break
                        if conn == None:
                            sep_token = torch.tensor([self.tokenizer.convert_tokens_to_ids('[SEP]')])
                            mask_token = torch.tensor([self.tokenizer.convert_tokens_to_ids('[MASK]')])
                            inputs = torch.cat([arg1, sep_token, mask_token, arg2])
                            inputs = F.pad(inputs, (0, 512 - inputs.size(-1)), 'constant', 0).unsqueeze(0)
                            mask = torch.tensor([1] * (len1 + 2 + len2)+ [0] * (510 - len1 - len2)).unsqueeze(0)
                            segement = torch.tensor([0] * (len1 + 1) + [1] * (511 - len1)).unsqueeze(0)
                        
                            # Get mask distribution
                            with torch.no_grad():
                                mask_distribution = self.roberta_mask(inputs.to(DEVICE), mask.to(DEVICE), segement.to(DEVICE))[0][0][len1 + 1]
                            
                            # Get conn
                            candidate_conn_score = []
                            for idx in range(len(candidate_conn_token)):
                                candidate_conn_score.extend(mask_distribution[candidate_conn_token[idx]].cpu())
                            max_index = torch.argmax(torch.tensor(candidate_conn_score))
                            conn = candidate_conn_token[max_index]
                    
                    # Get conn embedding
                    conn = torch.tensor(conn)
                    sep_token = torch.tensor([self.tokenizer.convert_tokens_to_ids('[SEP]')])
                    inputs = torch.cat([arg1, sep_token, conn, arg2])
                    inputs = F.pad(inputs, (0, 512 - inputs.size(-1)), 'constant', 0).unsqueeze(0)
                    mask = torch.tensor([1] * (len1 + 2 + len2)+ [0] * (510 - len1 - len2)).unsqueeze(0)
                    segement = torch.tensor([0] * (len1 + 1) + [1] * (511 - len1)).unsqueeze(0)

                    with torch.no_grad():
                        conn_embedding = self.roberta_eval(inputs.to(DEVICE), mask.to(DEVICE), segement.to(DEVICE))[0][0][len1 + 1]
                    
                    # Stack three embeddings for one pair presentation
                    pair_h = torch.cat([doc_sents_h[i][k], conn_embedding, doc_sents_h[i][cause_pos[i][j] - 1]], dim=-1)  # shape: 3 * feat_dim
                    
                    # Concatenate pairs for whole doc answer
                    if pairs_h.size(-1) == 0:
                        pairs_h = pair_h
                    else:
                        pairs_h = torch.vstack([pairs_h, pair_h])  # shape: doc_len * emotion_num, feat_dim * 3
            
            # Pad pairs_h to 8 * max_doc_len
            while pairs_h.size(0) < 8 * max_doc_len:
                pairs_h = torch.vstack([pairs_h, torch.zeros(([3 * self.feat_dim])).to(DEVICE)])  # shape: 8 * max_doc_len, feat_dim * 3
            
            # Concatenate pairs for whole batch answer
            if pairs_hs.size(-1) == 0:
                pairs_hs = pairs_h.unsqueeze(0)
            else:
                pairs_hs = torch.cat([pairs_hs, pairs_h.unsqueeze(0)], dim=0)  # shape: batch_size, 8 * max_doc_len, 3 * feat_dim

        pred_emo = self.out_emo(pairs_hs[:, :, :self.feat_dim]).squeeze(-1)  # shape: batch_size, 8 * max_doc_len
        pred_cau_emo = self.out_pair2(pairs_hs).squeeze(-1)  # shape: batch_size, 8 * max_doc_len
        pred_cau_emo = torch.sigmoid(pred_emo + pred_cau_emo)
        return pred_cau_emo # shape: batch_size, 8 * max_doc_len

    def loss_pre(self, pred, true, mask):
        true = torch.FloatTensor(true.float()).to(DEVICE)  # shape: batch_size, seq_len
        mask = torch.BoolTensor(mask.bool()).to(DEVICE)
        pred = pred.masked_select(mask)
        true = true.masked_select(mask)
        # weight = torch.where(true > 0.5, 2, 1)
        criterion = nn.BCELoss()
        return criterion(pred, true)