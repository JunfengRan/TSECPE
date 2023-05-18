import torch
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')

TORCH_SEED = 129


class Config(object):
    def __init__(self):
        # model (need change)
        self.bert_cache_path = 'bert-base-chinese'
        self.roberta_cache_path = 'hfl/chinese-roberta-wwm-ext'
        
        # dataset
        self.dataset_len = 2105
        self.fold = 10
        self.dataset_path = "../data/discourse_zh.csv"
        
        # hyper parameter
        self.epochs = 30
        self.batch_size = 2
        self.lr = 1e-5
        self.tuning_bert_rate = 1e-5
        self.gradient_accumulation_steps = 2
        self.dp = 0.1
        self.warmup_proportion = 0.1
        self.feat_dim = 768

        # gnn
        self.gnn_dims = '192'
        self.att_heads = '4'

