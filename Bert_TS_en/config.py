import torch
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')

TORCH_SEED = 129


class Config(object):
    def __init__(self):
        # model (need change)
        self.bert_cache_path = 'bert-base-uncased'
        self.bert_finetuned_path = '../model/bert_finetuned.pth'
        self.roberta_cache_path = 'roberta-base'
        
        # dataset
        self.dataset_len = 100
        self.fold = 10
        self.dataset_path = "../data/discourse_en.csv"
        
        # hyper parameter
        self.epochs = 1
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

