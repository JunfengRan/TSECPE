import torch
DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')

TORCH_SEED = 129


class Config(object):
    def __init__(self):
        # self.split = 'split10'
        self.bert_cache_path = 'bert-base-chinese'
        self.train_dataset_path = "../data/discourse_withconn_super_train.csv"
        self.test_dataset_path = "../data/discourse_withconn_super_test.csv"
        # self.train_dataset_path = "../data/train.csv"
        # self.test_dataset_path = "../data/test.csv"
        
        # hyper parameter
        self.epochs = 20
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
