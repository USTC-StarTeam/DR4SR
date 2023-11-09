import torch
import torch.nn as nn

from utils.utils import xavier_normal_initialization

class BaseModel(torch.nn.Module):
    def __init__(self, args, train_dataset) -> None:
        super().__init__()
        self.args = args
        self.embed_dim = args['embed_dim']
        self.max_seq_len = args['max_seq_len']
        self.device = args['device']
        self.num_users = train_dataset.num_users
        self.num_items = train_dataset.num_items

    def init_parameters(self):
        self.apply(xavier_normal_initialization)

    def forward(self, batch):
        raise NotImplementedError