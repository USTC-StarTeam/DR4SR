import torch
import torch.nn as nn
from model.basemodel import BaseModel
from module.layers import SeqPoolingLayer
from data import dataset

class BPR(BaseModel):
    def __init__(self, config, train_dataset) -> None:
        super().__init__(config, train_dataset)
        self.user_embedding = nn.Embedding(self.num_users, self.embed_dim)

    def _get_dataset_class():
        return dataset.SeparateDataset

    def forward(self, batch):
        if self.training:
            return self.user_embedding(batch['user_id']).unsqueeze(-2)
        else:
            return self.user_embedding(batch['user_id'])
