import torch
import torch.nn as nn
import torch.nn.functional as F
from model.basemodel import BaseModel
from model.layers import SeqPoolingLayer
from data import dataset

class SASRec2(BaseModel):
    def __init__(self, config, dataset_list : list[dataset.BaseDataset]) -> None:
        super().__init__(config, dataset_list)
        self.position_emb = torch.nn.Embedding(self.max_seq_len * dataset_list[0].num_domains, self.embed_dim)
        transformer_encoder = torch.nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=config['head_num'],
            dim_feedforward=config['hidden_size'],
            dropout=config['dropout_rate'],
            activation=config['activation'],
            layer_norm_eps=config['layer_norm_eps'],
            batch_first=True,
            norm_first=False
        )
        self.transformer_layer = torch.nn.TransformerEncoder(
            encoder_layer=transformer_encoder,
            num_layers=config['layer_num'],
        )
        self.dropout = torch.nn.Dropout(p=config['dropout_rate'])
        self.training_pooling_layer = SeqPoolingLayer(pooling_type='origin')
        self.eval_pooling_layer = SeqPoolingLayer(pooling_type='last')

    def _get_dataset_class():
        return dataset.SeparateDataset
        return dataset.MixDataset

    @staticmethod
    def alignment(x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    @staticmethod
    def uniformity(x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def forward(self, batch):
        seq_embs, seq_len = batch['seq_embs'], batch['seq_len']
        positions = torch.arange(seq_embs.size(1), dtype=torch.long, device=self.device)
        positions = positions.unsqueeze(0)
        position_embs = self.position_emb(positions)

        L = seq_embs.size(1)
        attention_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=self.device), 1)
        transformer_out = self.transformer_layer(
            src=self.dropout(seq_embs+position_embs),
            mask=attention_mask
        )  # BxLxD
        if self.training:
            return self.training_pooling_layer(transformer_out, seq_len)
        else:
            return self.eval_pooling_layer(transformer_out, seq_len)