import torch
import torch.nn as nn
from model.basemodel import BaseModel
from model.layers import SeqPoolingLayer
from data import dataset

class SASRec(BaseModel):
    def __init__(self, config, train_dataset) -> None:
        super().__init__(config, train_dataset)
        self.position_emb = torch.nn.Embedding(self.max_seq_len, self.embed_dim)
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
        self.training_pooling_layer = SeqPoolingLayer(pooling_type='last')
        self.eval_pooling_layer = SeqPoolingLayer(pooling_type='last')

    
    def _get_dataset_class():
        return dataset.NormalDataset

    def forward(self, batch):
        user_seq, seq_len = batch['user_seq'], batch['seq_len']
        positions = torch.arange(user_seq.size(1), dtype=torch.long, device=self.device)
        positions = positions.unsqueeze(0).expand_as(user_seq)
        position_embs = self.position_emb(positions)
        seq_embs = self.item_embedding(user_seq)

        mask4padding = user_seq == self.num_items  # BxL
        L = user_seq.size(-1)
        attention_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=user_seq.device), 1)
        transformer_out = self.transformer_layer(
            src=self.dropout(seq_embs+position_embs),
            mask=attention_mask,
            src_key_padding_mask=mask4padding
        )  # BxLxD
        if self.training:
            return self.training_pooling_layer(transformer_out, seq_len)
        else:
            return self.eval_pooling_layer(transformer_out, seq_len)
