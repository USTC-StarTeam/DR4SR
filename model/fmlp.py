import torch
import torch.nn as nn
import torch.nn.functional as F
from model.basemodel import BaseModel
from module.layers import SeqPoolingLayer, FMLPEncoder
from data import dataset

class FMLP(BaseModel):
    def __init__(self, config, dataset_list : list[dataset.BaseDataset]) -> None:
        super().__init__(config, dataset_list)
        self.position_embeddings = nn.Embedding(50, 64)
        self.LayerNorm = nn.LayerNorm(64, eps=1e-12)
        self.dropout = nn.Dropout(0.5)
        self.item_encoder = FMLPEncoder()
        self.training_pooling_layer = SeqPoolingLayer(pooling_type='last')
        self.eval_pooling_layer = SeqPoolingLayer(pooling_type='last')

    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embedding(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    def forward(self, batch, need_pooling=True):
        input_ids = batch['in_' + self.fiid]
        sequence_emb = self.add_position_embedding(input_ids)
        item_encoded_layers = self.item_encoder(
            sequence_emb,
            output_all_encoded_layers=True,
        )
        transformer_out = item_encoded_layers[-1]
        return transformer_out[:, -1]

    def current_epoch_trainloaders(self, nepoch):
        return super().current_epoch_trainloaders(nepoch)

    def training_step(self, batch, reduce=True, return_query=False, align=False):
        return super().training_step(batch, reduce, return_query)