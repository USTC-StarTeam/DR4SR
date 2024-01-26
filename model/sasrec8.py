import torch
import torch.nn as nn
import torch.nn.functional as F
from model.basemodel import BaseModel
from module.layers import SeqPoolingLayer, VanillaVectorQuantizer, MLPModule, TransformerEncoder
from data import dataset
from copy import deepcopy
from tqdm import tqdm

class SASRec8(BaseModel):
    def __init__(self, config, dataset_list : list[dataset.BaseDataset]) -> None:
        super().__init__(config, dataset_list)
        # Query encoder
        self.position_embedding = torch.nn.Embedding(self.max_seq_len, self.embed_dim)
        self.trm_encoder = TransformerEncoder(
            n_layers=config['model']['layer_num'],
            n_heads=config['model']['head_num'],
            hidden_size=self.embed_dim,
            inner_size=config['model']['hidden_size'],
            hidden_dropout_prob=config['model']['dropout_rate'],
            attn_dropout_prob=config['model']['dropout_rate'],
            hidden_act=config['model']['activation'],
            layer_norm_eps=config['model']['layer_norm_eps'],
        )
        self.LayerNorm = nn.LayerNorm(self.embed_dim, eps=config['model']['layer_norm_eps'])
        self.dropout = nn.Dropout(config['model']['dropout_rate'])
        self.training_pooling_layer = SeqPoolingLayer(pooling_type='last')
        self.eval_pooling_layer = SeqPoolingLayer(pooling_type='last')

    def current_epoch_trainloaders(self, nepoch):
        return super().current_epoch_trainloaders(nepoch)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask

    def forward(self, batch, need_pooling=True):
        if batch.get('pattern', None) is None:
            item_seq = batch['in_' + self.fiid]
            pattern = self.item_embedding(item_seq)
            att_mask = self.get_attention_mask(item_seq)
        else:
            pattern = batch['pattern']
            att_mask = batch['att_mask']

        last_seq_len = batch['seqlen']
        position_ids = torch.arange(
            pattern.size(1), dtype=torch.long, device=self.device
        )
        position_ids = position_ids.reshape(1, -1)
        position_embedding = self.position_embedding(position_ids)

        input_emb = pattern + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        trm_output = self.trm_encoder(
            input_emb, att_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.training_pooling_layer(output, last_seq_len)
        return output

    @staticmethod
    def alignment(x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    @staticmethod
    def uniformity(x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def training_step(self, batch, reduce=True, return_query=False, align=True):
        if align:
            query = self.forward(batch, need_pooling=True)
            alignment = self.alignment(query, self.item_embedding.weight[batch[self.fiid]])
            uniformity = 1 * (
                self.uniformity(query) +
                self.uniformity(self.item_embedding.weight[batch[self.fiid]])
            )
            loss_value = alignment + uniformity
            return loss_value
        else:
            return super().training_step(batch, reduce, return_query)
