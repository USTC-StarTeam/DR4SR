import torch
import torch.nn as nn
import torch.nn.functional as F
from model.basemodel import BaseModel
from module.layers import SeqPoolingLayer
from data import dataset

class SASRec2(BaseModel):
    def __init__(self, config, dataset_list : list[dataset.BaseDataset]) -> None:
        super().__init__(config, dataset_list)
        self.position_emb = torch.nn.Embedding(self.max_seq_len * dataset_list[0].num_domains, self.embed_dim)
        model_config = config['model']
        transformer_encoder = torch.nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=model_config['head_num'],
            dim_feedforward=model_config['hidden_size'],
            dropout=model_config['dropout_rate'],
            activation=model_config['activation'],
            layer_norm_eps=model_config['layer_norm_eps'],
            batch_first=True,
            norm_first=False
        )
        self.transformer_layer = torch.nn.TransformerEncoder(
            encoder_layer=transformer_encoder,
            num_layers=model_config['layer_num'],
        )
        self.dropout = torch.nn.Dropout(p=model_config['dropout_rate'])
        self.training_pooling_layer = SeqPoolingLayer(pooling_type='last')
        self.eval_pooling_layer = SeqPoolingLayer(pooling_type='last')

    def forward_eval(self, batch):
        user_hist = batch['in_'+self.fiid]
        positions = torch.arange(user_hist.size(1), dtype=torch.long, device=user_hist.device)
        positions = positions.unsqueeze(0).expand_as(user_hist)
        position_embs = self.position_emb(positions)
        seq_embs = self.item_embedding(user_hist)

        mask4padding = user_hist == 0  # BxL
        L = user_hist.size(-1)
        attention_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=user_hist.device), 1)
        transformer_out = self.transformer_layer(
            src=self.dropout(seq_embs+position_embs),
            mask=attention_mask,
            src_key_padding_mask=mask4padding)  # BxLxD

        if self.training:
            return self.training_pooling_layer(transformer_out, batch['seqlen'])
        else:
            return self.eval_pooling_layer(transformer_out, batch['seqlen'])

    def forward_ori(self, batch, need_pooling=True):
        user_hist = batch['in_'+self.fiid]
        positions = torch.arange(user_hist.size(1), dtype=torch.long, device=user_hist.device)
        positions = positions.unsqueeze(0).expand_as(user_hist)
        position_embs = self.position_emb(positions)
        seq_embs = self.item_embedding(user_hist)

        L = user_hist.size(-1)
        mask4padding = user_hist == 0  # BxL
        attention_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=user_hist.device), 1)
        transformer_input = seq_embs + position_embs
        transformer_out = self.transformer_layer(
            src=self.dropout(transformer_input),
            mask=attention_mask,
            src_key_padding_mask=mask4padding)  # BxLxD
        if not need_pooling:
            return transformer_out
        else:
            if self.training:
                return self.training_pooling_layer(transformer_out, batch['seqlen'])
            else:
                return self.eval_pooling_layer(transformer_out, batch['seqlen'])

    def forward_syn(self, batch, need_pooling=True):
        seq_embs, seq_len = batch['seq_embs'], batch['seqlen']
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

    def forward(self, batch, need_pooling=True):
        if batch.get('seq_embs', None) is not None:
            return self.forward_syn(batch, need_pooling=need_pooling)
        else:
            return self.forward_ori(batch, need_pooling=need_pooling)