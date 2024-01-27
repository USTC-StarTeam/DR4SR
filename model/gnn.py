import torch
import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
from model.basemodel import BaseModel
from module.layers import SeqPoolingLayer
from module import data_augmentation
from data import dataset
from copy import deepcopy

class GNNQueryEncoder(torch.nn.Module):
    def __init__(
            self, fiid, embed_dim, max_seq_len, n_head, hidden_size, dropout, activation, layer_norm_eps, n_layer, item_encoder, graph,
            gnn_layer=2, bidirectional=False, training_pooling_type='origin', eval_pooling_type='last') -> None:
        super().__init__()
        self.fiid = fiid
        self.item_encoder = item_encoder
        self.bidirectional = bidirectional
        self.training_pooling_type = training_pooling_type
        self.eval_pooling_type = eval_pooling_type
        self.position_emb = torch.nn.Embedding(max_seq_len, embed_dim)
        self.gnn_layer = gnn_layer
        transformer_encoder = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=hidden_size,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=False
        )
        self.transformer_layer = torch.nn.TransformerEncoder(
            encoder_layer=transformer_encoder,
            num_layers=n_layer,
        )
        self.dropout = torch.nn.Dropout(p=dropout)
        self.training_pooling_layer = SeqPoolingLayer(pooling_type=self.training_pooling_type)
        self.eval_pooling_layer = SeqPoolingLayer(pooling_type=self.eval_pooling_type)
        self.norm_adj = graph

    def get_gnn_embeddings(self):
        emb = self.item_encoder.weight
        emb_list = [emb]
        for _ in range(self.gnn_layer):
            emb = torch.sparse.mm(self.norm_adj, emb)
            emb_list.append(emb)
        emb = torch.stack(emb_list, dim=1).mean(1)
        return emb

    def forward(self, batch, need_pooling=True):
        user_hist = batch['in_'+self.fiid]
        positions = torch.arange(user_hist.size(1), dtype=torch.long, device=user_hist.device)
        positions = positions.unsqueeze(0).expand_as(user_hist)
        position_embs = self.position_emb(positions)
        seq_embs = self.get_gnn_embeddings()[user_hist]

        mask4padding = user_hist == 0  # BxL
        L = user_hist.size(-1)
        if not self.bidirectional:
            attention_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=user_hist.device), 1)
        else:
            attention_mask = torch.zeros((L, L), dtype=torch.bool, device=user_hist.device)
        transformer_out = self.transformer_layer(
            src=self.dropout(seq_embs+position_embs),
            mask=attention_mask,
            src_key_padding_mask=mask4padding)  # BxLxD
        if not need_pooling:
            return transformer_out
        else:
            if self.training:
                return self.training_pooling_layer(transformer_out, batch['seqlen'])
            else:
                return self.eval_pooling_layer(transformer_out, batch['seqlen'])

class GNN(BaseModel):
    def __init__(self, config, dataset_list : list[dataset.BaseDataset]) -> None:
        super().__init__(config, dataset_list)
        if config['model']['graph'] == 'old':
            graph = self._build_graph_old()
        elif config['model']['graph'] == 'new':
            graph = self._build_graph()
        self.query_encoder = GNNQueryEncoder(
            self.fiid,
            self.embed_dim,
            self.max_seq_len,
            config['model']['head_num'],
            config['model']['hidden_size'],
            config['model']['dropout_rate'],
            config['model']['activation'],
            config['model']['layer_norm_eps'],
            config['model']['layer_num'],
            self.item_embedding,
            graph,
            config['model']['gnn_layer']
        )

    def _build_graph_old(self):
        # get user seq from validation dataset
        domain = self.dataset_list[1].eval_domain
        history_matrix = self.dataset_list[1].data[domain][1].cpu()
        history_len = self.dataset_list[1].data[domain][3].cpu()
        n_items = self.num_items

        # build graph
        history_matrix = history_matrix.tolist()
        row, col, data = [], [], []
        for idx, item_list_len in enumerate(history_len):
            # -1 as the validation dataset includes the target item in training dataset
            item_list_len -= 1
            # get item_list
            item_list = history_matrix[idx][:item_list_len]
            # build graph with sliding window of size 2
            for item_idx in range(item_list_len - 1):
                target_num = min(2, item_list_len - item_idx - 1)
                row += [item_list[item_idx]] * target_num
                col += item_list[item_idx + 1: item_idx + 1 + target_num]
                data.append(1 / np.arange(1, 1 + target_num))
        data = np.concatenate(data)
        sparse_matrix = sp.csc_matrix((data, (row, col)), shape=(n_items, n_items))
        sparse_matrix = sparse_matrix + sparse_matrix.T + sp.eye(n_items)
        degree = np.array((sparse_matrix > 0).sum(1)).flatten()
        degree = np.nan_to_num(1 / degree, posinf=0)
        degree = sp.diags(degree)
        norm_adj = (degree @ sparse_matrix + sparse_matrix @ degree).tocoo()
        norm_adj = torch.sparse_coo_tensor(
            np.row_stack([norm_adj.row, norm_adj.col]),
            norm_adj.data,
            (n_items, n_items),
            dtype=torch.float32,
            device=self.device
        )
        return norm_adj

    def _build_graph(self):
        # get user seq from training dataset
        history_matrix = self.dataset_list[0].data[1].cpu().tolist()
        history_len = self.dataset_list[0].data[3].cpu()
        n_items = self.num_items

        # build graph
        row, col, data = [], [], []
        for idx, item_list_len in enumerate(history_len):
            # -1 as the validation dataset includes the target item in training dataset
            item_list_len -= 1
            # get item_list
            item_list = history_matrix[idx][:item_list_len]
            # build graph with sliding window of size 2
            for item_idx in range(item_list_len - 1):
                target_num = min(2, item_list_len - item_idx - 1)
                row += [item_list[item_idx]] * target_num
                col += item_list[item_idx + 1: item_idx + 1 + target_num]
                data.append(1 / np.arange(1, 1 + target_num))
        data = np.concatenate(data)
        sparse_matrix = sp.csc_matrix((data, (row, col)), shape=(n_items, n_items))
        sparse_matrix = sparse_matrix + sparse_matrix.T + sp.eye(n_items)
        degree = np.array((sparse_matrix > 0).sum(1)).flatten()
        degree = np.nan_to_num(1 / degree, posinf=0)
        degree = sp.diags(degree)
        norm_adj = (degree @ sparse_matrix + sparse_matrix @ degree).tocoo()
        norm_adj = torch.sparse_coo_tensor(
            np.row_stack([norm_adj.row, norm_adj.col]),
            norm_adj.data,
            (n_items, n_items),
            dtype=torch.float32,
            device=self.device
        )
        return norm_adj

    def forward(self, batch):
        return self.query_encoder(batch)

    def training_step(self, batch, reduce=True, return_query=False, align=False):
        return super().training_step(batch, reduce, return_query)