import torch
import torch.nn as nn
from data.dataset import BaseDataset
from module import data_augmentation
from model.sasrec import SASRec, SASRecQueryEncoder
from data import dataset
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F

r"""
CL4SRec
#############
    Contrastive Learning for Sequential Recommendation(SIGIR'21)
    Reference:
        https://arxiv.org/abs/2010.14395
"""
class APGL4SR(SASRec):
    r"""
    Model hyper parameters:
        - ``embed_dim(int)``: The dimension of embedding layers. Default: ``64``.
        - ``hidden_size(int)``: The output size of Transformer layer. Default: ``128``.
        - ``layer_num(int)``: The number of layers for the Transformer. Default: ``2``.
        - ``dropout_rate(float)``:  The dropout probablity for dropout layers after item embedding
         | and in Transformer layer. Default: ``0.5``.
        - ``head_num(int)``: The number of heads for MultiHeadAttention in Transformer. Default: ``2``.
        - ``activation(str)``: The activation function in transformer. Default: ``"gelu"``.
        - ``layer_norm_eps``: The layer norm epsilon in transformer. Default: ``1e-12``.
    """

    def __init__(self, config, dataset_list: list[BaseDataset]) -> None:
        super().__init__(config, dataset_list)
        self.item_embedding = nn.Embedding(
            self.num_items + 1, self.embed_dim, padding_idx=0
        ) # re-define item_embedding to add an item for augmentation
        self.query_encoder = SASRecQueryEncoder(
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
        )
        if config['model']['graph'] == 'old':
            self.norm_adj = self._build_graph_old()
        elif config['model']['graph'] == 'new':
            self.norm_adj = self._build_graph()
        self.infonce = data_augmentation.InfoNCELoss(config['model']['graph_tmp'])

    def get_gnn_embeddings(self, noise):
        emb = self.item_embedding.weight[:-1] # remove augmentation padding
        emb_list = [emb]
        for _ in range(self.config['model']['gnn_layer']):
            emb = torch.sparse.mm(self.norm_adj, emb)
            random_noise = torch.rand(emb.shape)
            random_noise = random_noise.sign() * F.normalize(random_noise, dim=-1)
            emb += self.config['model']['noise'] * random_noise
            emb_list.append(emb)
        emb = torch.stack(emb_list, dim=1).mean(1)
        return emb

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

    def gnn_cl(self, batch):
        item_ids = torch.unique(batch[self.fiid])
        gnn_emb1 = self.get_gnn_embeddings()
        return

    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.augmentation_model = data_augmentation.CL4SRecAugmentation(self.config['model'], train_data)

    def training_step(self, batch, reduce=True, return_query=False, align=False):
        rst = super().training_step(batch, reduce=reduce, return_query=return_query)
        cl_output = self.augmentation_model(batch, self.query_encoder, reduce=reduce)
        cl_loss = self.config['model']['cl_weight'] * cl_output['cl_loss']
        if not reduce:
            if return_query:
                loss_value, query = rst
                return (loss_value, cl_loss), query
            else:
                loss_value = rst
                return loss_value, cl_loss
        else:
            if return_query:
                loss_value, query = rst
                loss_value += cl_loss
                return loss_value, query
            else:
                loss_value = rst + cl_loss
                return loss_value

