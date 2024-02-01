import torch
import torch.nn as nn
from data.dataset import BaseDataset
from module import data_augmentation
from model.sasrec import SASRec, SASRecQueryEncoder
from data import dataset
from utils import prepare_datasets
from copy import deepcopy

r"""
CL4SRec
#############
    Contrastive Learning for Sequential Recommendation(SIGIR'21)
    Reference:
        https://arxiv.org/abs/2010.14395
"""
class CL4SRec2(SASRec):
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
        aug_config = deepcopy(config)
        aug_config['data']['train_file'] = '_ori'
        self.aug_dataset_list = prepare_datasets(aug_config)
        self.aug_loader = iter(self.aug_dataset_list[0])

    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.augmentation_model = data_augmentation.CL4SRecAugmentation(self.config['model'], train_data)

    def training_step(self, batch, reduce=True, return_query=False, align=False):
        rst = super().training_step(batch, reduce=reduce, return_query=return_query)
        try:
            aug_batch = next(self.aug_loader)
        except StopIteration:
            self.aug_loader = iter(self.aug_dataset_list[0])
            aug_batch = next(self.aug_loader)
        cl_output = self.augmentation_model(aug_batch, self.query_encoder, reduce=reduce)
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

