import torch
import torch.nn as nn
from data.dataset import BaseDataset
from module import data_augmentation
from model.sasrec import SASRec, SASRecQueryEncoder
from data import dataset

r"""
CL4SRec
#############
    Contrastive Learning for Sequential Recommendation(SIGIR'21)
    Reference:
        https://arxiv.org/abs/2010.14395
"""
class CL4SRec(SASRec):
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

    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.augmentation_model = data_augmentation.CL4SRecAugmentation(self.config['model'], train_data)

    def _get_dataset_class():
        return dataset.CondenseDataset
        return dataset.SeparateDataset
        return dataset.MixDataset

    def training_step(self, batch):
        query = self.forward(batch)
        pos_score = (query * self.item_embedding.weight[batch[self.fiid]]).sum(-1)
        neg_score = (query.unsqueeze(-2) * self.item_embedding.weight[batch['neg_item']]).sum(-1)
        pos_score[batch[self.fiid] == 0] = -torch.inf # padding

        loss_value = self.loss_fn(pos_score, neg_score)

        cl_output = self.augmentation_model(batch, self.query_encoder)
        loss_value += self.config['model']['cl_weight'] * cl_output['cl_loss']
        return loss_value

