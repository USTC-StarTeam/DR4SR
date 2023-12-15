import torch
import torch.nn as nn
import torch.nn.functional as F
from model.basemodel import BaseModel
from module.layers import HStackLayer, VStackLayer, LambdaLayer, GRULayer, SeqPoolingLayer
from data import dataset

class GRU4Rec(BaseModel):
    def __init__(self, config, dataset_list : list[dataset.BaseDataset]) -> None:
        super().__init__(config, dataset_list)
        model_config = self.config['model']
        self.query_encoder = (
            VStackLayer(
                torch.nn.Sequential(
                    LambdaLayer(lambda x: x['in_'+self.fiid]),
                    self.item_embedding,
                    torch.nn.Dropout(model_config['dropout_rate']),
                    GRULayer(self.embed_dim, model_config['hidden_size'], model_config['layer_num']),
                ),
                torch.nn.Linear(model_config['hidden_size'], self.embed_dim)
            )
        )
        self.training_pooling_layer = SeqPoolingLayer(pooling_type='origin')
        self.eval_pooling_layer = SeqPoolingLayer(pooling_type='last')

    def _get_dataset_class():
        return dataset.CondenseDataset
        return dataset.SeparateDataset
        return dataset.MixDataset

    def forward(self, batch):
        gru4rec_out = self.query_encoder(batch)
        if self.training:
            return self.training_pooling_layer(gru4rec_out, batch['seqlen'])
        else:
            return self.eval_pooling_layer(gru4rec_out, batch['seqlen'])
