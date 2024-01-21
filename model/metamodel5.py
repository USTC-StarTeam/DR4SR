import os
import wandb
import torch
import logging
import time
import evaluation
from data.dataset import BaseDataset, PatternDataset
from copy import deepcopy

from tqdm import tqdm
from torch import optim
from utils import get_model_class, MetaOptimizer, normal_initialization, load_config, SubsetOperator
from collections import defaultdict
from model.loss_func import *
from data.dataset import *
from typing import Dict, List, Optional, Tuple
from module.layers import MLPModule

from model.basemodel import BaseModel

class Selector(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.embed_dim = config['model']['embed_dim']
        transformer_encoder = torch.nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=2,
            dim_feedforward=2 * self.embed_dim,
            dropout=0.5,
            activation='gelu',
            layer_norm_eps=1e-12,
            batch_first=True,
            norm_first=False
        )
        self.query_encoder_twin = torch.nn.TransformerEncoder(
            encoder_layer=transformer_encoder,
            num_layers=2,
        )
        self.dropout = nn.Dropout(0.5)
        # self.query_transform = nn.Identity()
        # self.mlp = nn.Sequential(
        #     MLPModule([
        #             self.embed_dim,
        #             self.embed_dim,
        #         ],
        #         dropout=0.5,
        #         activation_func='relu',
        #     ),
        #     nn.Linear(self.embed_dim, 2),
        # )
        # self.scorer = nn.Sequential(
        #     nn.Sigmoid(),
        # )
        # self.gumbel_selector = SubsetOperator(5, True)
        self.alpha = 0.
        # self.tau = nn.Parameter(torch.ones(1, device=config['train']['device']) * 10)
        self.tau = 10
        self.L = 1
        self.condition_proj = nn.Identity()

    def forward(self, batch, query):
        B, D = query.shape
        device = query.device
        query_c = self.condition_proj(query).reshape(B, self.L, D)
        query_c = self.dropout(query_c)
        attention_mask = torch.triu(torch.ones((self.L, self.L), dtype=torch.bool, device=device), 1)
        query_q = self.query_encoder_twin(
            src=query_c,
            mask=attention_mask,
        )[:, self.L - 1]
        return query_q

class MetaModel5(BaseModel):
    def __init__(self, config: Dict, dataset_list: List[PatternDataset]) -> None:
        super().__init__(config, dataset_list)
        self.interval = config['train']['interval']
        self.step_counter = 0
        self.item_embedding = None # MetaModel is just a trainer without item embedding
        self.tau = 10
        self.annealing_factor = 0.9999

    def _init_model(self, train_data):
        self.sub_model : BaseModel = self._register_sub_model()
        self.sub_model._init_model(train_data)
        self.item_embedding = self.sub_model.item_embedding

        self.meta_module : nn.Module = self._register_meta_modules()
        self.meta_module = self.meta_module.to(self.device)
        self.meta_module.apply(normal_initialization)

        self.meta_optimizer = self._get_meta_optimizers()
        self.metaloader_iter = iter(self.current_epoch_metaloaders(nepoch=0))

        self.gumbel_selector = SubsetOperator(k=self.max_seq_len)

    def _register_sub_model(self) -> BaseModel:
        sub_model_config = {
            'dataset': self.config['data']['dataset'],
            'model': self.config['model']['sub_model']
        }
        sub_model_config = load_config(sub_model_config)
        sub_model_config['train']['device'] = 0
        model_class = get_model_class(sub_model_config['model'])
        return model_class(sub_model_config, self.dataset_list)

    def _register_meta_modules(self) -> nn.Module:
        return Selector(self.config)

    def _get_meta_optimizers(self):
        opt_name = self.config['train']['meta_optimizer']
        lr = self.config['train']['meta_learning_rate']
        hpo_lr = self.config['train']['hpo_learning_rate']
        weight_decay = self.config['train']['meta_weight_decay']
        params = self.meta_module.parameters()

        if opt_name.lower() == 'adam':
            optimizer = optim.Adam(params, lr=lr)
        elif opt_name.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif opt_name.lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=lr)
        elif opt_name.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=lr)
        elif opt_name.lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params, lr=lr)
        else:
            optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)

        optimizer = MetaOptimizer(optimizer, hpo_lr=hpo_lr)

        return optimizer

    def forward(self, batch):
        query = self.sub_model.forward(batch, need_pooling=True)
        self.query = query
        query_q = self.meta_module(
            batch,
            query,
        )
        return query_q

    def current_epoch_trainloaders(self, nepoch):
        self.dataset_list[0].set_mode('all')
        return self.dataset_list[0].get_loader()

    def current_epoch_metaloaders(self, nepoch):
        self.dataset_list[0].set_mode('original')
        return self.dataset_list[0].get_loader()

    def training_epoch(self, nepoch):
        output_list = []

        trn_dataloaders = self.current_epoch_trainloaders(nepoch)
        trn_dataloaders = [trn_dataloaders]

        for loader_idx, loader in enumerate(trn_dataloaders):
            outputs = []
            loader = tqdm(
                loader,
                total=len(loader),
                ncols=75,
                desc=f"Training {nepoch:>5}",
                leave=False,
            )
            for batch_idx, batch in enumerate(loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch['neg_item'] = self._neg_sampling(batch)
                self.sub_model.optimizer.zero_grad()
                training_step_args = {'batch': batch, 'align': True}
                if nepoch > self.config['train']['warmup_epoch']:
                    loss = self.training_step(**training_step_args)
                else:
                    loss = self.sub_model.training_step(**training_step_args)
                loss.backward()
                self.sub_model.optimizer.step()
                outputs.append({f"loss_{loader_idx}": loss.detach()})
                self.step_counter += 1
                if self.step_counter % self.config['train']['interval'] == 0 and \
                    nepoch > self.config['train']['warmup_epoch']:
                    self._outter_loop(nepoch)
                    self.dataset_list[0].set_mode('all')
            output_list.append(outputs)
        return output_list

    def _outter_loop(self, nepoch):
        # Update dataset
        meta_train_loss = 0
        for batch_idx, batch in enumerate(self.current_epoch_trainloaders(nepoch)):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            batch['neg_item'] = self._neg_sampling(batch)
            training_step_args = {'batch': batch, 'align': False}
            meta_train_loss = meta_train_loss + self.training_step(**training_step_args)
            break

        meta_loss = 0
        for batch_idx, batch in enumerate(self.current_epoch_metaloaders(nepoch)):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            batch['neg_item'] = self._neg_sampling(batch)
            training_step_args = {'batch': batch, 'align': False}
            meta_loss = meta_loss + self.sub_model.training_step(**training_step_args)
            break

        self.meta_optimizer.step(
            val_loss=meta_loss,
            train_loss=meta_train_loss,
            aux_params = list(self.meta_module.parameters()),
            parameters = list(self.sub_model.parameters()),
            return_grads = False,
        )

    @staticmethod
    def alignment(x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    @staticmethod
    def uniformity(x):
        x = F.normalize(x, dim=-1)
        # return torch.norm(x[:, None] - x, dim=2, p=2).pow(2).mul(-2).exp().mean().log()
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def training_step(self, batch, reduce=True, return_query=True, align=True):
        query_q = self.forward(batch)
        if align:
            alignment = 0
            # alignment += self.alignment(self.query, query_q)
            alignment += 0.5 * self.alignment(self.query, self.item_embedding.weight[batch[self.fiid]])
            alignment += 0.5 * self.alignment(query_q, self.item_embedding.weight[batch[self.fiid]])
            uniformity = self.config['model']['uniformity'] * (
                self.uniformity(self.query) +
                self.uniformity(query_q) +
                self.uniformity(self.item_embedding.weight[batch[self.fiid]])
            )
            loss_value = alignment + uniformity
        else:
            pos_score = (query_q * self.item_embedding.weight[batch[self.fiid]]).sum(-1)
            neg_score = (query_q.unsqueeze(1) * self.item_embedding.weight[batch['neg_item']]).sum(-1)
            pos_score[batch[self.fiid] == 0] = -torch.inf # padding
            loss_value = self.sub_model.loss_fn(pos_score, neg_score, reduce=reduce)


        return loss_value
