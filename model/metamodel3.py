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
from utils import get_model_class, MetaOptimizer, normal_initialization, load_config
from collections import defaultdict
from model.loss_func import *
from data.dataset import *
from typing import Dict, List, Optional, Tuple
from module.layers import MLPModule

from model.basemodel import BaseModel

class MetaModel3(BaseModel):
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
        return MLPModule([
                self.embed_dim,
                self.embed_dim,
                1,
            ],
            dropout=0.5,
            activation_func='leakyrelu',
        )

    def _get_meta_optimizers(self):
        opt_name = self.config['train']['meta_optimizer']
        lr = self.config['train']['meta_learning_rate']
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

        optimizer = MetaOptimizer(optimizer, hpo_lr=1e-4)

        return optimizer

    def forward(self, batch):
        return self.sub_model.forward(batch)

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
                training_step_args = {'batch': batch}
                loss = self.training_step(**training_step_args)
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
            training_step_args = {'batch': batch}
            meta_train_loss = meta_train_loss + self.training_step(**training_step_args)
            break

        meta_loss = 0
        for batch_idx, batch in enumerate(self.current_epoch_metaloaders(nepoch)):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            batch['neg_item'] = self._neg_sampling(batch)
            training_step_args = {'batch': batch}
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
    def _multi_round_gumbel(logits, k, tau):
        rst = 0
        for _ in range(k):
            rst += F.gumbel_softmax(logits, tau=tau, hard=False)
        # rst = rst.clamp_max(max=1)
        return rst

    def selection(self, query, seq_len):
        logits = self.meta_module(query).squeeze()
        mask = torch.arange(query.shape[1], device=self.device).unsqueeze(0) >= seq_len.unsqueeze(-1)
        logits = logits.masked_fill(mask, -torch.inf)
        rst = []
        for idx, logit in enumerate(logits):
            rst.append(self._multi_round_gumbel(logit, seq_len[idx], self.tau))
        rst = torch.stack(rst)
        self.tau = max(self.tau * self.annealing_factor, 1)
        return rst

    def training_step(self, batch, reduce=True, return_query=True):
        query = self.sub_model.forward(batch, need_pooling=False)
        batch['input_weight'] = self.selection(query, batch['seqlen'])
        loss_value = self.sub_model.training_step(batch, reduce=True, return_query=False)
        return loss_value
