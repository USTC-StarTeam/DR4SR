import os
import wandb
import torch
import logging
import time
import evaluation
from data.dataset import BaseDataset
from copy import deepcopy

from tqdm import tqdm
from torch import optim
from utils import get_model_class, MetaOptimizer, normal_initialization, load_config
from collections import defaultdict
from model.loss_func import *
from data.dataset import *
from typing import Dict, List, Optional, Tuple

from model.basemodel import BaseModel

class MetaModel7(BaseModel):
    def __init__(self, config: Dict, dataset_list: List[BaseDataset]) -> None:
        super().__init__(config, dataset_list)
        self.interval = config['train']['interval']
        self.step_counter = 0
        self.item_embedding = None # MetaModel is just a trainer without item embedding
        self.tau = nn.Parameter(torch.ones(1, device=self.device) * 10)
        self.counter = 0

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
        return nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 2),
        )

    def _get_meta_optimizers(self):
        opt_name = self.config['train']['meta_optimizer']
        lr = self.config['train']['meta_learning_rate']
        hpo_lr = self.config['train']['hpo_learning_rate']
        weight_decay = self.config['train']['meta_weight_decay']
        params = list(self.meta_module.parameters()) + [self.tau]

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
        return self.sub_model.forward(batch)

    def current_epoch_metaloaders(self, nepoch):
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
                training_step_args = {'batch': batch, 'align': False}
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
            output_list.append(outputs)
        return output_list

    def _outter_loop(self, nepoch):
        metaloader_iter = iter(self.current_epoch_metaloaders(nepoch))
        batch = next(metaloader_iter)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        batch['neg_item'] = self._neg_sampling(batch)
        training_step_args = {'batch': batch, 'align': False}
        meta_loss = self.sub_model.training_step(**training_step_args)

        trainloader = self.current_epoch_trainloaders(nepoch)
        batch = next(iter(trainloader))
        batch = {k: v.to(self.device) for k, v in batch.items()}
        batch['neg_item'] = self._neg_sampling(batch)
        training_step_args = {'batch': batch, 'align': False}
        meta_train_loss = self.training_step(**training_step_args)

        self.meta_optimizer.step(
            val_loss=meta_loss,
            train_loss=meta_train_loss,
            aux_params = list(self.meta_module.parameters()),
            parameters = list(self.sub_model.parameters()),
            return_grads = False,
        )

    def selection(self, query):
        logits = self.meta_module(query)
        logits = F.gumbel_softmax(logits, tau=torch.clip(self.tau, min=self.config['model']['tau_min']), dim=-1, hard=False)[..., 0]
        return logits.squeeze()

    def training_step(self, batch, reduce=True, return_query=True, align=False):
        loss_value, query = self.sub_model.training_step(batch, reduce=False, return_query=True, align=False)
        weight = self.selection(query)
        # weight = self.selection(query) * query.shape[0]
        mask = batch['user_id'] == 0
        weight = weight.masked_fill(mask.unsqueeze(-1), 1)
        pad_mask = batch[self.fiid] == 0
        weight = weight.masked_fill(pad_mask, 0)
        if self.counter % 500 == 0:
            torch.set_printoptions(precision=3, sci_mode=False)
            self.logger.info(weight)
            torch.set_printoptions(precision=4, sci_mode=False)
        self.counter += 1
        if not isinstance(loss_value, tuple):
            loss_value = (loss_value * weight).sum()
        else: # For CL4SRec
            rst = (loss_value[0] * weight).sum()
            # rst += (loss_value[1] * weight).sum()
            rst += loss_value[1].sum()
            loss_value = rst
        return loss_value

    def evaluate(self) -> Dict:
        return super().evaluate()
