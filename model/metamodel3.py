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
from torch.distributions import Bernoulli

class MetaModel3(BaseModel):
    def __init__(self, config: Dict, dataset_list: List[PatternDataset]) -> None:
        super().__init__(config, dataset_list)
        self.interval = config['train']['interval']
        self.step_counter = 0
        self.item_embedding = None # MetaModel is just a trainer without item embedding
        self.tau = 10
        self.annealing_factor = 0.9999
        self.H = config['model']['H']
        self.alpha = 0.3

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
        return MLPModule([
            self.embed_dim,
            self.embed_dim,
            self.embed_dim,
            self.H * 2,
        ], dropout=0.5, last_activation=False)

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

    def generate_pattern(self, batch):
        def find_last_nonezero(x):
            idx = torch.arange(self.max_seq_len, device=self.device)
            tmp2 = torch.einsum("abc,b->abc", (x, idx))
            indices = torch.argmax(tmp2, dim=1, keepdim=True)
            return indices + 1
        def binarize(x, seq_len):
            x = torch.clamp(x, min=0, max=1)
            d = Bernoulli(x)
            rst = d.sample()
            mask = torch.arange(x.shape[1], device=self.device).unsqueeze(0) >= \
                seq_len.unsqueeze(-1)
            rst = rst.masked_fill(mask.unsqueeze(-1), 0)
            return rst + x - x.detach()
        seq_embs = self.item_embedding(batch['in_' + self.fiid]) # BLD
        B, L, D = seq_embs.shape

        logits = self.meta_module(seq_embs).reshape(B, L, self.H, 2) # BLH2
        logits = F.softmax(logits, dim=-1)[..., 0]# BLH

        # logits = self.meta_module(seq_embs)

        logits = logits + self.alpha
        self.alpha = max(self.alpha * 0.999, 0.5)
        selection = binarize(logits, batch['seqlen']) # BLH
        self.selection = selection
        user_emb = selection.unsqueeze(-1) * seq_embs.unsqueeze(-2) # BLHD
        user_emb = user_emb.permute(0, 2, 1, 3) # BHLD
        user_emb = user_emb.flatten(0, 1) # (B*H)LD

        # att_mask = self.get_attention_mask(selection.permute(0, 2, 1).flatten(0, 1))
        att_mask = self.get_attention_mask(batch['in_' + self.fiid]).repeat_interleave(repeats=self.H, dim=0)
        new_seq_len = find_last_nonezero(selection).flatten()
        self.log_value = ((batch['seqlen'].unsqueeze(-1) - selection.sum(1)) / batch['seqlen'].unsqueeze(-1))
        return user_emb, att_mask, new_seq_len

    def merge_pattern(self, pattern, method='mean'):
        # pattern: [B, H, D]
        if method == 'sum':
            rst = pattern.reshape(-1, self.H, self.embed_dim)
            rst = rst.sum(1)
        elif method == 'mean':
            rst = pattern.reshape(-1, self.H, self.embed_dim)
            rst = rst.mean(1)
        return rst

    def forward(self, batch):
        pattern, att_mask, last_seq_len = self.generate_pattern(batch)
        batch['pattern'] = pattern
        batch['att_mask'] = att_mask
        batch['seqlen'] = last_seq_len
        query = self.sub_model.forward(batch, need_pooling=True)
        query = self.merge_pattern(query)
        return query

    def current_epoch_trainloaders(self, nepoch):
        self.dataset_list[0].set_mode('all')
        return self.dataset_list[0].get_loader()

    def current_epoch_metaloaders(self, nepoch):
        self.dataset_list[0].set_mode('all')
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
                    self.dataset_list[0].set_mode('all')
            output_list.append(outputs)
        if nepoch > self.config['train']['warmup_epoch']:
            print(self.log_value.mean(1).mean())
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

    def training_step(self, batch, reduce=True, return_query=True, align=False):
        query = self.forward(batch)
        pos_score = (query * self.item_embedding.weight[batch[self.fiid]]).sum(-1)
        neg_score = (query.unsqueeze(-2) * self.item_embedding.weight[batch['neg_item']]).sum(-1)
        pos_score[batch[self.fiid] == 0] = -torch.inf # padding
        loss_value = self.sub_model.loss_fn(pos_score, neg_score, reduce=reduce)
        return loss_value
