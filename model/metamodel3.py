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
        # transformer_encoder = torch.nn.TransformerEncoderLayer(
        #     d_model=self.embed_dim,
        #     nhead=2,
        #     dim_feedforward=4 * self.embed_dim,
        #     dropout=0.5,
        #     activation='gelu',
        #     layer_norm_eps=1e-12,
        #     batch_first=True,
        #     norm_first=False
        # )
        # self.transformer_layer = torch.nn.TransformerEncoder(
        #     encoder_layer=transformer_encoder,
        #     num_layers=1,
        # )
        self.dropout = nn.Dropout(0.5)
        self.query_transform = nn.Identity()
        self.mlp = nn.Sequential(
            MLPModule([
                    self.embed_dim,
                    self.embed_dim,
                ],
                dropout=0.5,
                activation_func='relu',
            ),
            nn.Linear(self.embed_dim, 2),
        )
        self.scorer = nn.Sequential(
            nn.Sigmoid(),
        )
        self.gumbel_selector = SubsetOperator(5, True)
        self.alpha = 0.
        self.tau = nn.Parameter(torch.ones(1, device=config['train']['device']) * 10)

    def forward(self, batch, input_emb, position_emb, query=None):
        user_hist = batch['in_item_id']
        seq_len = batch['seqlen']
        device = input_emb.weight.device
        L = user_hist.shape[1]
        if query is None:
            positions = torch.arange(user_hist.size(1), dtype=torch.long, device=device)
            positions = positions.unsqueeze(0).expand_as(user_hist)
            position_embs = position_emb(positions)
            seq_embs = input_emb(user_hist)
            query = self.dropout(position_embs + seq_embs)
            mask4padding = user_hist == 0
            attention_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=device), 1)
            query = self.transformer_layer(
                src=query,
                mask=attention_mask,
                src_key_padding_mask=mask4padding,
            )

        logits = self.mlp(query) # [B, L, 2]

        # -------Sigmoid---------
        # score = self.scorer(logits) + self.alpha
        # score_hard = (score >= 0.49).float()
        # selection = score_hard + score - score.detach()
        # self.alpha = self.alpha * 0.999

        # -----Gumbel softmax----
        score = F.gumbel_softmax(logits, tau=torch.clip(self.tau, min=2), dim=-1, hard=True)
        # score_hard = (score >= (1 / seq_len.reshape(-1, 1, 1) * 0.2)).float()
        # selection = score_hard + score - score.detach()
        # # selection = self.gumbel_selector(logits.squeeze(), self.tau).unsqueeze(-1)
        selection = score[:, :, 0:1]
        pattern_mask = batch['user_id'] == 0
        selection = selection.masked_fill(pattern_mask.reshape(-1, 1, 1), 1)
        mask = torch.arange(query.shape[1], device=device).unsqueeze(0) >= seq_len.unsqueeze(-1)
        selection = selection.masked_fill(mask.unsqueeze(-1), 0)
        
        return selection

        logits = self.meta_module(query).squeeze()
        mask = torch.arange(query.shape[1], device=self.device).unsqueeze(0) >= seq_len.unsqueeze(-1)
        logits = logits.masked_fill(mask, -torch.inf)
        rst = []
        # rst = self._multi_round_gumbel(logits, 10, self.tau)
        rst = self.gumbel_selector(logits, self.tau) * seq_len.unsqueeze(1) / self.max_seq_len
        self.tau = max(self.tau * self.annealing_factor, 1)
        return rst

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

    def training_step(self, batch, reduce=True, return_query=True):
        query = self.sub_model.forward(batch, need_pooling=False)
        batch['input_weight'] = self.meta_module(
            # query,
            batch,
            self.sub_model.item_embedding,
            self.sub_model.query_encoder.position_emb,
            query,
        )
        loss_value = self.sub_model.training_step(batch, reduce=True, return_query=False)
        return loss_value
