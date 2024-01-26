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

class MetaModel4(BaseModel):
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

        syn_dataset = torch.zeros(20000, self.max_seq_len + 1, self.embed_dim, device=self.device)
        nn.init.normal_(syn_dataset, std=0.02)
        self.syn_dataset = nn.Parameter(syn_dataset)

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
        # transformer_layer = torch.nn.TransformerEncoder(
        #     encoder_layer=transformer_encoder,
        #     num_layers=1,
        # )
        syn_dataset = torch.zeros(20000, self.max_seq_len + 1, self.embed_dim, device=self.device)
        nn.init.normal_(syn_dataset, std=0.02)
        return nn.Parameter(syn_dataset)
        return nn.ModuleDict({
            'syn_dataset': nn.Parameter(syn_dataset),
            'seq_encoder': transformer_layer,
        })

    def _get_meta_optimizers(self):
        opt_name = self.config['train']['meta_optimizer']
        lr = self.config['train']['meta_learning_rate']
        hpo_lr = self.config['train']['hpo_learning_rate']
        weight_decay = self.config['train']['meta_weight_decay']
        params = [self.syn_dataset]

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
        self.dataset_list[0].set_mode('all')
        return self.dataset_list[0].get_loader()

    def sampling(self, user_emb, item_emb):
        L = self.max_seq_len
        # attention_mask = torch.triu(torch.ones((L + 1, L + 1), dtype=torch.bool, device=self.device), 1)
        # user_emb = self.meta_module['seq_encoder'](
        #     user_emb,
        #     mask=attention_mask,
        # )
        logits = user_emb @ item_emb.T # [B, L, I]
        sample_rst = F.gumbel_softmax(logits, tau=10, dim=-1) # [B, L, I]

        seq_embs = sample_rst[:, :-1] @ item_emb # [B, L, D]

        target_item = sample_rst[:, -1] @ item_emb # [B, D]
        return seq_embs, target_item

    def training_epoch(self, nepoch):
        output_list = []
        outputs = []
        if nepoch < self.config['train']['warmup_epoch']:
            trn_dataloader = self.current_epoch_trainloaders(nepoch)
            for batch_idx, batch in enumerate(tqdm(trn_dataloader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch['neg_item'] = self._neg_sampling(batch)
                self.sub_model.optimizer.zero_grad()
                training_step_args = {'batch': batch}
                loss = self.sub_model.training_step(**training_step_args)
                loss.backward()
                self.sub_model.optimizer.step()
                outputs.append({f"loss_1": loss.detach()})
        else:
            indices = torch.randperm(self.syn_dataset.shape[0])
            indices_chunks = list(torch.split(indices, 256)) # greater than teacher
            for indices in tqdm(indices_chunks):
                x = self.syn_dataset[indices]
                item_emb = self.sub_model.item_embedding.weight
                seq_embs, target_item = self.sampling(x, item_emb)
                batch = {
                    'seq_embs': seq_embs,
                    self.fiid: target_item,
                    'seqlen': self.max_seq_len * torch.ones(seq_embs.shape[0], dtype=int, device=self.device)
                }
                query = self.sub_model.forward(batch)
                pos_score = (query * target_item).sum(-1)
                weight = torch.ones(query.shape[0], self.num_items, device=self.device)
                neg_idx = torch.multinomial(weight, 1, replacement=True)
                neg_score = (query.unsqueeze(1) * item_emb[neg_idx]).sum(-1)
                loss = self.sub_model.loss_fn(pos_score, neg_score)
                self.sub_model.optimizer.step()
                outputs.append({f"loss_1": loss.detach()})
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
        indices = torch.randperm(self.syn_dataset.shape[0])
        indices_chunks = list(torch.split(indices, 256)) # greater than teacher
        for indices in tqdm(indices_chunks):
            x = self.syn_dataset[indices]
            item_emb = self.sub_model.item_embedding.weight
            seq_embs, target_item = self.sampling(x, item_emb)
            batch = {
                'seq_embs': seq_embs,
                self.fiid: target_item,
                'seqlen': self.max_seq_len * torch.ones(seq_embs.shape[0], dtype=int, device=self.device)
            }
            query = self.sub_model.forward(batch)
            pos_score = (query * target_item).sum(-1)
            weight = torch.ones(query.shape[0], self.num_items, device=self.device)
            neg_idx = torch.multinomial(weight, 1, replacement=True)
            neg_score = (query.unsqueeze(1) * item_emb[neg_idx]).sum(-1)
            meta_train_loss = meta_train_loss + self.sub_model.loss_fn(pos_score, neg_score)
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
            aux_params = [self.syn_dataset],
            parameters = list(self.sub_model.parameters()),
            return_grads = False,
        )
