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

class MetaModel8(BaseModel):
    def __init__(self, config: Dict, dataset_list: List[BaseDataset]) -> None:
        super().__init__(config, dataset_list)
        self.step_counter = 0
        self.item_embedding = None # MetaModel is just a trainer without item embedding
        self.counter = 0
        sub_model = 'SASRec'
        path_loss_weight = f'paper/loss_weight_{sub_model}_toys.pth'
        self.loss_weight = torch.load(path_loss_weight, map_location=self.device)
        path_logits = f'paper/logits_{sub_model}_toys.pth'
        self.logits, self.tau = torch.load(path_logits, map_location=self.device)

    def _init_model(self, train_data):
        self.sub_model : BaseModel = self._register_sub_model()
        self.sub_model._init_model(train_data)
        self.item_embedding = self.sub_model.item_embedding

    def _register_sub_model(self) -> BaseModel:
        sub_model_config = {
            'dataset': self.config['data']['dataset'],
            'model': self.config['model']['sub_model']
        }
        sub_model_config = load_config(sub_model_config)
        sub_model_config['train']['device'] = 0
        self.logger.info(sub_model_config)
        model_class = get_model_class(sub_model_config['model'])
        return model_class(sub_model_config, self.dataset_list)

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
                loss = self.training_step(**training_step_args)
                loss.backward()
                self.sub_model.optimizer.step()
                outputs.append({f"loss_{loader_idx}": loss.detach()})
            output_list.append(outputs)
        return output_list

    def forward(self, batch):
        return self.sub_model.forward(batch)

    def training_step(self, batch, reduce=True, return_query=True, align=False):
        loss_value = self.sub_model.training_step(batch, reduce=False, return_query=False, align=False)

        # weight = self.loss_weight[batch['index']]

        weight = F.gumbel_softmax(self.logits[batch['index']], tau=self.tau, dim=-1)[..., 0]
        mask = batch['user_id'] == 0
        weight = weight.masked_fill(mask.unsqueeze(-1), 1)
        pad_mask = batch['item_id'] == 0
        weight = weight.masked_fill(pad_mask, 0)

        if not isinstance(loss_value, tuple):
            loss_value = (loss_value * weight).sum()
        else: # For CL4SRec
            rst = (loss_value[0] * weight).sum()
            # rst += (loss_value[1] * weight).sum()
            rst += loss_value[1].sum()
            loss_value = rst
        return loss_value
