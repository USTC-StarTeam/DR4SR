import torch
import numpy as np
import torch.nn as nn
from model.basemodel import BaseModel
from model.teacher import Teacher
from module.layers import SeqPoolingLayer
from data import dataset
from utils.utils import flatten_state_dict, get_model_class
import torch.nn.functional as F
from model.sasrec import SASRec
from collections import OrderedDict
import random
from copy import deepcopy
import wandb
from collections import defaultdict
from utils import callbacks

from utils.reparam_module import ReparamModule

class SASRecDM(BaseModel):
    def __init__(self, config, dataset_list : list[dataset.BaseDataset]) -> None:
        super().__init__(config, dataset_list)
        # self.synthetic_size = int(self.num_users * 0.5)
        self.synthetic_size = int(256 * 10)
        self.sub_model = ReparamModule(self.create_submodel())

    def _init_model(self, train_data):
        super()._init_model(train_data)

        self.item_embedding.requires_grad_(False) # item embedding of meta model is not used
        self.synthetic_dataset = nn.Embedding(self.synthetic_size, self.num_items, device=self.device)
        nn.init.xavier_uniform_(self.synthetic_dataset.weight)
        # self.synthetic_optimizer = torch.optim.SGD(self.synthetic_dataset.parameters(), lr=0.001, momentum=0.5)
        self.synthetic_optimizer = torch.optim.Adam(self.synthetic_dataset.parameters(), lr=0.001)
        self.experts = torch.load('saved/Teacher/amazon-toys/2023-12-26-14-39-21-629424-timestamp.pth', map_location='cpu')

    def create_submodel(self):
        sub_model_config = deepcopy(self.config)
        sub_model_config['model']['model'] = 'SASRec2'
        model_class = get_model_class(sub_model_config['model'])
        return model_class(sub_model_config, self.dataset_list)

    def sampling(self, logits, item_emb):
        sample_rst = []
        for _ in range(self.max_seq_len + 1):
            sample_rst.append(F.gumbel_softmax(logits, tau=1, hard=False))
        sample_rst = torch.stack(sample_rst, 1)

        # sample_rst = logits.unsqueeze(1).repeat(1, self.max_seq_len + 1, 1)

        seq_embs = sample_rst[:, :-1] @ item_emb

        target_item = sample_rst[:, 1:] @ item_emb
        return seq_embs, target_item



    def _remove_meta_parameters(self, parameters):
        parameters_new = {}
        for k, v in parameters.items():
            if 'synthetic' not in k:
                parameters_new[k] = v
        return parameters_new

    def _state_dict_to_device(self, state_dict):
        for k, v in state_dict.items():
            state_dict[k] = v.to(self.device)
        return state_dict

    @staticmethod
    def alignment(x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    @staticmethod
    def uniformity(x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def train_start(self):
        for _ in range(50):
            # train synthetic dataset
            self.sub_model = self.create_submodel().to(self.device)
            self.sub_model.train()
            self.synthetic_dataset.train()
            self.synthetic_dataset.requires_grad_(True)
            for outer_idx in range(50):
                
                x = self.synthetic_dataset.weight
                mask = torch.zeros(self.num_items, dtype=torch.bool, device=self.device)
                mask[0] = 1
                x = x.masked_fill(mask.unsqueeze(0), -torch.inf)
                item_emb = self.sub_model.item_embedding.weight
                seq_embs, target_item = self.sampling(x, item_emb)

                batch = {
                    'seq_embs': seq_embs,
                    self.fiid: target_item,
                    'seqlen': self.max_seq_len * torch.ones(seq_embs.shape[0], dtype=int, device=self.device)
                }

                query_syn = self.sub_model.forward(batch)
                batch = {
                    'in_' + self.fiid: self.dataset_list[0].data[1],
                    self.fiid: self.dataset_list[0].data[2],
                    'seqlen': self.dataset_list[0].data[3]
                }
                query_real = self.sub_model.forward_eval(batch).detach()
                loss = torch.sum((torch.mean(query_real, dim=0) - torch.mean(query_syn, dim=0))**2)
                
                self.synthetic_optimizer.zero_grad()
                loss.backward()

                self.synthetic_optimizer.step()

            # eval synthetic dataset
            self.sub_model = self.create_submodel().to(self.device)
            self.synthetic_dataset.requires_grad_(False)
            self.sub_model.requires_grad_(True)
            nepoch = 0
            model_opt = torch.optim.Adam(self.sub_model.parameters())
            for e in range(10):
                self.logged_metrics = {}
                self.logged_metrics['epoch'] = nepoch

                # training procedure
                self.train()
                output_list = []
                indices_chunks = []
                outputs = []
                for _ in range(100):
                    if not indices_chunks:
                        indices = torch.randperm(self.synthetic_dataset.weight.shape[0])
                        indices_chunks = list(torch.split(indices, 256)) # greater than teacher
                    these_indices = indices_chunks.pop()
                    x = self.synthetic_dataset.weight[these_indices]
                    mask = torch.zeros(self.num_items, dtype=torch.bool, device=self.device)
                    mask[0] = 1
                    # x = x.masked_fill(mask.unsqueeze(0), -torch.inf)
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
                    neg_idx = torch.multinomial(weight, self.max_seq_len, replacement=True)
                    neg_idx = neg_idx.reshape(query.shape[0], self.max_seq_len, 1)
                    neg_score = (query.unsqueeze(-2) * item_emb[neg_idx]).sum(-1)
                    loss = self.loss_fn(pos_score, neg_score)

                    model_opt.zero_grad()
                    loss.backward()
                    model_opt.step()
                    outputs.append({f"loss_{1}": loss.detach()})
                output_list.append(outputs)

                # validation procedure
                self.eval()
                if nepoch % 1 == 0:
                    for domain in self.domain_name_list:
                        val_dataset = self.dataset_list[1]
                        val_dataset.set_eval_domain(domain)
                        self.set_eval_domain(domain)
                        validation_output_list = self.validation_epoch(nepoch, val_dataset.get_loader())
                        self.validation_epoch_end(validation_output_list, domain)
                    all_domain_result = defaultdict(float)
                    for k, v in self.logged_metrics.items():
                        for domain_name in self.domain_name_list:
                            if domain_name in k: # result of a single domain
                                all_domain_result[k.removeprefix(domain_name + '_')] += v
                                break
                    self.logged_metrics.update(all_domain_result)
                    wandb.log(all_domain_result)
                self.training_epoch_end(output_list)

                stop_training = self.callback(self, nepoch, self.logged_metrics)
                if stop_training:
                    break
                nepoch += 1

            self.training_end()
            self.callback = callbacks.EarlyStopping(self, 'ndcg@20', self.config['data']['dataset'], patience=self.config['train']['early_stop_patience'])


    def forward(self, batch):
        # for eval
        return self.sub_model.forward_eval(batch)