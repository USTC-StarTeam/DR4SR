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

from utils.reparam_module import ReparamModule

class Student(BaseModel):
    def __init__(self, config, dataset_list : list[dataset.BaseDataset]) -> None:
        super().__init__(config, dataset_list)
        # self.synthetic_size = int(self.num_users * 0.5)
        self.synthetic_size = int(256 * 20)

        sub_model_config = deepcopy(config)
        sub_model_config['model']['model'] = 'SASRec2'
        model_class = get_model_class(sub_model_config['model'])
        self.sub_model = ReparamModule(model_class(sub_model_config, dataset_list))

    def _init_model(self, train_data):
        super()._init_model(train_data)

        self.item_embedding.requires_grad_(False) # item embedding of meta model is not used
        self.synthetic_dataset = nn.Embedding(self.synthetic_size, self.num_items, device=self.device)
        nn.init.xavier_uniform_(self.synthetic_dataset.weight)
        # self.synthetic_optimizer = torch.optim.SGD(self.synthetic_dataset.parameters(), lr=0.001, momentum=0.5)
        self.synthetic_optimizer = torch.optim.Adam(self.synthetic_dataset.parameters(), lr=0.001)
        self.experts = torch.load('saved/Teacher/amazon-toys/2023-12-26-14-39-21-629424-timestamp.pth', map_location='cpu')


    def sampling(self, logits):
        sample_rst = []
        for _ in range(self.max_seq_len + 1):
            sample_rst.append(F.gumbel_softmax(logits, tau=1, hard=False))
        sample_rst = torch.stack(sample_rst, 1)

        seq_embs = sample_rst[:, :-1] @ self.item_embedding.weight

        target_item = sample_rst[:, 1:] @ self.item_embedding.weight
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
        self.train()

        for outer_idx in range(50):
            start_epoch = np.random.randint(0, 25)
            starting_params = self.experts[start_epoch]
            target_params = self.experts[start_epoch+2]
            target_params = torch.cat([p.data.reshape(-1) for p in target_params]).to(self.device)
            student_params = [torch.cat([p.data.reshape(-1) for p in starting_params]).to(self.device).requires_grad_(True)]
            starting_params = torch.cat([p.data.reshape(-1) for p in starting_params]).to(self.device)
            # target_params = self._state_dict_to_device(self.experts[-1])
            # student_params = [self.experts[0]]
            # starting_params = self._state_dict_to_device(self.experts[0])
            # self.load_state_dict(self.experts[0], strict=False)
            param_loss_list = []
            param_dist_list = []
            indices_chunks = []

            for _ in range(40):
                if not indices_chunks:
                    indices = torch.randperm(self.synthetic_dataset.weight.shape[0])
                    indices_chunks = list(torch.split(indices, 256)) # greater than teacher
                these_indices = indices_chunks.pop()
                x = self.synthetic_dataset.weight[these_indices]
                mask = torch.zeros(self.num_items, dtype=torch.bool, device=self.device)
                mask[0] = 1
                x = x.masked_fill(mask.unsqueeze(0), -torch.inf)
                seq_embs, target_item = self.sampling(x)

                batch = {
                    'seq_embs': seq_embs,
                    self.fiid: target_item,
                    'seqlen': self.max_seq_len * torch.ones(seq_embs.shape[0], dtype=int, device=self.device)
                }

                query = self.sub_model.forward(batch, flat_param=student_params[-1])
                pos_score = (query * target_item).sum(-1)
                weight = torch.ones(query.shape[0], self.num_items, device=self.device)
                neg_idx = torch.multinomial(weight, self.max_seq_len, replacement=True)
                neg_idx = neg_idx.reshape(query.shape[0], self.max_seq_len, 1)
                item_emb = student_params[-1][:self.num_items * self.embed_dim].reshape(self.num_items, self.embed_dim)
                neg_score = (query.unsqueeze(-2) * item_emb[neg_idx]).sum(-1)
                loss_value = self.loss_fn(pos_score, neg_score)

                grad = torch.autograd.grad(loss_value, student_params[-1], create_graph=True)[0]
                student_params.append(student_params[-1] - 0.001 * grad)

            param_loss = torch.tensor(0.0).to(self.device)
            param_dist = torch.tensor(0.0).to(self.device)
            
            param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
            param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

            num_params = sum([np.prod(p.size()) for _, p in (self._remove_meta_parameters(self.state_dict())).items()])
            param_loss /= num_params
            param_dist /= num_params

            param_loss /= param_dist

            grand_loss = param_loss
            
            self.synthetic_optimizer.zero_grad()
            grand_loss.backward()

            self.synthetic_optimizer.step()
            for _ in student_params:
                del _

    # def training_step(self, batch):
    #     user_embed = self.forward(batch).flatten(1)
    #     item_embed = batch[self.fiid].flatten(1)

        
    #     align = self.alignment(user_embed, item_embed)
    #     uniform = (self.uniformity(user_embed) + self.uniformity(item_embed)) / 2
    #     loss_value = align + 3 * uniform

    #     return loss_value