import torch
import torch.nn as nn
import torch.nn.functional as F
from model.basemodel import BaseModel
from module.layers import SeqPoolingLayer, SeqVectorQuantizer
from module import data_augmentation
from data import dataset
from copy import deepcopy
from tqdm import tqdm
from model.sasrec import SASRecQueryEncoder

class SASRec5(BaseModel):
    def __init__(self, config, dataset_list : list[dataset.BaseDataset]) -> None:
        super().__init__(config, dataset_list)
        self.query_encoder = SASRecQueryEncoder(
            self.fiid,
            self.embed_dim,
            self.max_seq_len,
            config['model']['head_num'],
            config['model']['hidden_size'],
            config['model']['dropout_rate'],
            config['model']['activation'],
            config['model']['layer_norm_eps'],
            config['model']['layer_num'],
            self.item_embedding,
        )
        self.query_encoder_twin = deepcopy(self.query_encoder)
        for param in self.query_encoder_twin.parameters():
            param.requires_grad = False
        self.condition_proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 10 * self.embed_dim)
        )
        # self.condition_proj = nn.Linear(self.embed_dim, 10 * self.embed_dim)

    def current_epoch_trainloaders(self, nepoch):
        return super().current_epoch_trainloaders(nepoch)

    def forward(self, batch, need_pooling=True):
        query = self.query_encoder(batch, need_pooling=True)
        B, D = query.shape
        query_c = self.condition_proj(query).reshape(B, 10, D)
        self.query_encoder_twin.load_state_dict(self.query_encoder.state_dict())
        batch = {
            'seq_emb': query_c,
            'seqlen': 10 * torch.ones(B, dtype=torch.int64, device=self.device),
        }
        query_q = self.query_encoder_twin(batch, need_pooling=True)
        self.query = query
        return query_q

    @staticmethod
    def alignment(x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    @staticmethod
    def uniformity(x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def training_step(self, batch, reduce=True, return_query=False):
        # loss_value, query = super().training_step(batch, reduce=True, return_query=True)
        query_q = self.forward(batch)
        alignment = 0
        # alignment += self.alignment(self.query, query_q)
        # alignment += 0.5 * self.alignment(query, self.item_embedding.weight[batch[self.fiid]])
        alignment += self.alignment(query_q, self.item_embedding.weight[batch[self.fiid]])
        uniformity = self.config['model']['uniformity'] * (
            self.uniformity(self.query) +
            self.uniformity(query_q) +
            self.uniformity(self.item_embedding.weight[batch[self.fiid]])
        )
        loss_value = alignment + uniformity
        return loss_value

    # def training_step(self, batch, reduce=True, return_query=False):
    #     loss_value = super().training_step(batch, reduce=True, return_query=False)
    #     loss_value = loss_value + self.embedding_loss
    #     return loss_value

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
                batch['nepoch'] = nepoch
                self.optimizer.zero_grad()
                training_step_args = {'batch': batch}
                loss = self.training_step(**training_step_args)
                loss.backward()
                self.optimizer.step()
                outputs.append({f"loss_{loader_idx}": loss.detach()})
            output_list.append(outputs)
        return output_list

    @torch.no_grad()
    def validation_epoch(self, nepoch, dataloader):
        output_list = []
        dataloader = tqdm(
            dataloader,
            total=len(dataloader),
            ncols=75,
            desc=f"Evaluating {nepoch:>5}",
            leave=False,
        )
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            batch['nepoch'] = nepoch
            # model validation results
            output = self.validation_step(batch)
            output_list.append(output)
        return output_list

    @torch.no_grad()
    def test_epoch(self, dataloader):
        output_list = []
        dataloader = tqdm(
            dataloader,
            total=len(dataloader),
            ncols=75,
            desc=f"Testing: ",
            leave=False,
        )
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            batch['nepoch'] = 1000
            # model validation results
            output = self.test_step(batch)
            output_list.append(output)
        return output_list