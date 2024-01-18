import torch
import torch.nn as nn
import torch.nn.functional as F
from model.basemodel import BaseModel
from module.layers import SeqPoolingLayer, VectorQuantizer
from module import data_augmentation
from data import dataset
from copy import deepcopy
from tqdm import tqdm

class SASRecQueryEncoder(torch.nn.Module):
    def __init__(
            self, fiid, embed_dim, max_seq_len, n_head, hidden_size, dropout, activation, layer_norm_eps, n_layer, item_encoder,
            bidirectional=False, training_pooling_type='last', eval_pooling_type='last') -> None:
        super().__init__()
        self.fiid = fiid
        self.item_encoder = item_encoder
        self.bidirectional = bidirectional
        self.training_pooling_type = training_pooling_type
        self.eval_pooling_type = eval_pooling_type
        self.position_emb = torch.nn.Embedding(max_seq_len, embed_dim)
        transformer_encoder = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=hidden_size,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=False
        )
        self.transformer_layer = torch.nn.TransformerEncoder(
            encoder_layer=transformer_encoder,
            num_layers=n_layer,
        )
        self.dropout = torch.nn.Dropout(p=dropout)
        self.training_pooling_layer = SeqPoolingLayer(pooling_type=self.training_pooling_type)
        self.eval_pooling_layer = SeqPoolingLayer(pooling_type=self.eval_pooling_type)

    def forward(self, batch, need_pooling=True):
        user_hist = batch['in_'+self.fiid]
        positions = torch.arange(user_hist.size(1), dtype=torch.long, device=user_hist.device)
        positions = positions.unsqueeze(0).expand_as(user_hist)
        position_embs = self.position_emb(positions)
        seq_embs = self.item_encoder(user_hist)

        L = user_hist.size(-1)
        if batch.get('attention_mask', None) is not None:
            mask4padding = batch['attention_mask']
            if not self.bidirectional:
                attention_mask = torch.tril(torch.ones((L, L), device=user_hist.device), 0).float()
            else:
                attention_mask = torch.zeros((L, L), device=user_hist.device)
        else:    
            mask4padding = user_hist == 0  # BxL
            if not self.bidirectional:
                attention_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=user_hist.device), 1)
            else:
                attention_mask = torch.zeros((L, L), dtype=torch.bool, device=user_hist.device)
        try:
            transformer_input = batch['input_weight'] * (seq_embs + position_embs)
        except:
            transformer_input = seq_embs + position_embs
        transformer_out = self.transformer_layer(
            src=self.dropout(transformer_input),
            mask=attention_mask,
            src_key_padding_mask=mask4padding)  # BxLxD
        if not need_pooling:
            return transformer_out
        else:
            if self.training:
                return self.training_pooling_layer(transformer_out, batch['seqlen'])
            else:
                return self.eval_pooling_layer(transformer_out, batch['seqlen'])

class SASRec3(BaseModel):
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
        self.quantizer = VectorQuantizer(
            self.embed_dim,
            self.embed_dim // self.embed_dim,
            K=128,
            beta=0.25,
            depth=4
        )

    def current_epoch_trainloaders(self, nepoch):
        return super().current_epoch_trainloaders(nepoch)

    def forward(self, batch, need_pooling=True):
        if batch['nepoch'] < self.config['train']['warmup_epoch']:
            query_q = self.query_encoder(batch, need_pooling)
            embedding_loss, perplexity = 0, 0
        else:
            query = self.query_encoder(batch, need_pooling)
            embedding_loss, query_q, perplexity, _, _ = self.quantizer(query)
        self.embedding_loss = embedding_loss
        self.perplexity = perplexity
        return query_q

    def training_step(self, batch, reduce=True, return_query=False):
        loss_value = super().training_step(batch, reduce=True, return_query=False)
        loss_value = loss_value + self.embedding_loss
        return loss_value

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