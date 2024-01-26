import torch
import torch.nn as nn
import torch.nn.functional as F
from model.basemodel import BaseModel
from module.layers import SeqPoolingLayer, VanillaVectorQuantizer, MLPModule, TransformerEncoder
from data import dataset
from copy import deepcopy
from tqdm import tqdm
from model.sasrec import SASRecQueryEncoder
from torch.distributions import Bernoulli

class SASRec7(BaseModel):
    def __init__(self, config, dataset_list : list[dataset.BaseDataset]) -> None:
        super().__init__(config, dataset_list)
        self.H = self.config['model']['H']
        self.alpha = 0.5
        # self.quantizer = VectorQuantizer(
        #     self.embed_dim,
        #     2,
        #     K=self.config['model']['K'],
        #     beta=0.25,
        #     depth=self.config['model']['depth'],
        # )
        self.quantizer = VanillaVectorQuantizer(
            n_e=self.config['model']['K'],
            e_dim=self.embed_dim,
            beta=0.25,
            depth=self.config['model']['depth'],
        )
        self.selector = MLPModule([
            self.embed_dim,
            self.embed_dim,
            self.embed_dim,
            self.H,
        ], dropout=0.5, last_activation=False)
        # Query encoder
        self.position_embedding = torch.nn.Embedding(self.max_seq_len, self.embed_dim)
        self.trm_encoder = TransformerEncoder(
            n_layers=config['model']['layer_num'],
            n_heads=config['model']['head_num'],
            hidden_size=self.embed_dim,
            inner_size=config['model']['hidden_size'],
            hidden_dropout_prob=config['model']['dropout_rate'],
            attn_dropout_prob=config['model']['dropout_rate'],
            hidden_act=config['model']['activation'],
            layer_norm_eps=config['model']['layer_norm_eps'],
        )
        self.LayerNorm = nn.LayerNorm(self.embed_dim, eps=config['model']['layer_norm_eps'])
        self.dropout = nn.Dropout(config['model']['dropout_rate'])
        self.training_pooling_layer = SeqPoolingLayer(pooling_type='last')
        self.eval_pooling_layer = SeqPoolingLayer(pooling_type='last')


    def current_epoch_trainloaders(self, nepoch):
        return super().current_epoch_trainloaders(nepoch)

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

        # logits = self.selector(seq_embs).reshape(B, L, self.H, 2) # BLH2
        # logits = F.softmax(logits, dim=-1)[..., 0]# BLH

        logits = self.selector(seq_embs)

        logits = logits + self.alpha
        self.alpha = self.alpha * 0.999
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

    def forward(self, batch, need_pooling=True):
        if not self.config['model']['ab-wo-pattern']:
            pattern, att_mask, last_seq_len = self.generate_pattern(batch)
        else:
            item_seq = batch['in_' + self.fiid]
            pattern = self.item_embedding(item_seq)
            att_mask = self.get_attention_mask(item_seq)
            last_seq_len = batch['seqlen']
        position_ids = torch.arange(
            pattern.size(1), dtype=torch.long, device=self.device
        )
        position_ids = position_ids.reshape(1, -1)
        position_embedding = self.position_embedding(position_ids)

        input_emb = pattern + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        trm_output = self.trm_encoder(
            input_emb, att_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.training_pooling_layer(output, last_seq_len)
        if self.config['model']['quantization']:
            self.query_q = self.merge_pattern(output)
            self.embedding_loss, output = self.quantizer(output)
        else:
            self.query_q = self.merge_pattern(output)
            self.embedding_loss = 0
        if not self.config['model']['ab-wo-pattern']:
            output = self.merge_pattern(output)
        return output

    @staticmethod
    def alignment(x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    @staticmethod
    def uniformity(x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def training_step(self, batch, reduce=True, return_query=False):
        query = self.forward(batch)
        # pos_score = (query * self.item_embedding.weight[batch[self.fiid]]).sum(-1)
        # neg_score = (query.unsqueeze(-2) * self.item_embedding.weight[batch['neg_item']]).sum(-1)
        # pos_score[batch[self.fiid] == 0] = -torch.inf # padding
        # loss_value = self.loss_fn(pos_score, neg_score, reduce=reduce)

        embedding_loss = self.embedding_loss
        alignment = 0
        alignment += 0.5 * self.alignment(self.query_q, self.item_embedding.weight[batch[self.fiid]])
        alignment += 0.5 * self.alignment(query, self.item_embedding.weight[batch[self.fiid]])
        uniformity = self.config['model']['uniformity'] * (
            0.5 * self.uniformity(self.query_q) +
            0.5 * self.uniformity(query) +
            self.uniformity(self.item_embedding.weight[batch[self.fiid]])
        )
        # subspace_reg = F.relu(self.log_value - 0.7).pow(2).mean(1).mean()
        subspace_reg = self.selection.sum(1).mean()
        loss_value = alignment + uniformity + embedding_loss + 0.007 * subspace_reg
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
        print(self.log_value.mean(1).mean())
        return output_list