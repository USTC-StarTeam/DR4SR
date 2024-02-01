#!/usr/bin/env python
# coding: utf-8

# In[24]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.nn.utils.rnn import pad_sequence
from utils.reparam_module import ReparamModule
import numpy as np
import random
from tqdm import tqdm


# # Enviroment

# In[25]:


seed = 2024

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
import os


# # Dataset

# In[26]:


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

dataset_name = 'yelp-small'
full_dataset_name = 'yelp-small'
num_item_dict = {
    'toy': 11925,
    'sport': 18358,
    'beauty':12102,
    'yelp-small': 20034,
}
path = f'./{dataset_name}-pair2.pth'
data = torch.load(path)
num_item = num_item_dict[dataset_name]
SOS = num_item
EOS = num_item + 1


# In[27]:


source, target = [], []
source_seqlen, target_seqlen = [], []
for _ in data:
    s, t = _
    source_seqlen.append(len(s) + 2)
    target_seqlen.append(len(t) + 2)
    s = torch.tensor([SOS] + s + [EOS])
    t = torch.tensor([SOS] + t + [EOS])
    source.append(s)
    target.append(t)
source = pad_sequence(source, batch_first=True, padding_value=0)
target = pad_sequence(target, batch_first=True, padding_value=0)
if target.shape[1] < 20:
    target = torch.cat([target, torch.zeros(target.shape[0], 20 - target.shape[1], dtype=torch.int)], dim=-1)
source_seqlen = torch.tensor(source_seqlen)
target_seqlen = torch.tensor(target_seqlen)


# In[28]:


from torch.utils.data import Dataset, DataLoader
class MyDataset(Dataset):
    def __init__(
            self,
            source,
            target,
            source_seqlen,
            target_seqlen,
        ) -> None:
        super().__init__()
        self.source = source.to('cuda')
        self.target = target.to('cuda')
        self.source_seq_len = source_seqlen.to('cuda')
        self.target_seq_len = target_seqlen.to('cuda')

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        src = self.source[index]
        tgt = self.target[index]
        src_len = self.source_seq_len[index]
        tgt_len = self.target_seq_len[index]
        return src, tgt, src_len, tgt_len


# # Model

# In[29]:


from utils import normal_initialization
from module.layers import SeqPoolingLayer
K = 5

class ConditionEncoder(nn.Module):
    def __init__(self, K) -> None:
        super().__init__()
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=2,
            dim_feedforward=256,
            dropout=0.5,
            activation='gelu',
            layer_norm_eps=1e-12,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=transformer_layer,
            num_layers=2,
        )
        self.condition_layer = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, K),
        )
        self.pooling_layer = SeqPoolingLayer('mean')
        self.tau = 1

    def forward(self, trm_input, src_mask, memory_key_padding_mask, src_seqlen):
        trm_out = self.encoder(
            src=trm_input,
            mask=src_mask,  # BxLxD
            src_key_padding_mask=memory_key_padding_mask,
        )
        trm_out = self.pooling_layer(trm_out, src_seqlen) # BD
        condition = self.condition_layer(trm_out) # BK
        condition = F.gumbel_softmax(condition, tau=self.tau, dim=-1) # BK
        self.condition4loss = condition
        self.tau = max(self.tau * 0.995, 0.1)
        return condition

class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.item_embedding = nn.Embedding(num_item + 2, 64, padding_idx=0)
        # self.item_embedding_decoder = nn.Embedding(num_item + 2, 64, padding_idx=0)
        self.transformer = nn.Transformer(
            d_model=64,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=256,
            dropout=0.5,
            activation='gelu',
            layer_norm_eps=1e-12,
            batch_first=True,
        )
        self.condition_linear = nn.Sequential(
            nn.Linear(64, 64 * K),
            nn.ReLU(),
            nn.Linear(64 * K, 64 * K)
        )
        self.dropout = nn.Dropout(0.5)
        self.position_embedding = torch.nn.Embedding(50, 64)
        self.condition_encoder = ConditionEncoder(K)
        self.device = 'cuda'
        self.apply(normal_initialization)
        self.load_pretrained()

    def load_pretrained(self):
        path_dict = {
            'toy': 'saved/SASRec/amazon-toys-noise-50/2024-01-22-19-24-37-334415.ckpt',
            'beauty': 'saved/SASRec/amazon-beauty-noise-50/2024-01-25-10-39-46-322830.ckpt',
            'sport': 'saved/SASRec/amazon-sport-noise-50/2024-01-25-09-36-23-316839.ckpt',
            'yelp-small': 'saved/SASRec/yelp-small-noise-50/2024-01-25-20-34-58-431582.ckpt',
        }
        path = path_dict[dataset_name]
        saved = torch.load(path, map_location='cpu')
        pretrained = saved['parameters']['item_embedding.weight']
        pretrained = torch.cat([
            pretrained,
            nn.init.normal_(torch.zeros(2, 64), std=0.02)
        ])
        self.item_embedding = nn.Embedding.from_pretrained(pretrained, padding_idx=0, freeze=False)
        self.item_embedding_decoder = self.item_embedding

    def condition_mask(self, logits, src):
        mask = torch.zeros_like(logits, device=logits.device, dtype=torch.bool)
        mask = mask.scatter(-1, src.unsqueeze(-2).repeat(1, mask.shape[1], 1), 1)
        logits = torch.masked_fill(logits, ~mask, -torch.inf)
        return logits

    def forward(self, src, tgt, src_mask, tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
            src_seqlen,
            tgt_seqlen,
        ):
        position_ids = torch.arange(src.size(1), dtype=torch.long, device=self.device)
        position_ids = position_ids.reshape(1, -1)
        src_position_embedding = self.position_embedding(position_ids)
        src_emb = self.dropout(self.item_embedding(src) + src_position_embedding)

        memory = self.transformer.encoder(src_emb, src_mask, src_padding_mask)
        B, L, D = memory.shape
        memory = self.condition_linear(memory).reshape(B, L, K, D)

        position_ids = torch.arange(tgt.size(1), dtype=torch.long, device=self.device)
        position_ids = position_ids.reshape(1, -1)
        tgt_position_embedding = self.position_embedding(position_ids)
        tgt_emb = self.dropout(self.item_embedding(tgt) + tgt_position_embedding)

        condition = self.condition_encoder(tgt_emb, tgt_mask, tgt_padding_mask, tgt_seqlen) # BK
        condition = condition.reshape(B, 1, K, 1)
        memory_cond = (memory * condition).sum(-2)

        outs = self.transformer.decoder(tgt_emb, memory_cond, tgt_mask, None, tgt_padding_mask, memory_key_padding_mask)

        logits = outs @ self.item_embedding_decoder.weight.T
        logits = self.condition_mask(logits, src)

        return logits
    
    def encode(self, src, src_mask):
        position_ids = torch.arange(src.size(1), dtype=torch.long, device=self.device)
        position_ids = position_ids.reshape(1, -1)
        src_position_embedding = self.position_embedding(position_ids)
        src_emb = self.dropout(self.item_embedding(src) + src_position_embedding)

        return self.transformer.encoder(src_emb, src_mask)

    def set_condition(self, condition):
        self.condition = condition

    def decode(self, tgt, memory, tgt_mask):
        B, L, D = memory.shape
        memory = self.condition_linear(memory).reshape(B, L, K, D)[:, :, self.condition]
        position_ids = torch.arange(tgt.size(1), dtype=torch.long, device=self.device)
        position_ids = position_ids.reshape(1, -1)
        tgt_position_embedding = self.position_embedding(position_ids)
        tgt_emb = self.dropout(self.item_embedding(tgt) + tgt_position_embedding)
        
        return self.transformer.decoder(tgt_emb, memory, tgt_mask)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device='cuda')) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, -100000).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    # src_mask = torch.zeros((src_seq_len, src_seq_len),device='cuda').type(torch.bool)
    src_mask = generate_square_subsequent_mask(src_seq_len)

    src_padding_mask = (src == 0)
    tgt_padding_mask = (tgt == 0)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


# # Train

# In[30]:


from torch.optim.lr_scheduler import CosineAnnealingLR
NUM_EPOCHS = 40
model = Generator().to('cuda')
loss_fn = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
dataset = MyDataset(source, target, source_seqlen, target_seqlen)
lr_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS,
)


# In[31]:


def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    for src, tgt, src_seqlen, tgt_seqlen in tqdm(train_dataloader):
        tgt_input = tgt[:, :-1]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        logits = model(
            src, tgt_input, src_mask, tgt_mask,
            src_padding_mask, tgt_padding_mask, src_padding_mask,
            src_seqlen, tgt_seqlen)
        optimizer.zero_grad()
        tgt_out = tgt[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        condition_prob = model.condition_encoder.condition4loss
        reg_loss = - (condition_prob * torch.log(condition_prob + 1e-12)).sum(-1).mean()
        losses += loss.item()
        loss = loss + 1 * reg_loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
    return losses / len(list(train_dataloader))


# In[32]:


from timeit import default_timer as timer


for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(model, optimizer)
    end_time = timer()
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))


# In[33]:


torch.save(model.state_dict(), f'translator-{dataset_name}-con2-real.pth')
