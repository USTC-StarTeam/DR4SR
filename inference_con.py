import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.nn.utils.rnn import pad_sequence
from utils.reparam_module import ReparamModule
import numpy as np
import random
from tqdm import tqdm
from argparse import ArgumentParser
from utils import normal_initialization
from module.layers import SeqPoolingLayer

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
        self.condition_emb = nn.Embedding(K, 64)
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
        rst = condition @ self.condition_emb.weight # BD
        return rst.unsqueeze(1)

class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(num_item + 2, 64, padding_idx=0)
        self.item_embedding_decoder = nn.Embedding(num_item + 2, 64, padding_idx=0)
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
        self.dropout = nn.Dropout(0.5)
        self.position_embedding = torch.nn.Embedding(50, 64)
        self.condition_encoder = ConditionEncoder(5)
        self.device = 'cuda'
        self.apply(normal_initialization)
        # self.load_pretrained()

    def load_pretrained(self):
        # path = 'saved/SASRec8/amazon-toys-seq-noise-50/2024-01-24-16-37-41-603118.ckpt'
        path = 'saved/SASRec7/amazon-toys-seq-noise-50/2024-01-24-17-16-57-368371.ckpt'
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

        con_emb = self.condition_encoder(src_emb, src_mask, src_padding_mask, src_seqlen)

        position_ids = torch.arange(tgt.size(1), dtype=torch.long, device=self.device)
        position_ids = position_ids.reshape(1, -1)
        tgt_position_embedding = self.position_embedding(position_ids)
        tgt_emb = self.dropout(
            torch.cat([con_emb, self.item_embedding(tgt[:, 1:])], dim=1) + \
            tgt_position_embedding
        ) # replace [SOS] with condition embedding

        outs = self.transformer(
            src_emb, tgt_emb, src_mask, tgt_mask, None,
            src_padding_mask, tgt_padding_mask, memory_key_padding_mask
        )
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
        # position_ids = torch.arange(tgt.size(1), dtype=torch.long, device=self.device)
        # position_ids = position_ids.reshape(1, -1)
        # tgt_position_embedding = self.position_embedding(position_ids)
        # tgt_emb = self.dropout(self.item_embedding(tgt) + tgt_position_embedding)
        
        # return self.transformer.decoder(tgt_emb, memory, tgt_mask)
        con_emb = self.condition_encoder.condition_emb.weight[self.condition].unsqueeze(0).unsqueeze(0)
        position_ids = torch.arange(tgt.size(1), dtype=torch.long, device=self.device)
        position_ids = position_ids.reshape(1, -1)
        tgt_position_embedding = self.position_embedding(position_ids)
        if tgt.shape[1] == 1:
            # only SOS in
            tgt_emb = self.dropout(con_emb + tgt_position_embedding)
        else:
            # replace SOS with Condition embedding
            tgt_emb = self.dropout(
                torch.cat([con_emb, self.item_embedding(tgt[:, 1:])], dim=1) + \
                tgt_position_embedding
            )
        
        return self.transformer.decoder(tgt_emb, memory, tgt_mask)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device='cuda')) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, -100000).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = generate_square_subsequent_mask(src_seq_len)

    src_padding_mask = (src == 0)
    tgt_padding_mask = (tgt == 0)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def inference_mask(logits, src, ys):
    mask = torch.zeros_like(logits, device=logits.device, dtype=torch.bool)
    mask = mask.scatter(-1, src, 1)
    mask = mask.scatter(-1, ys, 0)
    logits = torch.masked_fill(logits, ~mask, -torch.inf)
    return logits

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to('cuda')
    src_mask = src_mask.to('cuda')

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to('cuda')
    for i in range(max_len-1):
        memory = memory.to('cuda')
        tgt_mask = (generate_square_subsequent_mask(ys.size(1))
                    .type(torch.bool)).to('cuda')
        out = model.decode(ys, memory, tgt_mask)
        prob = out[:, -1] @ model.item_embedding_decoder.weight.T
        prob = inference_mask(prob, src, ys)
        _, next_word = torch.max(prob, dim=-1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == EOS:
            break
    return ys

def translate(model: torch.nn.Module, src):
    model.eval()
    src = src.reshape(1, -1)
    num_tokens = src.shape[1]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=25, start_symbol=SOS).flatten()
    return tgt_tokens

parser = ArgumentParser()
parser.add_argument('--begin', '-b', type=int, default=0, help='model name')
parser.add_argument('--end', '-e', type=int, default=1, help='dataset name')
parser.add_argument('--condition', '-c', type=int, default=0, help='dataset name')
args = parser.parse_args()
begin = args.begin * 5000
end = args.end * 5000
condition = args.condition

dataset_name = 'beauty'
full_dataset_name = 'amazon-beauty'
num_item_dict = {
    'toy': 11925,
    'sport': 18358,
    'beauty': 12102,
    'yelp-small': 20034,
}
path = f'./{dataset_name}-pair5.pth'
data = torch.load(path)
num_item = num_item_dict[dataset_name]
SOS = num_item
EOS = num_item + 1

model = Generator().to('cuda')
model.load_state_dict(torch.load(f'./translator-{dataset_name}-con.pth'))

def preprocess(seq):
    return torch.tensor([SOS] + seq + [EOS], device='cuda')
original_data = torch.load(f'./dataset/{full_dataset_name}-noise-50/{dataset_name}/train_ori.pth')
seqlist = [_[1][:_[3]] + [_[2][_[3] - 1]] for _ in original_data]
seqlist = [preprocess(_) for _ in seqlist]

filtered_sequences = []
for i in range(5):
    model.set_condition(condition)
    for seq in tqdm(seqlist[begin:end]):
        rst = translate(model, seq)
        filtered_sequences.append(seq)

torch.save(filtered_sequences, f'./f-seq-con-{dataset_name}-{begin}-{end}.pth')
