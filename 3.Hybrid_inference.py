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
        path = os.path.join(args.root_path, 'pre-trained_embedding.ckpt')
        # path = path_dict[dataset_name]
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


def inference_mask(logits, src, ys):
    mask = torch.zeros_like(logits, device=logits.device, dtype=torch.bool)
    mask = mask.scatter(-1, src, 1)
    mask = mask.scatter(-1, ys, 0)
    logits = torch.masked_fill(logits, ~mask, -torch.inf)
    return logits

def inference_mask_generative(logits, src, ys):
    mask = torch.ones_like(logits, device=logits.device, dtype=torch.bool)
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
        if random.random() > 1 or i <= 1:
            prob = inference_mask(prob, src, ys)
        else:
            prob = inference_mask_generative(prob, src, ys)
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


if __name__ == '__main__':


    parser = ArgumentParser()
    parser.add_argument('--root_path', type=str, default='./dataset/amazon-toys/toy/', help='The path to the dataset.')
    parser.add_argument('--ckpt_name', type=str, default="regenerator.pth", help='The name of pretrained regenerator')
    parser.add_argument('--begin', '-b', type=int, default=0, help='Used for multi-processing. Beginning of the inference.')
    parser.add_argument('--end', '-e', type=int, default=1000000, help='Used for multi-processing. End of the inference.')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    begin = args.begin * 5000
    end = args.end * 5000

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    dataset_name = args.root_path.split('/')[-2] # e.g., 'toy' in './dataset/amazon-toys/toy/'
    num_item_dict = {
        'toy': 11925,
        'sport': 18358,
        'beauty': 12102,
        'yelp': 20034,
    }
    num_item = num_item_dict[dataset_name]
    SOS = num_item
    EOS = num_item + 1

    model = Generator().to('cuda')
    model.load_state_dict(torch.load(os.path.join(args.root_path, args.ckpt_name)))

    def preprocess(seq):
        return torch.tensor([SOS] + seq + [EOS], device='cuda')
    original_data = torch.load(os.path.join(args.root_path, 'train.pth'))
    seqlist = [_[1][:_[3]] + [_[2][_[3] - 1]] for _ in original_data]
    seqlist = [preprocess(_) for _ in seqlist]

    ori_pattern = torch.load(os.path.join(args.root_path, 'patterns.pth'))

    filtered_sequences = []
    for i in range(K):
        model.set_condition(i)
        for seq in tqdm(seqlist[begin:end]):
            rst = translate(model, seq)
            filtered_sequences.append(rst)
    
    train_set = set()

    for pattern in filtered_sequences:
        seq = pattern.tolist()[1:-1]
        train_set.add(tuple(seq))

    max_seq_len = 50
    def truncate_or_pad(seq):
        cur_seq_len = len(seq)
        if cur_seq_len > max_seq_len:
            return seq[-max_seq_len:]
        else:
            return seq + [0] * (max_seq_len - cur_seq_len)
    train_list = []
    for _ in train_set:
        seq_len = sum([a != 0 for a in list(_)[:-1]])
        if seq_len == 0:
            continue
        train_list.append([
            1,
            truncate_or_pad(list(_)[:-1]),
            truncate_or_pad(list(_)[1:]),
            seq_len,
            [1] * max_seq_len,
            [0] * max_seq_len,
        ])

    out_path = os.path.join(args.root_path, 'train_regen.pth')
    torch.save(original_data + ori_pattern + train_list, out_path)
