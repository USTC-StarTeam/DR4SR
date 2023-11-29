import os
import torch
import random
import logging
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class BaseDataset(Dataset):
    def __init__(self, config, phase='train') -> None:
        super().__init__()
        self.name = config['dataset']
        self.logger = logging.getLogger('CDR')
        self.config = config
        self.phase = phase
        self.device = config['device']
        self.domain_name_list = self.config['domain_name_list']

        # register attribute
        self._data = None
        self.data = None
        self.user_hist = None

        self._load_datasets()
        self.domain_user_mapping = self.get_domain_user_mapping()
        self.domain_item_mapping = self.get_domain_item_mapping()

    def __len__(self):
        if self.phase == 'train':
            return len(self.data[0]) # self.data[0] is the user_id list
        else:
            return len(self.data[self.eval_domain][0])
    
    def __getitem__(self, idx):
        raise NotImplementedError

    def get_domain_user_mapping(self):
        rst = {}
        for idx, domain in enumerate(self.domain_name_list):
            sub_df = self._inter_data[self._inter_data['domain'] == idx]
            rst[domain] = sub_df['user_id'].unique().tolist()
        return rst
    
    def get_domain_item_mapping(self):
        rst = {}
        for idx, domain in enumerate(self.domain_name_list):
            sub_df = self._inter_data[self._inter_data['domain'] == idx]
            rst[domain] = sub_df['item_id'].unique().tolist()
        return rst

    def _load_datasets(self):
        path_prefix = 'dataset'
        inter_list = []
        for domain_name in self.domain_name_list:
            path = os.path.join(path_prefix, self.name, domain_name)
            inter_data = pd.read_csv(os.path.join(path, 'inter.csv'))
            inter_list.append(inter_data)
        self._inter_data = pd.concat(inter_list)
        self._num_users = len(self._inter_data['user_id'].unique())
        self._num_items = len(self._inter_data['item_id'].unique())

    @property
    def num_users(self):
        return self._num_users
    
    @property
    def num_items(self):
        return self._num_items
    
    @property
    def num_domains(self):
        return len(self.domain_name_list)

    def unpack(self, to_be_unpacked):
        user_id = torch.tensor([_[0] for _ in to_be_unpacked])
        user_seq = torch.tensor([_[1] for _ in to_be_unpacked])
        target_item = torch.tensor([_[2] for _ in to_be_unpacked])
        seq_len = torch.tensor([_[3] for _ in to_be_unpacked])
        label = torch.tensor([_[4] for _ in to_be_unpacked])
        domain_id = torch.tensor([_[5] for _ in to_be_unpacked])
        return user_id, user_seq, target_item, seq_len, label, domain_id
    
    def _neg_sampling(self, user_hist, seq_len):
        weight = torch.ones(self.num_items + 1)
        weight[user_hist] = 0.0
        weight[-1] = 0 # padding
        neg_idx = torch.multinomial(weight, self.config['num_neg'] * seq_len, replacement=True)
        neg_idx = neg_idx.reshape(seq_len, self.config['num_neg'])
        return neg_idx
    
    def _build(self):
        self.data = None
        raise NotImplementedError
    
    def build(self):
        self._build()

    def get_loader(self):
        if self.phase == 'train':
            return DataLoader(self, self.config['batch_size'], shuffle=True, pin_memory=True)
        else:
            return DataLoader(self, self.config['eval_batch_size'], pin_memory=True)

    def set_eval_domain(self, domain):
        self.eval_domain = domain

class SeparateDataset(BaseDataset):
    """Simply put sequences in each domain togather.
    """

    def _load_datasets(self):
        super()._load_datasets()
        path_prefix = 'dataset'
        data_list = []
        for domain_name in self.domain_name_list:
            path = os.path.join(path_prefix, self.name, domain_name)
            data = torch.load(os.path.join(path, self.phase + '.pth'))
            data_list.append(data)
        self._data = data_list

    def _build(self,):
        if self.phase == 'train':
            self.data = []
            for _ in self._data:
                self.data += _
            self.data = self.unpack(self.data)
        else:
            self.data = {
                self.domain_name_list[idx]: self.unpack(_) for idx, _ in enumerate(self._data)
            }

    def __getitem__(self, idx):
        if self.phase == 'train':
            data = self.data
        else:
            data = self.data[self.eval_domain]
        batch = {}
        batch['user_id'] = data[0][idx]
        batch['user_seq'] = data[1][idx]
        batch['target_item'] = data[2][idx]
        batch['seq_len'] = data[3][idx]
        batch['label'] = data[4][idx]
        batch['domain_id'] = data[5][idx]
        if self.phase == 'train':
            batch['neg_item'] = self._neg_sampling(batch['user_seq'], self.config['max_seq_len'])
        else:
            batch['user_hist'] = batch['user_seq']
        return batch

class MixDataset(BaseDataset):
    """Merge train/eval sequences into a mixed sequence.
    """

    def _load_datasets(self):
        super()._load_datasets()
        path_prefix = 'dataset'
        if self.phase == 'train':
            path = os.path.join(path_prefix, self.name)
            data = torch.load(os.path.join(path, self.phase + '.pth'))
            self._data = data
        else:
            data_list = []
            for domain_name in self.domain_name_list:
                path = os.path.join(path_prefix, self.name, domain_name)
                data = torch.load(os.path.join(path, self.phase + '.pth'))
                data_list.append(data)
            self._data = data_list

    def _build(self,):
        if self.phase == 'train':
            self.data = self.unpack(self._data)
        else:
            self.data = {
                self.domain_name_list[idx]: self.unpack(_) for idx, _ in enumerate(self._data)
            }

    def __getitem__(self, idx):
        if self.phase == 'train':
            data = self.data
        else:
            data = self.data[self.eval_domain]
        batch = {}
        batch['user_id'] = data[0][idx]
        batch['user_seq'] = data[1][idx]
        batch['target_item'] = data[2][idx]
        batch['seq_len'] = data[3][idx]
        batch['label'] = data[4][idx]
        batch['domain_id'] = data[5][idx]
        if self.phase == 'train':
            batch['neg_item'] = self._neg_sampling(batch['user_seq'], self.num_domains * self.config['max_seq_len'])
        else:
            batch['user_hist'] = batch['user_seq']
        return batch