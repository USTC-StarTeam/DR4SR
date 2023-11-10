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

        self._load_datasets()
        self._get_user_hist()
        self.domain_user_mapping = self.get_domain_user_mapping()
        self.domain_item_mapping = self.get_domain_item_mapping()

    def __len__(self):
        if self.phase == 'train':
            return len(self.data[0]) # self.data[0] is the user_id list
        else:
            return len(self.data[self.eval_domain][0])
    
    def __getitem__(self, idx):
        raise NotImplementedError

    @property
    def domain_name_list(self):
        if self.name == 'amazon':
            return [
                'book',
                'cloth',
                'movie',
                'sport',
                'toy'
            ]
        else:
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

    def _get_user_hist(self):
        train_inter_data = self._inter_data.sort_values(by=['user_id', 'timestamp'])
        if self.phase == 'train' or self.phase == 'val':
            train_inter_data = train_inter_data.groupby(by=['user_id'])['item_id'].apply(list)
            train_inter_data = train_inter_data.apply(lambda x: x[-self.config['topk']:]).apply(lambda x: x[:-2])
        else:
            train_inter_data = train_inter_data.groupby(by=['user_id'])['item_id'].apply(list)
            train_inter_data = train_inter_data.apply(lambda x: x[-self.config['topk']:]).apply(lambda x: x[:-1])
        user_id, _user_hist = train_inter_data.index, train_inter_data.tolist()

        user_hist = [torch.tensor([]) for _ in range(self.num_users)]
        for u_id, u_hist in zip(user_id, _user_hist):
            user_hist[u_id] = torch.tensor(u_hist)
        self.user_hist = pad_sequence(user_hist, batch_first=True, padding_value=self.num_items)

    def unpack(self, to_be_unpacked):
        user_id = torch.tensor([_[0] for _ in to_be_unpacked])
        user_seq = torch.tensor([_[1] for _ in to_be_unpacked])
        target_item = torch.tensor([_[2] for _ in to_be_unpacked])
        seq_len = torch.tensor([_[3] for _ in to_be_unpacked])
        return user_id, user_seq, target_item, seq_len
    
    def _neg_sampling(self, user_id):
        neg_idx = random.randint(0, self.num_items - 1)
        return neg_idx
    
    def _build(self):
        self.data = None
        raise NotImplementedError
    
    def build(self):
        self._build()

    def get_loader(self):
        if self.phase == 'train':
            return DataLoader(self, self.config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        else:
            return DataLoader(self, self.config['batch_size'] * 10, shuffle=True, num_workers=4, pin_memory=True)

    def set_eval_domain(self, domain):
        self.eval_domain = domain

class NormalDataset(BaseDataset):
    """Normal datasets mix all source datasets for training and evaluate them separately.
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
            user_id = self.data[0][idx]
            user_seq = self.data[1][idx]
            target_item = self.data[2][idx]
            seq_len = self.data[3][idx]
            neg_item = self._neg_sampling(user_id)
            return {
                'user_id': user_id,
                'user_seq': user_seq,
                'target_item': target_item,
                'seq_len': seq_len,
                'neg_item': neg_item,
            }
        else:
            user_id = self.data[self.eval_domain][0][idx]
            user_seq = self.data[self.eval_domain][1][idx]
            target_item = self.data[self.eval_domain][2][idx]
            seq_len = self.data[self.eval_domain][3][idx]
            return {
                'user_id': user_id,
                'user_seq': user_seq,
                'target_item': target_item,
                'seq_len': seq_len,
                'user_hist': self.user_hist[user_id]
            }