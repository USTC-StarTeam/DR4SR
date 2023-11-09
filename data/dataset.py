import os
import torch
import logging
import pandas as pd

from torch.utils.data import Dataset, DataLoader

class BaseDataset(Dataset):
    def __init__(self, args, phase='train') -> None:
        super().__init__()
        self.name = args['dataset']
        self.logger = logging.getLogger('CDR')
        self.args = args
        self.phase = phase
        self.device = args['device']

        self._load_datasets()

    def __len__(self):
        return len(self.data[0]) # self.data[0] is the user_id list
    
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

    def unpack(self, to_be_unpacked):
        user_id = torch.tensor([_[0] for _ in to_be_unpacked], device=self.device)
        user_seq = torch.tensor([_[1] for _ in to_be_unpacked], device=self.device)
        target_item = torch.tensor([_[2] for _ in to_be_unpacked], device=self.device)
        seq_len = torch.tensor([_[3] for _ in to_be_unpacked], device=self.device)
        return user_id, user_seq, target_item, seq_len
    
    def _neg_sampling(self, user_hist):
        weight = torch.ones(size=(len(user_hist), self.num_items), device=self.device)
        _idx = torch.arange(user_hist.size(0), device=self.device).view(-1, 1).expand_as(user_hist)
        weight[_idx, user_hist] = 0.0
        neg_idx = torch.multinomial(weight, self.args['num_neg'], replacement=True)
        return neg_idx

    def _build(self):
        raise NotImplementedError
    
    def build(self):
        self._build()

    def get_loader(self):
        return DataLoader(self, self.args['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

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
            self.user_hist = self.data[1]
        else:
            self.data = {
                self.domain_name_list[idx]: self.unpack(_) for idx, _ in enumerate(self._data)
            }

    def set_eval_domain(self, domain):
        self.eval_domain = domain

    def __getitem__(self, idx):
        if self.phase == 'train':
            user_id = self.data[0][idx]
            user_seq = self.data[1][idx]
            target_item = self.data[2][idx]
            seq_len = self.data[3][idx]
            neg_item = self._neg_sampling(user_seq)
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
            seq_len = self.data[3][idx]
            return {
                'user_id': user_id,
                'user_seq': user_seq,
                'target_item': target_item,
                'seq_len': seq_len,
            }