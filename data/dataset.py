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
        self.name = config['data']['dataset']
        self.fuid = 'user_id'
        self.fiid = 'item_id'
        self.logger = logging.getLogger('CDR')
        self.config = config
        self.phase = phase
        self.device = config['train']['device']
        self.domain_name_list = self.config['data']['domain_name_list']
        self.max_seq_len = self.config['data']['max_seq_len']

        # register attribute
        self._data = None
        self.data = None
        self.user_hist = None

        self._load_datasets()
        self.domain_user_mapping = self.get_domain_user_mapping()
        self.domain_item_mapping = self.get_domain_item_mapping()
        self.eval_domain = self.domain_name_list[0]

    def __len__(self):
        if self.phase == 'train':
            return len(self.data_index) # self.data[0] is the user_id list
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
        self._num_users = len(self._inter_data['user_id'].unique()) + 1 # +1 for padding
        self._num_items = len(self._inter_data['item_id'].unique()) + 1 # +1 for padding

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
        user_id = torch.tensor([_[0] for _ in to_be_unpacked], device=self.device)
        user_seq = torch.tensor([_[1] for _ in to_be_unpacked], device=self.device)
        target_item = torch.tensor([_[2] for _ in to_be_unpacked], device=self.device)
        seq_len = torch.tensor([_[3] for _ in to_be_unpacked], device=self.device)
        label = torch.tensor([_[4] for _ in to_be_unpacked], device=self.device)
        domain_id = torch.tensor([_[5] for _ in to_be_unpacked], device=self.device)
        if self.phase != 'train':
            user_hist = user_seq
            # user_hist = pad_sequence([torch.tensor(_[6], device=self.device) for _ in to_be_unpacked], batch_first=True, padding_value=0)
            return user_id, user_seq, target_item, seq_len, label, domain_id, user_hist
        else:
            return user_id, user_seq, target_item, seq_len, label, domain_id
    
    def _build(self):
        self.data = None
        raise NotImplementedError
    
    def build(self):
        self._build()
        if self.phase == 'train':
            self.data_index = torch.arange(len(self._data[0]))
            self.data = self._data
        else:
            self.data = self._data

    def get_loader(self, batch_size=None, shuffle=True):
        if self.phase == 'train':
            batch_size = self.config['train']['batch_size'] if batch_size == None else batch_size
            return DataLoader(self, batch_size, shuffle=shuffle)
        else:
            batch_size = self.config['eval']['batch_size'] if batch_size == None else batch_size
            return DataLoader(self, batch_size)

    def set_eval_domain(self, domain):
        self.eval_domain = domain

    def set_data_index(self, data_index):
        assert self.phase == 'train'
        self.data_index = data_index
        self.data = [_[self.data_index] for _ in self._data]

class SeparateDataset(BaseDataset):
    """Simply put sequences in each domain togather.
    """

    def _load_datasets(self):
        super()._load_datasets()
        path_prefix = 'dataset'
        data_list = []
        for domain_name in self.domain_name_list:
            path = os.path.join(path_prefix, self.name, domain_name)
            if self.phase == 'train':
                data = torch.load(os.path.join(path, self.phase + self.config['data']['train_file'] + '.pth'))
            else:
                data = torch.load(os.path.join(path, self.phase + '.pth'))
            data_list.append(data)
        self._data = data_list

    def _build(self,):
        if self.phase == 'train':
            data = []
            for _ in self._data:
                data += _
            self._data = self.unpack(data)
        else:
            self._data = {
                self.domain_name_list[idx]: self.unpack(_) for idx, _ in enumerate(self._data)
            }

    def __getitem__(self, idx):
        if self.phase == 'train':
            data = self.data
        else:
            data = self.data[self.eval_domain]
        batch = {}
        batch[self.fuid] = data[0][idx]
        batch['in_' + self.fiid] = data[1][idx]
        batch[self.fiid] = data[2][idx]
        batch['seqlen'] = data[3][idx]
        batch['label'] = data[4][idx]
        batch['domain_id'] = data[5][idx]
        batch['index'] = idx
        if self.phase != 'train':
            batch['user_hist'] = data[6][idx]
        return batch

class PatternDataset(SeparateDataset):
    def _load_datasets(self):
        super()._load_datasets()
        path_prefix = 'dataset'
        data_list = []
        if self.phase == 'train':
            pattern_data_list = []
        for domain_name in self.domain_name_list:
            path = os.path.join(path_prefix, self.name, domain_name)
            if self.phase == 'train':
                data = torch.load(os.path.join(path, self.phase + '.pth'))
                pattern_data = torch.load(os.path.join(path, self.phase + self.config['data']['train_file'] + '.pth'))
                pattern_data_list.append(pattern_data)
            else:
                data = torch.load(os.path.join(path, self.phase + '.pth'))
            data_list.append(data)
        self._data = data_list
        if self.phase == 'train':
            self._pattern_data = pattern_data_list

    def _build(self,):
        if self.phase == 'train':
            data = []
            for _ in self._data:
                data += _
            self._data = self.unpack(data)
            pattern_data = []
            for _ in self._pattern_data:
                pattern_data += _
            self._pattern_data = self.unpack(pattern_data)
        else:
            self._data = {
                self.domain_name_list[idx]: self.unpack(_) for idx, _ in enumerate(self._data)
            }

    def build(self):
        self._build()
        if self.phase == 'train':
            self.data_index = torch.arange(len(self._data[0]))
            self.original_index = torch.arange(len(self._data[0]))
            self.data = self._data
            self.pattern_index = torch.arange(len(self._pattern_data[0]))
            self.pattern_data = self._pattern_data
        else:
            self.data = self._data

    def set_mode(self, mode):
        if mode == 'original':
            self.data = self._data
        elif mode == 'pattern':
            self.data = self._pattern_data
        elif mode == 'all':
            self.data = list(zip(self._data, self._pattern_data))
            self.data = [_[0] + _[1] for _ in self.data]
        else:
            raise NotImplementedError

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
        batch[self.fuid] = data[0][idx]
        batch['in_' + self.fiid] = data[1][idx]
        batch[self.fiid] = data[2][idx]
        batch['seqlen'] = data[3][idx]
        batch['label'] = data[4][idx]
        batch['domain_id'] = data[5][idx]
        if self.phase != 'train':
            batch['user_hist'] = batch['in_' + self.fiid]
        return batch
    
class CondenseDataset(SeparateDataset):
    # def _condense_sequences(self, data):
    #     user_id, user_seq, target_item, seq_len, label, domain_id = data
    #     sorted_seq_len, sorted_index = torch.sort(seq_len, descending=True)
    #     sorted_seq_len = sorted_seq_len.tolist()
    #     sorted_seq = user_seq[sorted_index].tolist()
    #     sorted_target_item = target_item[sorted_index]
    #     merged_data = [[], [], [], [], [], []]
    #     pre_pointer, post_pointer = 0, len(user_seq) - 1
    #     while(pre_pointer <= post_pointer):
    #         cur_seq, cur_seq_len, cur_target_item = sorted_seq[pre_pointer], sorted_seq_len[pre_pointer], sorted_target_item[pre_pointer]
    #         # find to to appended
    #         while cur_seq_len <= self.max_seq_len:
    #             post_seq_len = sorted_seq_len[post_pointer]
    #             if (cur_seq_len + post_seq_len <= self.max_seq_len) and pre_pointer != post_pointer:
    #                 cur_seq = sorted_seq[post_pointer][:post_seq_len] + cur_seq[:cur_seq_len]
    #                 cur_seq_len = cur_seq_len + post_seq_len
    #                 post_pointer -= 1
    #             else:# Add padding and record this sequence
    #                 cur_seq = torch.tensor(cur_seq[:cur_seq_len] + [0] * (self.max_seq_len - cur_seq_len))
    #                 cur_seq_len = torch.tensor([cur_seq_len])
    #                 user_id_, label_, domain_id_ = torch.tensor([0]), torch.tensor([0]), torch.zeros_like(cur_seq)
    #                 merged_data[0].append(user_id_) # useless
    #                 merged_data[1].append(cur_seq)
    #                 merged_data[2].append(cur_target_item)
    #                 merged_data[3].append(cur_seq_len)
    #                 merged_data[4].append(label_) # useless
    #                 merged_data[5].append(domain_id_) # useless
    #                 pre_pointer += 1
    #                 break
    #     merged_data = [torch.stack(_).squeeze() for _ in merged_data]
    #     return merged_data

    def _condense_sequences(self, data):
        user_id, user_seq, target_item, seq_len, label, domain_id = data
        sorted_seq_len, sorted_index = torch.sort(seq_len, descending=True)
        sorted_seq_len = sorted_seq_len.tolist()
        sorted_seq = user_seq[sorted_index].tolist()
        sorted_target_item = target_item[sorted_index].tolist()
        merged_data = [[], [], [], [], [], []]
        pre_pointer, post_pointer = 0, len(user_seq) - 1
        while(pre_pointer <= post_pointer):
            cur_seq, cur_seq_len, cur_target_item = sorted_seq[pre_pointer], sorted_seq_len[pre_pointer], sorted_target_item[pre_pointer]
            # find to to appended
            while cur_seq_len <= self.max_seq_len:
                post_seq_len = sorted_seq_len[post_pointer]
                if (cur_seq_len + post_seq_len <= self.max_seq_len) and pre_pointer != post_pointer:
                    cur_seq = cur_seq[:cur_seq_len] + sorted_seq[post_pointer][:post_seq_len]
                    cur_target_item = cur_target_item[:cur_seq_len] + sorted_target_item[post_pointer][:post_seq_len]
                    cur_seq_len = cur_seq_len + post_seq_len
                    post_pointer -= 1
                else:# Add padding and record this sequence
                    cur_seq = torch.tensor(cur_seq[:cur_seq_len] + [0] * (self.max_seq_len - cur_seq_len))
                    cur_target_item = torch.tensor(cur_target_item[:cur_seq_len] + [0] * (self.max_seq_len - cur_seq_len))
                    cur_seq_len = torch.tensor([cur_seq_len])
                    user_id_, label_, domain_id_ = torch.tensor([0]), torch.tensor([0]), torch.zeros_like(cur_seq)
                    merged_data[0].append(user_id_) # useless
                    merged_data[1].append(cur_seq)
                    merged_data[2].append(cur_target_item)
                    merged_data[3].append(cur_seq_len)
                    merged_data[4].append(label_) # useless
                    merged_data[5].append(domain_id_) # useless
                    pre_pointer += 1
                    break
        merged_data = [torch.stack(_).squeeze() for _ in merged_data]
        return merged_data

    def _build(self):
        if self.phase == 'train':
            data = []
            for _ in self._data:
                data += _
            data = self.unpack(data)
            self._data = self._condense_sequences(data)
        else:
            self._data = {
                self.domain_name_list[idx]: self.unpack(_) for idx, _ in enumerate(self._data)
            }

class SplitDataset(SeparateDataset):
    def _condense_sequences(self, data):
        user_id, user_seq, target_item, seq_len, label, domain_id = data
        user_id, user_seq, target_item, seq_len, label, domain_id = user_id.tolist(), user_seq.tolist(), target_item.tolist(), seq_len.tolist(), label.tolist(), domain_id.tolist()
        merged_data = [[], [], [], [], [], []]
        for idx in range(len(user_seq)):
            if seq_len[idx] > self.max_seq_len // 2: # split into half
                cur_len = random.randint(5, seq_len[idx] - 5)
                merged_data[0].append(user_id[idx]) # useless
                merged_data[1].append(user_seq[idx][:cur_len] + [0] * (self.max_seq_len - cur_len))
                merged_data[2].append(target_item[idx][:cur_len] + [0] * (self.max_seq_len - cur_len))
                merged_data[3].append(cur_len)
                merged_data[4].append(label[idx]) # useless
                merged_data[5].append(domain_id[idx]) # useless

                merged_data[0].append(user_id[idx]) # useless
                merged_data[1].append(user_seq[idx][cur_len:] + [0] * cur_len)
                merged_data[2].append(target_item[idx][cur_len:] + [0] * cur_len)
                merged_data[3].append(seq_len[idx] - cur_len)
                merged_data[4].append(label[idx]) # useless
                merged_data[5].append(domain_id[idx]) # useless
            else:
                merged_data[0].append(user_id[idx]) # useless
                merged_data[1].append(user_seq[idx])
                merged_data[2].append(target_item[idx])
                merged_data[3].append(seq_len[idx])
                merged_data[4].append(label[idx]) # useless
                merged_data[5].append(domain_id[idx]) # useless
        merged_data = [torch.tensor(_, device=self.device) for _ in merged_data]
        return merged_data

    def _build(self):
        if self.phase == 'train':
            self.data = []
            for _ in self._data:
                self.data += _
            self.data = self.unpack(self.data)
            self.data = tuple(self._condense_sequences(self.data))
        else:
            self.data = {
                self.domain_name_list[idx]: self.unpack(_) for idx, _ in enumerate(self._data)
            }

class SelectionDataset(SeparateDataset):
    def __init__(self, config, phase='train') -> None:
        super().__init__(config, phase)
        self.strategy = 'random'

    def _condense_sequences(self, data):
        user_id, user_seq, target_item, seq_len, label, domain_id = data
        N = len(user_id)
        if self.strategy == 'random':
            selection = torch.randperm(N)[: int(N * 0.5)].tolist()
        return (
            user_id[selection],
            user_seq[selection],
            target_item[selection],
            seq_len[selection],
            label[selection],
            domain_id[selection],
        )

    def _build(self):
        if self.phase == 'train':
            self.data = []
            for _ in self._data:
                self.data += _
            self.data = self.unpack(self.data)
            self.data = tuple(self._condense_sequences(self.data))
        else:
            self.data = {
                self.domain_name_list[idx]: self.unpack(_) for idx, _ in enumerate(self._data)
            }

class ClusterDataset(SeparateDataset):
    def truncate_or_pad(self, seq_list, seqlen_list):
        

        return
        cur_seq_len = len(seq)
        if cur_seq_len > self.max_seq_len:
            return seq[-self.max_seq_len:], self.max_seq_len
        else:
            return seq + [0] * (self.max_seq_len - cur_seq_len), cur_seq_len

    def condense_sequences(self, similar_index):
        user_id, user_seq, target_item, seq_len, label, domain_id = self._data
        seq_list, seqlen_list = user_seq[similar_index], seq_len[similar_index]
        seq_list = seq_list.flip(-1).flatten(-2, -1)

        s_mask = seq_list != 0
        d_mask = torch.sum(s_mask, dim=1, keepdims=True) > torch.arange(seq_list.shape[1], device=self.device)
        new_seq_list = torch.zeros_like(seq_list, device=self.device) 
        new_seq_list[d_mask] = seq_list[s_mask]
        seq_list = new_seq_list[:, :self.max_seq_len]

        seq_list = seq_list.flip(-1)
        s_mask = seq_list != 0
        d_mask = torch.sum(s_mask, dim=1, keepdims=True) > torch.arange(seq_list.shape[1], device=self.device)
        new_seq_list = torch.zeros_like(seq_list, device=self.device) 
        new_seq_list[d_mask] = seq_list[s_mask]
        seq_list = new_seq_list
        seq_len = (seq_list != 0).sum(1)

        self.data = [user_id, seq_list, target_item, seq_len, label, domain_id]
