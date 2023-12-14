import os
import torch
import logging
import time
import evaluation

from tqdm import tqdm
from torch import optim
from utils import callbacks
from utils.utils import get_parameter_list
from collections import defaultdict
from model.loss_func import *
from data.dataset import BaseDataset
from typing import Dict, List, Optional, Tuple

from model.sasrec import SASRec

class Teacher(SASRec):
    def __init__(self, config: Dict, dataset_list: List[BaseDataset]) -> None:
        super().__init__(config, dataset_list)
        self.timestamps = [get_parameter_list(self, detach=True)]

    @staticmethod
    def detach_state_dict(state_dict):
        for k, v in state_dict.items():
            state_dict[k] = v.detach().cpu().clone()
        return state_dict

    def training_epoch_end(self, output_list):
        super().training_epoch_end(output_list)
        self.timestamps.append(get_parameter_list(self, detach=True))

    def training_end(self):
        path = self.callback.save_path[:-5] # remove the last .ckpt postfix
        path = path + '-timestamp.pth' # add postfix for file name
        torch.save(self.timestamps, path)