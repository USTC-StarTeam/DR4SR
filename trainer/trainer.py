import torch
import logging

from utils import callbacks
from models.basemodel import BaseModel

class BaseTrainer(object):
    def __init__(self, args : dict, model : BaseModel, dataset_list : list) -> None:
        self.args = args
        self.ckpt_path = args['ckpt_path']
        self.logger = logging.getLogger('CDR')
        self.model = model
        self.dataset_list = dataset_list

        self.model.init_parameters()

    def fit(self):
        self.callbacks = callbacks.EarlyStopping(self, 'ndcg@20', self.args['dataset'])