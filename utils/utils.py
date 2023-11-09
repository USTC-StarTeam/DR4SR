import random
import torch
import numpy as np
import os
import importlib
import torch.nn as nn

def seed_everything(seed=1111):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_device(args):
    if isinstance(args['device'], int):
        # gpu id
        os.environ['CUDA_VISIBLE_DEVICES'] = args['device']
        args['device'] = 'cuda'

def setup_environment(args):
    seed_everything(args['seed'])
    set_device(args['device'])

def get_model_class(args):
    path = '.'.join(['model', args['model'].lower()])
    module = importlib.import_module(path, __name__)
    model_class = getattr(module, args['model'])
    return model_class

def prepare_datasets(args):
    model_class = get_model_class(args)
    dataset_class = model_class._get_dataset_class()

    train_dataset = dataset_class(args, phase='train')
    val_dataset = dataset_class(args, phase='val')
    test_dataset = dataset_class(args, phase='test')

    train_dataset.build()
    val_dataset.build()
    test_dataset.build()

    return train_dataset, val_dataset, test_dataset

def prepare_model(args, train_dataset):
    model_class = get_model_class(args)
    model = model_class(args, train_dataset)
    return model

def prepare_trainer(args, model, dataset_list):
    model_class = get_model_class(args)
    trainer_class = model_class._get_trainer_class()

    trainer = trainer_class(args, model, dataset_list)
    return trainer

def xavier_normal_initialization(module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)
