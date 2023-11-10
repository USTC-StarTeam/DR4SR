import yaml
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

def set_device(config):
    if isinstance(config['device'], int):
        # gpu id
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config['device'])
        config['device'] = 'cuda'

def setup_environment(config):
    seed_everything(config['seed'])
    set_device(config)

def get_model_class(config):
    path = '.'.join(['model', config['model'].lower()])
    module = importlib.import_module(path, __name__)
    model_class = getattr(module, config['model'])
    return model_class

def prepare_datasets(config):
    model_class = get_model_class(config)
    dataset_class = model_class._get_dataset_class()

    train_dataset = dataset_class(config, phase='train')
    val_dataset = dataset_class(config, phase='val')
    test_dataset = dataset_class(config, phase='test')

    train_dataset.build()
    val_dataset.build()
    test_dataset.build()

    return train_dataset, val_dataset, test_dataset

def prepare_model(config, dataset_list):
    model_class = get_model_class(config)
    model = model_class(config, dataset_list)
    return model

def xavier_normal_initialization(module):
    if isinstance(module, nn.Embedding):
        nn.init.xavier_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight.data)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)

def load_config(config : dict):
    # dataset config
    path = os.path.join('configs', config['dataset'].lower() + '.yaml')
    with open(path, "r") as stream:
        config.update(yaml.safe_load(stream))
    # basemodel config
    path = os.path.join('configs', 'basemodel.yaml')
    with open(path, "r") as stream:
        config.update(yaml.safe_load(stream))
    # model config
    path = os.path.join('configs', config['model'].lower() + '.yaml')
    with open(path, "r") as stream:
        config.update(yaml.safe_load(stream))
    return config