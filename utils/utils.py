import yaml
import random
import torch
import numpy as np
import os
import importlib
import torch.nn as nn
from copy import deepcopy

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
    model_class = get_model_class(config['model'])
    dataset_class = model_class._get_dataset_class(config)

    train_dataset = dataset_class(config, phase='train')
    val_dataset = dataset_class(config, phase='val')
    test_dataset = dataset_class(config, phase='test')

    train_dataset.build()
    val_dataset.build()
    test_dataset.build()

    return train_dataset, val_dataset, test_dataset

def prepare_model(config, dataset_list):
    model_class = get_model_class(config['model'])
    model = model_class(config, dataset_list)
    return model

def xavier_normal_initialization(module):
    if isinstance(module, nn.Embedding):
        nn.init.xavier_normal_(module.weight.data)
        if module.padding_idx is not None:
            nn.init.constant_(module.weight.data[module.padding_idx], 0.)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight.data)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

def normal_initialization(module, initial_range=0.02):
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=initial_range)
        if module.padding_idx is not None:
            nn.init.constant_(module.weight.data[module.padding_idx], 0.)
    elif isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=initial_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

def get_parameter_list(model, detach=True):
    para_list = [p.detach().cpu() for p in model.parameters()]
    return para_list

def flatten_state_dict(state_dict):
    return torch.cat([p.flatten() for _, p in state_dict.items()])

def load_config(config : dict):
    # dataset config
    path = os.path.join('configs', config['dataset'].lower() + '.yaml')
    with open(path, "r") as stream:
        config['data'] = yaml.safe_load(stream)
    config['data']['dataset'] = deepcopy(config['dataset'])
    config.pop('dataset')
    # basemodel config
    model_name = deepcopy(config['model'])
    path = os.path.join('configs', 'basemodel.yaml')
    with open(path, "r") as stream:
        config.update(yaml.safe_load(stream))
    # model config
    path = os.path.join('configs', model_name.lower() + '.yaml')
    with open(path, "r") as stream:
        model_config = yaml.safe_load(stream)
        for key, value in model_config.items():
            config[key].update(value)
    config['model']['model'] = model_name
    return config