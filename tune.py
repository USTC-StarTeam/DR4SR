import os
import yaml
import wandb
import quickstart
import pprint

from utils import *

if __name__ == '__main__':
    parser = get_default_parser()
    config = vars(parser.parse_args())

    config = load_config(config)
    setup_environment(config['train'])

    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'ndcg@20',
            'goal': 'maximize'   
        }
    }
    sweep_config['parameters'] = load_sweep_config(config)
    sweep_config = transform_config_into_sweep_config(sweep_config, config)

    sweep_id = wandb.sweep(sweep_config, project=f"KDD24-sweep-{config['model']['model']}")
    wandb.agent(sweep_id, quickstart.tune)
