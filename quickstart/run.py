import os
import wandb
import datetime

from utils import *

def run(config: dict):
    wandb.init(
        # Set the project where this run will be logged
        project="KDD2024", 
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"{config['model']['model'] + config['data']['dataset']}", 
        # Track hyperparameters and run metadata
        config=config,
        # mode="disabled",
    )

    log_path = f"{config['model']['model']}/{config['data']['dataset']}/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}.log"
    logger = get_logger(log_path)

    logger.info('PID of this process: {}'.format(os.getpid()))

    dataset_list = prepare_datasets(config)
    logger.info(dataset_list[0])

    model = prepare_model(config, dataset_list)

    model.fit()
    model.evaluate()
    wandb.finish()