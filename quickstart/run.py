import os
import datetime

from utils import *

def run(config: dict):
    log_path = f"{config['model']}/{config['dataset']}/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}.log"
    logger = get_logger(log_path)

    logger.info('PID of this process: {}'.format(os.getpid()))

    dataset_list = prepare_datasets(config)
    logger.info(dataset_list[0])

    model = prepare_model(config, dataset_list)

    model.fit()
    model.evaluate()