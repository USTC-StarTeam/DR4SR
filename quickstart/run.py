import datetime

from utils import *

def run(args: dict):
    log_path = f"{args['model']}/{args['dataset']}/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}.log"
    logger = get_logger(log_path)

    dataset_list = prepare_datasets(args)
    logger.info(dataset_list[0])

    model = prepare_model(args, dataset_list[0])

    trainer = prepare_trainer(args, model, dataset_list)

    trainer.fit(*dataset_list[:2], model)
    trainer.evaluate(dataset_list[-1])