import torch
import os
import numpy as np
import torch
import copy
import logging

from typing import List, Union, Tuple, Dict, Optional

class EarlyStopping(object):
    def __init__(
        self,
        model: torch.nn.Module,
        monitor: str,
        dataset_name: str,
        save_dir: Optional[str] = 'saved',
        filename: Optional[str] = None,
        patience: Optional[int] = 10,
        delta: Optional[float] = 0,
        mode: Optional[str] = 'max',
        ):
        r"""
        Early Stop and Model Checkpoint save callback.

        Args:

            monitor: quantity to monitor. By default it is None
                which saves a checkpoint only for the last epoch.

            save_dir: directory to save checkpoint. By default it is None
                which means not saving checkpoint.

            filename: filename of the checkpoint file. By default it is
                None which will be set as "epoch={}-val_{}={}.ckpt"

            patience: number of checks with no improvement after which training
                will be stopped. One check happens after every training epoch.

            delta: minimum change in the monitored quantity to qualify as an
                improvement, i.e. an absolute change of less than or equal to
                `min_delta`, will count as no improvement.

            mode: one of ``'min'``, ``'max'``. In ``'min'`` mode, training will
                stop when the quantity monitored has stopped decreasing and
                in ``'max'`` mode it will stop when the quantity monitored has
                stopped increasing.

        """

        self.monitor = monitor
        self.patience = patience
        self.delta = delta

        self.model_name = model.__class__.__name__
        self.save_dir = save_dir
        self.filename = filename

        if mode in ['min', 'max']:
            self.mode = mode
        else:
            raise ValueError(f"`mode` can only be `min` or `max`, \
                but `{mode}` is given.")

        self._counter = 0
        self.best_value = np.inf if self.mode=='min' else -np.inf
        self.logger = logging.getLogger('CDR')

        self.best_ckpt = {
            'config': model.config,
            'model': self.model_name,
            'epoch': 0,
            'parameters': copy.deepcopy(model.state_dict()),
            'metric': {self.monitor: np.inf if self.mode=='min' else -np.inf}
        }

        if filename != None:
            self._best_ckpt_path = filename
        else:
            for handler in self.logger.handlers:
                if type(handler) == logging.FileHandler:
                    _file_name = os.path.basename(handler.baseFilename).split('.')[0]
            self._best_ckpt_path = f"{self.model_name}/{dataset_name}/{_file_name}.ckpt"
        self.__check_save_dir()


    def __check_save_dir(self):
        if self.save_dir is not None:
            dir = os.path.dirname(os.path.join(self.save_dir, self._best_ckpt_path))
            if not os.path.exists(dir):
                os.makedirs(dir)

    def __call__(self, model, epoch, metrics):
        if self.monitor not in metrics:
            raise ValueError(f"monitor {self.monitor} not in given `metrics`.")
        if self.mode == 'max':
            if metrics[self.monitor] >= self.best_value+self.delta:
                self._reset_counter(model, epoch, metrics)
                self.logger.info("{} improved. Best value: {:.4f}".format(
                                self.monitor, metrics[self.monitor]))
                self.save_checkpoint(epoch)
            else:
                self._counter += 1
        else:
            if metrics[self.monitor] <= self.best_value-self.delta:
                self._reset_counter(model, epoch, metrics)
                self.logger.info("{} improved. Best value: {:.4f}".format(
                                self.monitor, metrics[self.monitor]))
                self.save_checkpoint(epoch)
            else:
                self._counter += 1

        if self._counter >= self.patience:
            self.logger.info(f"Early stopped. Since the metric {self.monitor} "
                             f"haven't been improved for {self._counter} epochs.")
            self.logger.info(f"The best score of {self.monitor} is "
                             f"{self.best_value:.4f} on epoch {self.best_ckpt['epoch']}")
            return True
        else:
            return False

    def _reset_counter(self, model: torch.nn.Module, epoch, value):
        self._counter = 0
        self.best_value = value[self.monitor]
        self.best_ckpt['parameters'] = copy.deepcopy(model.state_dict())
        self.best_ckpt['metric'] = value
        self.best_ckpt['epoch'] = epoch

    def save_checkpoint(self, epoch): # TODO haddle saving checkpoint in ddp
        if self.save_dir is not None:
            self.save_path = os.path.join(self.save_dir, self._best_ckpt_path)
            torch.save(self.best_ckpt, self.save_path)
            self.logger.info(f"Best model checkpoint saved in {self.save_path}.")
        else:
            raise ValueError(f"fail to save the model, self.save_dir can't be None!")

    def get_checkpoint_path(self):
        return self._best_ckpt_path
