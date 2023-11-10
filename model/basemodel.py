import os
import torch
import logging
import time
import eval

from tqdm import tqdm
from torch import optim
from utils import callbacks
from collections import defaultdict
from model.loss_func import *
from data.dataset import BaseDataset
from typing import Dict, List, Optional, Tuple

from utils.utils import xavier_normal_initialization


class BaseModel(nn.Module):

    def __init__(self, config : Dict, dataset_list : List[BaseDataset]) -> None:
        super().__init__()
        # register model-irrelevant config
        self.config = config
        self.ckpt_path = None
        self.logger = logging.getLogger('CDR')
        self.dataset_list = dataset_list
        self.device = config['device']
        self.domain_name_list = dataset_list[0].domain_name_list

        # register mode-relevant parameters
        self.embed_dim = config['embed_dim']
        self.max_seq_len = config['max_seq_len']
        self.num_users = dataset_list[0].num_users
        self.num_items = dataset_list[0].num_items
        self.item_embedding = nn.Embedding(self.num_items + 1, self.embed_dim)
        self.optimizer = self._get_optimizers()
        self.loss_fn = self._get_loss_func()

    def init_model(self):
        self = self.to(self.device)
        self.apply(xavier_normal_initialization)

    def _get_optimizers(self):
        opt_name = self.config['optimizer']
        lr = self.config['learning_rate']
        weight_decay = self.config['weight_decay']
        params = self.parameters()

        if opt_name.lower() == 'adam':
            optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif opt_name.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=lr, weight_decay=weight_decay)
        elif opt_name.lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
        elif opt_name.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
        elif opt_name.lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params, lr=lr)
        else:
            optimizer = optim.Adam(params, lr=lr)

        return optimizer

    def _get_loss_func(self):
        if self.config['loss_fn'] == 'bce':
            return BinaryCrossEntropyLoss()

    def forward(self):
        raise NotImplementedError

    def fit(self):
        self.callback = callbacks.EarlyStopping(self, 'ndcg@20', self.config['dataset'])
        self.logger.info('save_dir:' + self.callback.save_dir)
        self.init_model()
        self.logger.info(self)
        self.fit_loop()

    def fit_loop(self):
        try:
            nepoch = 0
            for e in range(self.config['epochs']):
                self.logged_metrics = {}
                self.logged_metrics['epoch'] = nepoch

                # training procedure
                tik_train = time.time()
                self.train()
                training_output_list = self.training_epoch(nepoch)
                tok_train = time.time()

                # validation procedure
                tik_valid = time.time()
                self.eval()
                if nepoch % 1 == 0:
                    for domain in self.domain_name_list:
                        val_dataset = self.dataset_list[1]
                        val_dataset.set_eval_domain(domain)
                        validation_output_list = self.validation_epoch(nepoch, val_dataset.get_loader())
                        self.validation_epoch_end(validation_output_list, domain)
                    all_domain_result = defaultdict(float)
                    for k, v in self.logged_metrics.items():
                        for domain_name in self.domain_name_list:
                            if domain_name in k: # result of a single domain
                                all_domain_result[k.removeprefix(domain_name + '_')] += v
                                break
                    self.logged_metrics.update(all_domain_result)
                tok_valid = time.time()

                self.training_epoch_end(training_output_list)

                # model is saved in callback when the callback return True.
                stop_training = self.callback(self, nepoch, self.logged_metrics)
                if stop_training:
                    break

                nepoch += 1

            self.callback.save_checkpoint(nepoch)
            self.ckpt_path = self.callback.get_checkpoint_path()
        except KeyboardInterrupt:
            # if catch keyboardinterrupt in training, save the best model.
            self.callback.save_checkpoint(nepoch)
            self.ckpt_path = self.callback.get_checkpoint_path()
        return

    def current_epoch_trainloaders(self, nepoch):
        return self.dataset_list[0].get_loader()

    def training_epoch(self, nepoch):
        output_list = []

        trn_dataloaders = self.current_epoch_trainloaders(nepoch)
        trn_dataloaders = [trn_dataloaders]

        for loader_idx, loader in enumerate(trn_dataloaders):
            outputs = []
            loader = tqdm(
                loader,
                total=len(loader),
                ncols=75,
                desc=f"Training {nepoch:>5}",
                leave=True,
            )
            for batch_idx, batch in enumerate(loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.optimizer.zero_grad()
                training_step_args = {'batch': batch}
                loss = self.training_step(**training_step_args)
                loss.backward()
                outputs.append({f"loss_{loader_idx}": loss.detach()})
                self.optimizer.step()
            output_list.append(outputs)
        return output_list
    
    def training_step(self, batch):
        query = self.forward(batch)
        pos_score = (query * self.item_embedding(batch['target_item'])).sum(-1)
        neg_score = (query * self.item_embedding(batch['neg_item'])).sum(-1)
        loss_value = self.loss_fn(pos_score, neg_score)
        return loss_value
    
    def training_epoch_end(self, output_list):
        output_list = [output_list] if not isinstance(output_list, list) else output_list
        for outputs in output_list:
            if isinstance(outputs, List):
                loss_metric = {'train_'+ k: torch.hstack([e[k] for e in outputs]).mean() for k in outputs[0]}
            elif isinstance(outputs, torch.Tensor):
                loss_metric = {'train_loss': outputs.item()}
            elif isinstance(outputs, Dict):
                loss_metric = {'train_'+k : v for k, v in outputs}
            self.logged_metrics.update(loss_metric)

        self.logger.info(self.logged_metrics)

    @torch.no_grad()
    def validation_epoch(self, nepoch, dataloader):
        output_list = []
        dataloader = tqdm(
            dataloader,
            total=len(dataloader),
            ncols=75,
            leave=False,
        )
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            # model validation results
            output = self.validation_step(batch)
            output_list.append(output)
        return output_list

    @torch.no_grad()
    def test_epoch(self, dataloader):
        output_list = []
        dataloader = tqdm(
            dataloader,
            total=len(dataloader),
            ncols=75,
            leave=False,
        )
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            # model validation results
            output = self.test_step(batch)
            output_list.append(output)
        return output_list

    def validation_epoch_end(self, outputs, domain):
        val_metrics = self.config['val_metrics']
        cutoff = self.config['cutoff']
        val_metric = eval.get_eval_metrics(val_metrics, cutoff, validation=True)
        if isinstance(outputs[0][0], List):
            out = self._test_epoch_end(outputs, val_metric)
            out = dict(zip(val_metric, out))
        elif isinstance(outputs[0][0], Dict):
            out = self._test_epoch_end(outputs, val_metric)
        out = {domain + '_' + k: v for k, v in out.items()}
        self.logged_metrics.update(out)
        return out

    def test_epoch_end(self, outputs, domain):
        test_metrics = self.config['test_metrics']
        cutoff = self.config['cutoff']
        test_metric = eval.get_eval_metrics(test_metrics, cutoff, validation=False)
        if isinstance(outputs[0][0], List):
            out = self._test_epoch_end(outputs, test_metric)
            out = dict(zip(test_metric, out))
        elif isinstance(outputs[0][0], Dict):
            out = self._test_epoch_end(outputs, test_metric)
        out = {domain + '_' + k: v for k, v in out.items()}
        self.logged_metrics.update(out)
        return out
    
    def _test_epoch_end(self, outputs, metrics):
        if isinstance(outputs[0][0], List):
            metric, bs = zip(*outputs)
            metric = torch.tensor(metric)
            bs = torch.tensor(bs)
            out = (metric * bs.view(-1, 1)).sum(0) / bs.sum()
        elif isinstance(outputs[0][0], Dict):
            metric_list, bs = zip(*outputs)
            bs = torch.tensor(bs)
            out = defaultdict(list)
            for o in metric_list:
                for k, v in o.items():
                    out[k].append(v)
            for k, v in out.items():
                metric = torch.tensor(v)
                out[k] = (metric * bs).sum() / bs.sum()
        return out

    def validation_step(self, batch):
        eval_metric = self.config['val_metrics']
        cutoff = self.config['cutoff'][0]
        return self._test_step(batch, eval_metric, [cutoff])

    def test_step(self, batch):
        eval_metric = self.config['test_metrics']
        cutoffs = self.config['cutoff']
        return self._test_step(batch, eval_metric, cutoffs)
    
    def _test_step(self, batch, metric, cutoffs):
        rank_m = eval.get_rank_metrics(metric)
        topk = self.config['topk']
        bs = batch['user_id'].size(0)
        assert len(rank_m) > 0
        score, topk_items = self.topk(batch, topk, batch['user_hist'])
        label = batch['target_item'].view(-1, 1) == topk_items
        # pos_rating = batch[self.frating].view(-1, 1)
        pos_rating = torch.ones_like(label, device=self.device)
        return {f"{name}@{cutoff}": func(label, pos_rating, cutoff) for cutoff in cutoffs for name, func in rank_m}, bs
    
    def topk(self, batch, k, user_h=None):
        query = self.forward(batch)
        more = user_h.size(1) if user_h is not None else 0
        score, topk_items = torch.topk(query @ self.item_embedding.weight.T, k + more)
        topk_items = topk_items + 1
        if user_h is not None:
            existing, _ = user_h.sort()
            idx_ = torch.searchsorted(existing, topk_items)
            idx_[idx_ == existing.size(1)] = existing.size(1) - 1
            score[torch.gather(existing, 1, idx_) == topk_items] = -float('inf')
            score, idx = score.topk(k)
            topk_items = torch.gather(topk_items, 1, idx)

        return score, topk_items
    
    def evaluate(self) -> Dict:
        r""" Predict for test data.
        
        Args:
            test_data(recstudio.data.Dataset): The dataset of test data, which is generated by RecStudio.

            verbose(bool, optimal): whether to show the detailed information.

        Returns:
            dict: dict of metrics. The key is the name of metrics.
        """
        test_data = self.dataset_list[-1]
        output = defaultdict(float)
        self.load_checkpoint(os.path.join(self.config['save_path'], self.ckpt_path))
        self.eval()

        for domain in self.domain_name_list:
            test_data.set_eval_domain(domain)
            test_loader = test_data.get_loader()
            output_list = self.test_epoch(test_loader)
            output.update(self.test_epoch_end(output_list), domain)
        for k, v in output.items():
            for domain_name in self.domain_name_list:
                if domain_name in k: # result of a single domain
                    output[k.removeprefix(domain_name + '_')] += v

        self.logger.info(output)
        return output
    
    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path)
        self.config = ckpt['config']
        self.load_state_dict(ckpt['parameters'])