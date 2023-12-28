import os
import wandb
import torch
import logging
import time
import evaluation

from tqdm import tqdm
from torch import optim
from utils import callbacks
from collections import defaultdict
from model.loss_func import *
from data.dataset import *
from typing import Dict, List, Optional, Tuple

from utils.utils import xavier_normal_initialization, normal_initialization


class BaseModel(nn.Module):

    def __init__(self, config : Dict, dataset_list : List[BaseDataset]) -> None:
        super().__init__()
        # register model-irrelevant config
        self.config = config
        self.ckpt_path = None
        self.logger = logging.getLogger('CDR')
        self.dataset_list = dataset_list
        self.device = config['train']['device']
        self.fuid = 'user_id'
        self.fiid = 'item_id'
        self.domain_name_list = dataset_list[0].domain_name_list
        self.domain_user_mapping = dataset_list[0].domain_user_mapping
        self.domain_item_mapping = dataset_list[0].domain_item_mapping
        self.training_time = 0
        self.inference_time = 0

        # register mode-relevant parameters
        self.embed_dim = config['model']['embed_dim']
        self.max_seq_len = config['data']['max_seq_len']
        self.num_users = dataset_list[0].num_users
        self.num_items = dataset_list[0].num_items
        self.item_embedding = nn.Embedding(self.num_items, self.embed_dim, padding_idx=0)

    def _init_model(self, train_data):
        self.apply(normal_initialization)
        self = self.to(self.device)
        self.optimizer = self._get_optimizers()
        self.loss_fn = self._get_loss_func()

    def _neg_sampling(self, batch):
        user_seq = batch['in_' + self.fiid]
        weight = torch.ones(user_seq.shape[0], self.num_items, device=self.device)
        _idx = torch.arange(user_seq.size(0), device=self.device).view(-1, 1).expand_as(user_seq)
        weight[_idx, user_seq] = 0.0
        weight[:, 0] = 0 # padding
        neg_idx = torch.multinomial(weight, self.config['train']['num_neg'] * self.max_seq_len, replacement=True)
        neg_idx = neg_idx.reshape(user_seq.shape[0], self.max_seq_len, self.config['train']['num_neg'])
        return neg_idx

    def _get_dataset_class(config):
        if config['data']['dataset_class'] == 'condense':
            return CondenseDataset
        elif config['data']['dataset_class'] == 'general':
            return SeparateDataset
        elif config['data']['dataset_class'] == 'selection':
            return SelectionDataset

    def _get_optimizers(self):
        opt_name = self.config['train']['optimizer']
        lr = self.config['train']['learning_rate']
        weight_decay = self.config['train']['weight_decay']
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
        if self.config['model']['loss_fn'] == 'bce':
            return BinaryCrossEntropyLoss()
        elif self.config['model']['loss_fn'] == 'bpr':
            return BPRLoss()

    def forward(self):
        raise NotImplementedError

    def fit(self):
        self.callback = callbacks.EarlyStopping(self, 'ndcg@20', self.config['data']['dataset'], patience=self.config['train']['early_stop_patience'])
        self.analyzer = callbacks.Analyzer(self)
        self.logger.info('save_dir:' + self.callback.save_dir)
        self._init_model(self.dataset_list[0])
        self.logger.info(self)
        self.fit_loop()

    def fit_loop(self):
        try:
            nepoch = 0
            self.train_start()
            for e in range(self.config['train']['epochs']):
                self.logged_metrics = {}
                self.logged_metrics['epoch'] = nepoch

                # training procedure
                tik_train = time.time()
                self.train()
                training_output_list = self.training_epoch(nepoch)
                tok_train = time.time()
                self.training_time += tok_train - tik_train

                # validation procedure
                tik_valid = time.time()
                self.eval()
                if nepoch % 1 == 0:
                    for domain in self.domain_name_list:
                        val_dataset = self.dataset_list[1]
                        val_dataset.set_eval_domain(domain)
                        self.set_eval_domain(domain)
                        validation_output_list = self.validation_epoch(nepoch, val_dataset.get_loader())
                        self.validation_epoch_end(validation_output_list, domain)
                    all_domain_result = defaultdict(float)
                    for k, v in self.logged_metrics.items():
                        for domain_name in self.domain_name_list:
                            if domain_name in k: # result of a single domain
                                all_domain_result[k.removeprefix(domain_name + '_')] += v
                                break
                    self.logged_metrics.update(all_domain_result)
                    wandb.log(all_domain_result)
                tok_valid = time.time()
                self.inference_time += tok_valid - tik_valid

                self.training_epoch_end(training_output_list)

                # model is saved in callback when the callback return True.
                stop_training = self.callback(self, nepoch, self.logged_metrics)
                if stop_training:
                    break

                nepoch += 1

            self.training_end()
            self.callback.save_checkpoint(nepoch)
            self.ckpt_path = self.callback.get_checkpoint_path()
        except KeyboardInterrupt:
            # if catch keyboardinterrupt in training, save the best model.
            self.callback.save_checkpoint(nepoch)
            self.ckpt_path = self.callback.get_checkpoint_path()
        return

    def current_epoch_trainloaders(self, nepoch):
        return self.dataset_list[0].get_loader()

    def train_start(self):
        pass

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
                leave=False,
            )
            for batch_idx, batch in enumerate(loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch['neg_item'] = self._neg_sampling(batch)
                self.optimizer.zero_grad()
                training_step_args = {'batch': batch}
                loss = self.training_step(**training_step_args)
                loss.backward()
                self.optimizer.step()
                outputs.append({f"loss_{loader_idx}": loss.detach()})
            output_list.append(outputs)
        return output_list
    
    def training_step(self, batch):
        query = self.forward(batch)
        pos_score = (query * self.item_embedding.weight[batch[self.fiid]]).sum(-1)
        neg_score = (query.unsqueeze(-2) * self.item_embedding.weight[batch['neg_item']]).sum(-1)
        pos_score[batch[self.fiid] == 0] = -torch.inf # padding

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
        self.logger.info(f'training_time: {self.training_time}')
        self.logger.info(f'inference_time: {self.inference_time}')

    def training_end(self):
        pass

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
        val_metrics = self.config['eval']['val_metrics']
        cutoff = self.config['eval']['cutoff']
        val_metric = evaluation.get_eval_metrics(val_metrics, cutoff, validation=True)
        if isinstance(outputs[0][0], List):
            out = self._test_epoch_end(outputs, val_metric)
            out = dict(zip(val_metric, out))
        elif isinstance(outputs[0][0], Dict):
            out = self._test_epoch_end(outputs, val_metric)
        out = {domain + '_' + k: v for k, v in out.items()}
        self.logged_metrics.update(out)
        self.analyzer.analyze_epoch()
        return out

    def test_epoch_end(self, outputs, domain):
        test_metrics = self.config['eval']['test_metrics']
        cutoff = self.config['eval']['cutoff']
        test_metric = evaluation.get_eval_metrics(test_metrics, cutoff, validation=False)
        if isinstance(outputs[0][0], List):
            out = self._test_epoch_end(outputs, test_metric)
            out = dict(zip(test_metric, out))
        elif isinstance(outputs[0][0], Dict):
            out = self._test_epoch_end(outputs, test_metric)
        out = {domain + '_' + k: v for k, v in out.items()}
        self.logged_metrics.update(out)
        self.analyzer.analyze_epoch()
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
        eval_metric = self.config['eval']['val_metrics']
        cutoff = self.config['eval']['cutoff'][0]
        return self._test_step(batch, eval_metric, [cutoff])

    def test_step(self, batch):
        eval_metric = self.config['eval']['test_metrics']
        cutoffs = self.config['eval']['cutoff']
        return self._test_step(batch, eval_metric, cutoffs)
    
    def _test_step(self, batch, metric, cutoffs):
        rank_m = evaluation.get_rank_metrics(metric)
        topk = self.config['eval']['topk']
        bs = batch['user_id'].size(0)
        assert len(rank_m) > 0
        score, topk_items = self.topk(batch, topk, batch['user_hist'])
        label = batch[self.fiid].view(-1, 1) == topk_items
        pos_rating = batch['label'].view(-1, 1)

        # analyzer
        analyzer_rst = {f"{name}@{cutoff}": func(label, pos_rating, cutoff, mean=False) for cutoff in cutoffs for name, func in rank_m}
        self.analyzer.record_batch(batch[self.fuid], batch['user_hist'], analyzer_rst)

        return {k: v.mean() for k, v in analyzer_rst.items()}, bs

        return {f"{name}@{cutoff}": func(label, pos_rating, cutoff) for cutoff in cutoffs for name, func in rank_m}, bs
    
    def topk(self, batch, k, user_h=None):
        query = self.forward(batch)
        more = user_h.size(1) if user_h is not None else 0
        real_score = query @ self.item_embedding.weight[:self.num_items].T # remove extra padding
        domain_mask = torch.ones(1, self.num_items, dtype=torch.bool, device=self.device)
        domain_mask[:, self.domain_item_mapping[self.eval_domain]] = 0
        masked_score : torch.Tensor = real_score.masked_fill(domain_mask, -torch.inf)
        user_h[user_h == -1] = 0 # index -1 is invalid for torch.scatter, so we just change it with the PAD id
        masked_score = torch.scatter(masked_score, 1, user_h, -torch.inf)

        score, topk_items = torch.topk(masked_score, k)

        return score, topk_items

    def set_eval_domain(self, domain):
        self.eval_domain = domain

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
        self.load_checkpoint(os.path.join(self.config['eval']['save_path'], self.ckpt_path))
        self.eval()

        for domain in self.domain_name_list:
            test_data.set_eval_domain(domain)
            self.set_eval_domain(domain)
            test_loader = test_data.get_loader()
            output_list = self.test_epoch(test_loader)
            output.update(self.test_epoch_end(output_list, domain))
        all_domain_result = defaultdict(float)
        for k, v in output.items():
            for domain_name in self.domain_name_list:
                if domain_name in k: # result of a single domain
                    all_domain_result[k.removeprefix(domain_name + '_')] += v
        output.update(all_domain_result)

        self.logger.info(output)
        wandb.log(output)
        wandb.log({'training_time': self.training_time, 'inference_time': self.inference_time})
        return output
    
    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path)
        self.config = ckpt['config']
        self.load_state_dict(ckpt['parameters'])