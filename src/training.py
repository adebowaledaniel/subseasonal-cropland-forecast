"""Training loop with masked loss, LR scheduling, early stopping and best-model checkpointing."""

import json
import logging
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import (DayOfYearDataset, DecoupledHistoryDataset, PatchSequenceDataset,
                   load_growing_season, model_kwargs)

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def masked_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                kind: str = 'huber', delta: float = 1.0) -> torch.Tensor:
    """Mean MSE or Huber loss over observed target pixels."""
    if not mask.any():
        return pred.sum() * 0.0
    if kind == 'mse':
        return F.mse_loss(pred[mask], target[mask])
    if kind == 'huber':
        return F.huber_loss(pred[mask], target[mask], delta=delta)
    raise ValueError(f'unknown loss: {kind}')


def build_dataset(config: dict, split_dir, growing_season_file=None, **kwargs):
    """Dataset for one split as specified by the config; keyword arguments override it."""
    growing_season = None
    if config['growing_season_only']:
        growing_season = load_growing_season(growing_season_file)
    params = dict(target_band=config['target_band'], forecast_horizon=config['forecast_horizon'],
                  growing_season=growing_season, min_valid_ratio=config['min_valid_ratio'])
    if config['decoupled_history']:
        dataset_cls = DecoupledHistoryDataset
        params.update(zip(('tlen_nirv', 'tlen_rzsm'), config['decoupled_history']))
    else:
        dataset_cls = DayOfYearDataset if config['day_of_year'] else PatchSequenceDataset
        params.update(input_bands=config['input_bands'], tlen=config['tlen'],
                      lagged_rzsm=config['lagged_rzsm'], lag_variable=config['lag_variable'])
    params.update(kwargs)
    return dataset_cls(split_dir, **params)


def make_loader(dataset, config: dict, train: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=train, drop_last=train,
                      num_workers=config['num_workers'])


class Trainer:
    """
    Adam with ReduceLROnPlateau on the validation loss; stops after `patience` epochs
    without an improvement larger than `min_delta` and keeps the lowest-loss weights.
    """

    def __init__(self, model, config: dict, output_dir, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint = self.output_dir / 'best.pt'
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                          weight_decay=config['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=config['lr_factor'], patience=config['lr_patience'])

    def loss(self, batch: dict) -> torch.Tensor:
        inputs = batch['inputs'].to(self.device)
        target = batch['target'].to(self.device)
        mask = batch['target_mask'].to(self.device)
        pred = self.model(inputs, **model_kwargs(batch, self.device))
        return masked_loss(pred, target, mask, self.config['loss'], self.config['huber_delta'])

    def train_epoch(self, loader) -> float:
        self.model.train()
        losses = []
        for batch in loader:
            self.optimizer.zero_grad()
            loss = self.loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            self.optimizer.step()
            losses.append(loss.item())
        return float(np.mean(losses))

    @torch.no_grad()
    def validate(self, loader) -> float:
        self.model.eval()
        return float(np.mean([self.loss(batch).item() for batch in loader]))

    def fit(self, train_loader, val_loader) -> Dict:
        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        best_loss, stop_best, wait = float('inf'), float('inf'), 0
        for epoch in range(1, self.config['epochs'] + 1):
            start = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            self.scheduler.step(val_loss)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['lr'].append(self.optimizer.param_groups[0]['lr'])
            log.info('epoch %d  train %.5f  val %.5f  (%.0fs)', epoch, train_loss, val_loss,
                     time.time() - start)

            if val_loss < best_loss:
                best_loss = val_loss
                history['best_epoch'] = epoch
                torch.save({'epoch': epoch, 'model': self.model.state_dict(),
                            'config': self.config}, self.checkpoint)
            if val_loss < stop_best - self.config['min_delta']:
                stop_best, wait = val_loss, 0
            else:
                wait += 1
                if wait >= self.config['patience']:
                    log.info('early stopping at epoch %d', epoch)
                    break

        history['best_val_loss'] = best_loss
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=1)
        self.model.load_state_dict(torch.load(self.checkpoint, map_location=self.device)['model'])
        return history
