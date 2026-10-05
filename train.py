"""
Train a forecasting model for one or more seeds and evaluate it on the test years.

    python train.py --config configs/nirv_rzsm.yaml --data_root work/zscore \
        --aridity_dir work/aridity --output_dir runs/nirv_rzsm_t6
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from src.config import load_config, save_config
from src.evaluation import evaluate, summarise
from src.models import build_model, count_parameters
from src.training import Trainer, build_dataset, make_loader, set_seed

SEEDS = [42, 123, 2024, 7, 99]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--data_root', required=True, help='directory with train/, val/ and test/')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--aridity_dir', help='per-location binary_ai.tif rasters, for WL/EL skill')
    p.add_argument('--growing_season', default='configs/growing_season.yml')
    p.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    p.add_argument('--tlen', type=int)
    p.add_argument('--loss', choices=['huber', 'mse'])
    p.add_argument('--epochs', type=int)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    config = load_config(args.config, tlen=args.tlen, loss=args.loss, epochs=args.epochs)
    output_dir = Path(args.output_dir)
    save_config(config, output_dir / 'config.yaml')

    root = Path(args.data_root)
    train_set, val_set, test_set = (build_dataset(config, root / split, args.growing_season)
                                    for split in ('train', 'val', 'test'))
    logging.info('samples: train %d, val %d, test %d', len(train_set), len(val_set), len(test_set))

    runs = []
    for seed in args.seeds:
        set_seed(seed)
        model = build_model(config)
        run_dir = output_dir / f'seed_{seed}'
        trainer = Trainer(model, config, run_dir, args.device)
        history = trainer.fit(make_loader(train_set, config, train=True), make_loader(val_set, config))

        metrics = evaluate(trainer.model, make_loader(test_set, config), args.device, args.aridity_dir)
        metrics.update(seed=seed, best_epoch=history['best_epoch'],
                       best_val_loss=history['best_val_loss'], parameters=count_parameters(model))
        with open(run_dir / 'test_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=1)
        logging.info('seed %d: ACC %.4f  RMSE %.4f  SS %.4f', seed, metrics['acc'],
                     metrics['rmse'], metrics['skill_score'])
        runs.append(metrics)

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summarise(runs), f, indent=1)


if __name__ == '__main__':
    main()
