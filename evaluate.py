"""
Re-evaluate trained runs on the test years, e.g. with a different aridity product.

    python evaluate.py --run_dir runs/nirv_rzsm_t6 --data_root work/zscore --aridity_dir work/aridity
"""

import argparse
import json
from pathlib import Path

import torch

from src.config import load_config
from src.evaluation import evaluate, summarise
from src.models import build_model
from src.training import build_dataset, make_loader


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run_dir', required=True, help='output directory of train.py')
    p.add_argument('--data_root', required=True)
    p.add_argument('--aridity_dir')
    p.add_argument('--growing_season', default='configs/growing_season.yml')
    p.add_argument('--split', default='test')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    config = load_config(run_dir / 'config.yaml')
    dataset = build_dataset(config, Path(args.data_root) / args.split, args.growing_season)
    loader = make_loader(dataset, config)

    runs = []
    for seed_dir in sorted(run_dir.glob('seed_*')):
        model = build_model(config)
        model.load_state_dict(torch.load(seed_dir / 'best.pt', map_location=args.device)['model'])
        metrics = evaluate(model.to(args.device), loader, args.device, args.aridity_dir)
        metrics['seed'] = int(seed_dir.name.split('_')[1])
        runs.append(metrics)
        print(f"{seed_dir.name}: ACC {metrics['acc']:.4f}  RMSE {metrics['rmse']:.4f}")

    with open(run_dir / f'{args.split}_summary.json', 'w') as f:
        json.dump({'runs': runs, 'summary': summarise(runs)}, f, indent=1)


if __name__ == '__main__':
    main()
