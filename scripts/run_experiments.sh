#!/usr/bin/env bash
# Train every model behind the paper (five seeds each).
# scripts/run_experiments.sh DATA_ROOT ARIDITY_DIR
set -euo pipefail

DATA=$1
ARIDITY=$2
OUT=${3:-runs}

train() { python train.py --data_root "$DATA" --aridity_dir "$ARIDITY" "$@"; }

# Huber loss, all input windows: the NIRv+RZSM domain skill, every aridity-stratified result
# and the ConvLSTM reference runs of the supplementary analyses.
for tlen in 2 4 6 8 10 12; do
  for cfg in nirv_rzsm nirv_only nirv_lagrzsm; do
    train --config configs/$cfg.yaml --tlen $tlen --output_dir "$OUT/huber/${cfg}_t$tlen"
  done
done

# MSE loss: the NIRv-only and NIRv+lagRZSM domain skill. With the NIRv+RZSM runs below,
# seeds 42, 123 and 2024 at T=6 give the skill-vs-lead curves, and seed 42 gives the skill
# maps (T=6), the SHAP attribution and the drought case study (T=10).
for tlen in 2 4 6 8 10 12; do
  for cfg in nirv_only nirv_lagrzsm; do
    train --config configs/$cfg.yaml --tlen $tlen --loss mse --output_dir "$OUT/mse/${cfg}_t$tlen"
  done
done
for tlen in 6 10; do
  train --config configs/nirv_rzsm.yaml --tlen $tlen --loss mse --output_dir "$OUT/mse/nirv_rzsm_t$tlen"
done

# NDVI target.
for cfg in ndvi_rzsm ndvi_only ndvi_lagrzsm; do
  train --config configs/$cfg.yaml --output_dir "$OUT/huber/${cfg}_t6"
done

# Non-spatial LSTM baselines.
for cfg in lstm_nirv_rzsm lstm_nirv_only; do
  train --config configs/$cfg.yaml --output_dir "$OUT/baselines/${cfg}_t6"
done

# Input-window ablations.
for cfg in history_a1 history_a2 history_a3 history_a4; do
  train --config configs/ablation/$cfg.yaml --output_dir "$OUT/ablation/$cfg"
done
for tlen in 6 10; do
  for cfg in doy_nirv_rzsm doy_nirv_only; do
    train --config configs/ablation/$cfg.yaml --tlen $tlen --output_dir "$OUT/ablation/${cfg}_t$tlen"
  done
done

# Leave-one-country-out folds, trained with MSE loss as published.
for country in TZA ZWE; do
  python supplementary/loco.py build --data_root "$DATA" --country $country --output "${DATA}_loco_$country"
  for tlen in 6 10; do
    for cfg in nirv_rzsm nirv_only; do
      python train.py --config configs/$cfg.yaml --data_root "${DATA}_loco_$country" --aridity_dir "$ARIDITY" \
        --tlen $tlen --loss mse --output_dir "$OUT/loco/${country}_${cfg}_t$tlen"
    done
  done
done
