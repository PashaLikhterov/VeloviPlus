#!/usr/bin/env bash

set -euo pipefail

JOB_NAME="${JOB_NAME:-velovi-pancreas-trans-baseline}"
WORKSPACE="${WORKSPACE:-/gpfs0/bgu-ofircohen/users/likhtepi/proj/scvi-tools}"
DATA_DIR="${DATA_DIR:-/gpfs0/bgu-ofircohen/users/likhtepi/proj/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE}/results/velovi_pancreas_trans_baseline}"
CONDA_ENV="${CONDA_ENV:-/gpfs0/bgu-ofircohen/users/likhtepi/conda_dirs/env/velovi-gnn}"
SCVELO_N_JOBS="${SCVELO_N_JOBS:-32}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-120}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TRANS_ENCODER_NEIGHBORS="${TRANS_ENCODER_NEIGHBORS:-16}"
TRANS_ENCODER_WEIGHT="${TRANS_ENCODER_WEIGHT:-0.2}"

runai-bgu submit cmd \
  -n "${JOB_NAME//_/-}" \
  -c 64 \
  -m 128G \
  -g 1 \
  --conda "${CONDA_ENV}" \
  --working-dir "${WORKSPACE}" \
  -- "
    python -m scvi.experimental.velovi_improvements.runner \
      ${DATA_DIR} \
      --datasets pancreas_endocrinogenesis \
      --output-dir ${OUTPUT_DIR} \
      --warmup-epochs ${WARMUP_EPOCHS} \
      --total-epochs ${TOTAL_EPOCHS} \
      --batch-size ${BATCH_SIZE} \
      --latent-dim 10 \
      --hidden-dim 256 \
      --baseline-encoder transformer \
      --transformer-encoder-hidden-dim 256 \
      --transformer-encoder-layers 3 \
      --transformer-encoder-heads 8 \
      --transformer-encoder-dropout 0.1 \
      --transformer-encoder-max-neighbors ${TRANS_ENCODER_NEIGHBORS} \
      --transformer-encoder-neighbor-weight ${TRANS_ENCODER_WEIGHT} \
      --disable-gnn \
      --disable-latent-smoothing \
      --disable-scvelo-dynamic \
      --num-workers 0 \
      --skip-preprocess \
      --checkpoint-dir ${WORKSPACE}/checkpoints \
      --use-wandb \
      --wandb-project \"RNA-Velocity\" \
      --wandb-run-group \"pancreas_transformer_baseline\" \
      --plot-results \
      --scvelo-n-jobs ${SCVELO_N_JOBS}
  "
