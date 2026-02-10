#!/usr/bin/env bash

set -euo pipefail

JOB_NAME="${JOB_NAME:-velovi-pancreas-uncertainty}"
WORKSPACE="${WORKSPACE:-/gpfs0/bgu-ofircohen/users/likhtepi/proj/scvi-tools}"
DATA_DIR="${DATA_DIR:-/gpfs0/bgu-ofircohen/users/likhtepi/proj/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE}/results/velovi_pancreas_uncertainty}"
CONDA_ENV="${CONDA_ENV:-/gpfs0/bgu-ofircohen/users/likhtepi/conda_dirs/env/velovi-gnn}"
SCVELO_N_JOBS="${SCVELO_N_JOBS:-32}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-120}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TRANS_ENCODER_NEIGHBORS="${TRANS_ENCODER_NEIGHBORS:-16}"
DIRECTION_THRESHOLD="${DIRECTION_THRESHOLD:-0.8}"

runai-bgu submit cmd \
  -n "${JOB_NAME//_/-}" \
  -c 128 \
  -m 256G \
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
      --direction-check-threshold ${DIRECTION_THRESHOLD} \
      --enable-transformer-refinement \
      --transformer-use-uncertainty-weight \
      --transformer-epochs 15 \
      --transformer-hidden-dim 256 \
      --transformer-layers 3 \
      --transformer-heads 8 \
      --transformer-dropout 0.1 \
      --transformer-batch-size 128 \
      --transformer-learning-rate 1e-3 \
      --transformer-weight-alignment 1.0 \
      --transformer-weight-smooth 0.2 \
      --transformer-weight-direction 0.7 \
      --transformer-neighbor-max-distance 4.0 \
      --transformer-max-neighbors 8 \
      --transformer-weight-supervised 0.5 \
      --transformer-celltype-penalty cosine \
      --scvelo-n-jobs ${SCVELO_N_JOBS} \
      --checkpoint-dir ${WORKSPACE}/checkpoints \
      --use-wandb \
      --wandb-project \"RNA-Velocity\" \
      --wandb-run-group \"pancreas_uncertainty_weight\" \
      --plot-results
  "
