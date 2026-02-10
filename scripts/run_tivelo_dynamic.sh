#!/usr/bin/env bash

set -euo pipefail

JOB_NAME="${JOB_NAME:-velovi-tivelo-dynamic}"
SAVE_FIGURES_LOCALLY="${SAVE_FIGURES_LOCALLY:-1}"
WORKSPACE="${WORKSPACE:-/gpfs0/bgu-ofircohen/users/likhtepi/proj/scvi-tools}"
DATA_DIR="${DATA_DIR:-/gpfs0/bgu-ofircohen/users/likhtepi/proj/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE}/results/velovi_tivelo_dynamic}"
CONDA_ENV="${CONDA_ENV:-/gpfs0/bgu-ofircohen/users/likhtepi/conda_dirs/env/velovi-gnn}"
DATASETS="${DATASETS:-pancreas_endocrinogenesis}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-120}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LATENT_DIM="${LATENT_DIM:-10}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SCVELO_N_JOBS="${SCVELO_N_JOBS:-32}"
TIVELO_PRIOR_STRENGTH="${TIVELO_PRIOR_STRENGTH:-0.4}"

PLOT_FLAG=""
if [[ "${PLOT_RESULTS:-0}" == "1" ]]; then
  PLOT_FLAG="--plot-results" \
      $( [[ "${SAVE_FIGURES_LOCALLY}" == "0" ]] && echo "--disable-local-figures" )
fi

runai-bgu submit cmd \
  -n "${JOB_NAME//_/-}" \
  -c 32 \
  -m 64G \
  -g 1 \
  --conda "${CONDA_ENV}" \
  --working-dir "${WORKSPACE}" \
  -- "
    python -m scvi.experimental.velovi_improvements.runner \
      ${DATA_DIR} \
      --datasets ${DATASETS} \
      --output-dir ${OUTPUT_DIR} \
      --warmup-epochs ${WARMUP_EPOCHS} \
      --total-epochs ${TOTAL_EPOCHS} \
      --batch-size ${BATCH_SIZE} \
      --latent-dim ${LATENT_DIM} \
      --hidden-dim ${HIDDEN_DIM} \
      --num-workers ${NUM_WORKERS} \
      --scvelo-n-jobs ${SCVELO_N_JOBS} \
      --disable-gnn \
      --disable-latent-smoothing \
      --transformer-use-tivelo \
      --tivelo-prior-strength ${TIVELO_PRIOR_STRENGTH} \
      --disable-checkpoints \
      ${PLOT_FLAG}
  "
