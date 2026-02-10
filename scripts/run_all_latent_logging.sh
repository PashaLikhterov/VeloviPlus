#!/usr/bin/env bash

set -euo pipefail

JOB_PREFIX="${JOB_PREFIX:-velovi-latent-log}"
WORKSPACE="${WORKSPACE:-/gpfs0/bgu-ofircohen/users/likhtepi/proj/scvi-tools}"
DATA_DIR="${DATA_DIR:-/gpfs0/bgu-ofircohen/users/likhtepi/proj/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE}/results/velovi_latent_logging_all}"
CONDA_ENV="${CONDA_ENV:-/gpfs0/bgu-ofircohen/users/likhtepi/conda_dirs/env/velovi-gnn}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-150}"
LOG_INTERVAL="${LOG_INTERVAL:-5}"
BATCH_SIZE="${BATCH_SIZE:-256}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_API_KEY="${WANDB_API_KEY:-}"

DATASETS=(
  pancreas_endocrinogenesis
  dentate_gyrus
  dentategyrus_lamanno
  gastrulation_erythroid
  fibroblast_reprogramming
  intestinal_organoid
  retina_development
  scnt_neuron_kcl
  human_hspc
  developing_human_brain
  fucci_u2os
  fucci_rpe1
)

WANDB_ENTITY_ARG=""
if [[ -n "${WANDB_ENTITY}" ]]; then
  WANDB_ENTITY_ARG="--wandb-entity ${WANDB_ENTITY}"
fi
WANDB_API_KEY_ARG=""
if [[ -n "${WANDB_API_KEY}" ]]; then
  WANDB_API_KEY_ARG="--wandb-api-key ${WANDB_API_KEY}"
fi

for DATASET in "${DATASETS[@]}"; do
  JOB_NAME="${JOB_PREFIX}-${DATASET}"
  JOB_SAFE="${JOB_NAME//_/-}"
  OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET}"

  runai-bgu submit cmd \
    -n "${JOB_SAFE}" \
    -c 32 \
    -m 80G \
    -g 1 \
    --conda "${CONDA_ENV}" \
    --working-dir "${WORKSPACE}" \
    -- "
      python -m scvi.experimental.velovi_improvements.log_latent_baseline \
        ${DATA_DIR} \
        --dataset ${DATASET} \
        --output-dir ${OUTPUT_DIR} \
        --total-epochs ${TOTAL_EPOCHS} \
        --log-interval ${LOG_INTERVAL} \
        --batch-size ${BATCH_SIZE} \
        --latent-dim 10 \
        --hidden-dim 256 \
        --num-workers 0 \
        --skip-preprocess \
        --use-wandb \
        --wandb-project RNA-Velocity \
        --wandb-run-group \"latent_logging_${DATASET}\" \
        ${WANDB_ENTITY_ARG} \
        ${WANDB_API_KEY_ARG}
    "
done
