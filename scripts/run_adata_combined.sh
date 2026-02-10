#!/usr/bin/env bash
set -euo pipefail

JOB_NAME="${JOB_NAME:-velovi-adata-combined}" 
WORKSPACE="${WORKSPACE:-/gpfs0/bgu-ofircohen/users/likhtepi/proj/scvi-tools}" 
DATA_DIR="${DATA_DIR:-/gpfs0/bgu-ofircohen/users/likhtepi/files_adata}" 
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE}/results/velovi_adata_combined}" 
CONDA_ENV="${CONDA_ENV:-/gpfs0/bgu-ofircohen/users/likhtepi/conda_dirs/env/velovi-gnn}" 
SCVELO_N_JOBS="${SCVELO_N_JOBS:-32}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-150}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-450}"
BATCH_SIZE="${BATCH_SIZE:-256}"
BRANCH_COUNT="${BRANCH_COUNT:-3}"
BRANCH_ENTROPY_WEIGHT="${BRANCH_ENTROPY_WEIGHT:-5e-4}"
DIRECTION_THRESHOLD="${DIRECTION_THRESHOLD:-0.6}"
BRANCH_ASSIGN_THRESHOLD="${BRANCH_ASSIGN_THRESHOLD:-0.5}"
BRANCH_DIRECTION_THRESHOLD="${BRANCH_DIRECTION_THRESHOLD:-0.6}"
MARKER_CONFIG="${MARKER_CONFIG:-${WORKSPACE}/src/scvi/experimental/velovi_improvements/default_markers.json}"
BRANCH_PRIOR_WEIGHT="${BRANCH_PRIOR_WEIGHT:-0.3}"
MARKER_WEIGHT="${MARKER_WEIGHT:-0.1}"
CHRONOLOGY_WEIGHT="${CHRONOLOGY_WEIGHT:-0.1}"

runai-bgu submit cmd \
  -n "${JOB_NAME//_/-}" \
  -c 128 \
  -m 320G \
  -g 1 \
  --conda "${CONDA_ENV}" \
  --working-dir "${WORKSPACE}" \
  -- "
    python -m scvi.experimental.velovi_improvements.runner \
      ${DATA_DIR} \
      --datasets adata_combined \
      --output-dir ${OUTPUT_DIR} \
      --warmup-epochs ${WARMUP_EPOCHS} \
      --total-epochs ${TOTAL_EPOCHS} \
      --batch-size ${BATCH_SIZE} \
      --latent-dim 10 \
      --hidden-dim 256 \
      --enable-branch-clocks \
      --branch-clock-count ${BRANCH_COUNT} \
      --branch-entropy-weight ${BRANCH_ENTROPY_WEIGHT} \
      --branch-assignment-threshold ${BRANCH_ASSIGN_THRESHOLD} \
      --branch-direction-threshold ${BRANCH_DIRECTION_THRESHOLD} \
      --branch-decoder-mode adapter \
      --branch-kinetics-mode offset \
      --branch-prior-weight ${BRANCH_PRIOR_WEIGHT} \
      --enable-marker-loss \
      --marker-weight ${MARKER_WEIGHT} \
      --marker-config ${MARKER_CONFIG} \
      --chronology-weight ${CHRONOLOGY_WEIGHT} \
      --enable-cycle-latent \
      --cycle-weight 0.1 \
      --enable-transformer-refinement \
      --baseline-encoder transformer \
      --transformer-encoder-hidden-dim 256 \
      --transformer-encoder-layers 3 \
      --transformer-encoder-heads 8 \
      --transformer-encoder-dropout 0.1 \
      --transformer-encoder-max-neighbors 20 \
      --transformer-hidden-dim 256 \
      --transformer-layers 3 \
      --transformer-heads 8 \
      --transformer-dropout 0.1 \
      --transformer-batch-size 128 \
      --transformer-learning-rate 7e-4 \
      --transformer-weight-alignment 1.0 \
      --transformer-weight-smooth 0.3 \
      --transformer-weight-direction 0.8 \
      --transformer-max-neighbors 8 \
      --transformer-weight-supervised 0.4 \
      --transformer-celltype-penalty cosine \
      --scvelo-n-jobs ${SCVELO_N_JOBS} \
      --checkpoint-dir ${WORKSPACE}/checkpoints \
      --use-wandb \
      --wandb-project "RNA-Velocity" \
      --wandb-run-group "adata_combined_branch" \
      --plot-results
  "
