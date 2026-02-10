#!/usr/bin/env bash

set -euo pipefail

# Purpose: compare GNN with expression-only neighbors vs. GNN with boundary-aware
#          secondary neighbor graph (expression primary + boundary secondary).

JOB_PREFIX="${JOB_PREFIX:-velovi-pancreas-gnn-boundary}"
WORKSPACE="${WORKSPACE:-/gpfs0/bgu-ofircohen/users/likhtepi/proj/scvi-tools}"
DATA_DIR="${DATA_DIR:-/gpfs0/bgu-ofircohen/users/likhtepi/proj/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE}/results/velovi_pancreas_gnn_boundary}"
CONDA_ENV="${CONDA_ENV:-/gpfs0/bgu-ofircohen/users/likhtepi/conda_dirs/env/velovi-gnn}"
SCVELO_N_JOBS="${SCVELO_N_JOBS:-32}"
SAVE_FIGURES_LOCALLY="${SAVE_FIGURES_LOCALLY:-1}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-120}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-256}"
GNN_EPOCHS="${GNN_EPOCHS:-${TOTAL_EPOCHS}}"
GNN_BATCH_SIZE="${GNN_BATCH_SIZE:-${BATCH_SIZE}}"

DATASET="${DATASET:-pancreas_endocrinogenesis}"

BASE_ARGS=(
  --datasets ${DATASET}
  --warmup-epochs ${WARMUP_EPOCHS}
  --total-epochs ${TOTAL_EPOCHS}
  --batch-size ${BATCH_SIZE}
  --gnn-epochs ${GNN_EPOCHS}
  --gnn-batch-size ${GNN_BATCH_SIZE}
  --latent-dim 10
  --hidden-dim 256
  --gnn-hidden-dim 128
  --gnn-dropout 0.1
  --num-workers 0
  --stream-embed umap
  --velocity-laplacian-weight 0.05
  --velocity-angle-weight 0.02
  --checkpoint-dir ${WORKSPACE}/checkpoints
  --use-wandb
  --wandb-project RNA-Velocity
  --plot-results
)

maybe_no_local=""
if [[ "${SAVE_FIGURES_LOCALLY}" == "0" ]]; then
  maybe_no_local="--disable-local-figures"
fi

submit_job() {
  local mode="$1"   # expr or both
  local job_name="${JOB_PREFIX}-${mode}"
  local output_dir="${OUTPUT_ROOT}/${mode}"
  local run_group="pancreas_gnn_boundary_${mode}"

  echo "[VELOVI][TEST] Submitting ${job_name} (mode=${mode})"

  local job_safe="${job_name//_/-}"
  runai-bgu submit cmd \
    -n "${job_safe}" \
    -c 64 \
    -m 192G \
    -g 1 \
    --conda "${CONDA_ENV}" \
    --working-dir "${WORKSPACE}" \
    -- "
      python -m scvi.experimental.velovi_improvements.runner \
        ${DATA_DIR} \
        --output-dir ${output_dir} \
        ${BASE_ARGS[*]} \
        --scvelo-n-jobs ${SCVELO_N_JOBS} \
        --gnn-attention \
        --gnn-gate \
        --skip-preprocess \
        --wandb-run-group ${run_group} \
        $( [[ \"${mode}\" == \"expr\" ]] && echo "--gnn-neighbor-source expression" ) \
        $( [[ \"${mode}\" == \"both\" ]] && echo "--gnn-neighbor-source both" ) \
        ${maybe_no_local}
    "
}

# 1) Baseline: expression-only neighbors (no secondary graph)
submit_job expr

# 2) Boundary-aware: expression primary + boundary secondary via cross-graph gate
submit_job both
