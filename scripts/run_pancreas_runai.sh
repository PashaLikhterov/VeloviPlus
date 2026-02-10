#!/usr/bin/env bash

set -euo pipefail

JOB_NAME="${JOB_NAME:-velovi-pancreas-gnn2}"
SAVE_FIGURES_LOCALLY="${SAVE_FIGURES_LOCALLY:-1}"
WORKSPACE="${WORKSPACE:-/gpfs0/bgu-ofircohen/users/likhtepi/proj/scvi-tools}"
DATA_DIR="${DATA_DIR:-/gpfs0/bgu-ofircohen/users/likhtepi/proj/velocity/scvi-tools/velovi_datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE}/results/velovi_pancreas_dualgraph}"
CONDA_ENV="${CONDA_ENV:-/gpfs0/bgu-ofircohen/users/likhtepi/conda_dirs/env/velovi-gnn}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-120}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-256}"
GNN_EPOCHS="${GNN_EPOCHS:-${TOTAL_EPOCHS}}"
GNN_BATCH_SIZE="${GNN_BATCH_SIZE:-${BATCH_SIZE}}"

runai-bgu submit cmd \
  -n "${JOB_NAME//_/-}" \
  -c 32 \
  -m 64G \
  -g 2 \
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
      --gnn-epochs ${GNN_EPOCHS} \
      --gnn-batch-size ${GNN_BATCH_SIZE} \
      --latent-dim 10 \
      --hidden-dim 256 \
      --gnn-hidden-dim 128 \
      --gnn-dropout 0.1 \
      --gnn-continuity-weight 0.2 \
      --gnn-continuity-horizon 1.0 \
      --num-workers 0 \
      --gnn-neighbor-source both \
      --gnn-attention \
      --gnn-gate \
      --skip-preprocess \
      --enable-gnn-latent \
      --velocity-laplacian-weight 0.05 \
      --velocity-angle-weight 0.02 \
      --stream-embed umap \
      --plot-results \
          $( [[ "${SAVE_FIGURES_LOCALLY}" == "0" ]] && echo "--disable-local-figures" )
  "
