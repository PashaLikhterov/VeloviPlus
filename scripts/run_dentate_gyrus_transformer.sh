#!/usr/bin/env bash

set -euo pipefail

JOB_NAME="${JOB_NAME:-velovi-dentategyrus-transformer-2}"
SAVE_FIGURES_LOCALLY="${SAVE_FIGURES_LOCALLY:-1}"
WORKSPACE="${WORKSPACE:-/gpfs0/bgu-ofircohen/users/likhtepi/proj/scvi-tools}"
DATA_DIR="${DATA_DIR:-${SCVELO_CACHE_DIR:-/gpfs0/bgu-ofircohen/users/likhtepi/.cache/scvelo}}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE}/results/velovi_dentategyrus_transformer}"
CONDA_ENV="${CONDA_ENV:-/gpfs0/bgu-ofircohen/users/likhtepi/conda_dirs/env/velovi-gnn}"

JOB_NAME_SAFE="${JOB_NAME//_/-}"
runai-bgu submit cmd \
  -n "${JOB_NAME_SAFE}" \
  -c 64 \
  -m 64G \
  -g 2 \
  --conda "${CONDA_ENV}" \
  --working-dir "${WORKSPACE}" \
  -- "
    python -m scvi.experimental.velovi_improvements.runner \
      ${DATA_DIR} \
      --datasets dentate_gyrus \
      --stream-umap-neighbors 30 \
      --stream-umap-min-dist 0.05 \
      --stream-umap-spread 1.5 \
      --output-dir ${OUTPUT_DIR} \
      --warmup-epochs 120 \
      --total-epochs 600 \
      --batch-size 256 \
      --latent-dim 10 \
      --hidden-dim 256 \
      --gnn-hidden-dim 128 \
      --gnn-dropout 0.1 \
      --num-workers 0 \
      --gnn-neighbor-source both \
      --gnn-attention \
      --gnn-gate \
      --enable-gnn-latent \
      --velocity-laplacian-weight 0.05 \
      --velocity-angle-weight 0.09 \
      --stream-embed umap \
      --enable-transformer-refinement \
      --transformer-epochs 20 \
      --transformer-hidden-dim 256 \
      --transformer-layers 3 \
      --transformer-heads 8 \
      --transformer-dropout 0.1 \
      --transformer-batch-size 128 \
      --transformer-learning-rate 1e-3 \
      --transformer-weight-smooth 0.4 \
      --transformer-weight-direction 0.7 \
      --transformer-weight-celltype 0.3 \
      --transformer-weight-celltype-dir 0.35 \
      --transformer-weight-celltype-mag 0.0 \
      --transformer-celltype-penalty both \
      --transformer-aux-cluster-loss-weight 0.5 \
      --transformer-weight-distill 1.0 \
      --transformer-distill-start 1.0 \
      --transformer-distill-end 0.3 \
      --transformer-max-neighbors 8 \
      --skip-preprocess \
      --checkpoint-dir ${WORKSPACE}/checkpoints \
      --use-wandb \
      --wandb-project \"RNA-Velocity\" \
      --wandb-run-group \"pancreas_transformer\" \
      --plot-results \
      $( [[ "${SAVE_FIGURES_LOCALLY}" == "0" ]] && echo "--disable-local-figures" )
  "
