#!/usr/bin/env bash

set -euo pipefail

WORKSPACE="${WORKSPACE:-$(pwd)}"
DATA_DIR="${DATA_DIR:-/gpfs0/bgu-ofircohen/users/likhtepi/proj/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE}/results/velovi_test_all}" 
CONDA_ENV="${CONDA_ENV:-/gpfs0/bgu-ofircohen/users/likhtepi/conda_dirs/env/velovi-gnn}"
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

source "${CONDA_ENV}/bin/activate"
python -m scvi.experimental.velovi_improvements.runner \
  "${DATA_DIR}" \
  --datasets "${DATASETS[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --warmup-epochs 5 \
  --total-epochs 20 \
  --batch-size 128 \
  --latent-dim 10 \
  --hidden-dim 128 \
  --gnn-hidden-dim 64 \
  --gnn-dropout 0.1 \
  --num-workers 0 \
  --gnn-neighbor-source both \
  --gnn-attention \
  --gnn-gate \
  --velocity-laplacian-weight 0.01 \
  --stream-embed umap \
  --enable-transformer-refinement \
  --transformer-epochs 5 \
  --transformer-hidden-dim 128 \
  --transformer-heads 4 \
  --transformer-weight-smooth 0.05 \
  --transformer-weight-direction 0.2 \
  --transformer-weight-smooth-same 0.1 \
  --transformer-weight-boundary-align 0.2 \
  --transformer-weight-boundary-contrast 0.05 \
  --skip-preprocess \
  --checkpoint-dir "${WORKSPACE}/checkpoints" \
  --disable-scvelo-dynamic \
  --plot-results
