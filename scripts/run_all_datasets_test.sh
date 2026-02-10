#!/usr/bin/env bash

set -euo pipefail

JOB_NAME="${JOB_NAME:-velovi-test-all}"
WORKSPACE="${WORKSPACE:-/gpfs0/bgu-ofircohen/users/likhtepi/proj/scvi-tools}"
DATA_DIR="${DATA_DIR:-/gpfs0/bgu-ofircohen/users/likhtepi/proj/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE}/results/velovi_test_all}"
CONDA_ENV="${CONDA_ENV:-/gpfs0/bgu-ofircohen/users/likhtepi/conda_dirs/env/velovi-gnn}"
SAVE_FIGURES_LOCALLY="${SAVE_FIGURES_LOCALLY:-1}"
# Short list for smoke test; extend as needed
DATASETS=(
  pancreas_endocrinogenesis
  dentate_gyrus
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

RUN_CMD="python -m scvi.experimental.velovi_improvements.runner \
  ${DATA_DIR} \
  --datasets ${DATASETS[*]} \
  --output-dir ${OUTPUT_DIR} \
  --warmup-epochs 10 \
  --total-epochs 40 \
  --batch-size 128 \
  --gnn-epochs 40 \
  --gnn-batch-size 128 \
  --latent-dim 10 \
  --hidden-dim 128 \
  --gnn-hidden-dim 64 \
  --gnn-dropout 0.1 \
  --num-workers 0 \
  --gnn-neighbor-source both \
  --gnn-attention \
  --gnn-gate \
  --gnn-continuity-weight 0.15 \
  --gnn-continuity-horizon 1.0 \
  --enable-gnn-latent \
  --velocity-laplacian-weight 0.02 \
  --velocity-angle-weight 0.01 \
  --stream-embed umap \
  --use-paga-for-refinements \
  --enable-transformer-refinement \
  --transformer-epochs 10 \
  --transformer-hidden-dim 128 \
  --transformer-layers 2 \
  --transformer-heads 4 \
  --transformer-dropout 0.1 \
  --transformer-batch-size 128 \
  --transformer-learning-rate 1e-3 \
  --transformer-weight-smooth 0.1 \
  --transformer-weight-direction 0.3 \
  --transformer-weight-smooth-same 0.15 \
  --transformer-weight-boundary-align 0.25 \
  --transformer-weight-boundary-contrast 0.05 \
  --transformer-aux-cluster-loss-weight 0.1 \
  --transformer-max-neighbors 12 \
  --skip-preprocess \
  --checkpoint-dir ${WORKSPACE}/checkpoints \
  --disable-scvelo-dynamic \
  --plot-results \
  --use-wandb \
  --wandb-project \"RNA-Velocity\" \
  --wandb-run-group \"test_all_boundary\" \
  $( [[ \"${SAVE_FIGURES_LOCALLY}\" == \"0\" ]] && echo \"--disable-local-figures\" )"

runai-bgu submit cmd \
  -n "${JOB_NAME//_/-}" \
  -c 64 \
  -m 120G \
  -g 1 \
  --conda "${CONDA_ENV}" \
  --working-dir "${WORKSPACE}" \
  -- "${RUN_CMD}"
