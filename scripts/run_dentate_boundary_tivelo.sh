#!/usr/bin/env bash

set -euo pipefail

JOB_NAME="${JOB_NAME:-velovi-dentate-boundary-tivelo}" 
WORKSPACE="${WORKSPACE:-/gpfs0/bgu-ofircohen/users/likhtepi/proj/scvi-tools}"
DATA_DIR="${DATA_DIR:-/gpfs0/bgu-ofircohen/users/likhtepi/proj/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE}/results/velovi_dentate_boundary_tivelo}"
CONDA_ENV="${CONDA_ENV:-/gpfs0/bgu-ofircohen/users/likhtepi/conda_dirs/env/velovi-gnn}"
SAVE_FIGURES_LOCALLY="${SAVE_FIGURES_LOCALLY:-1}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-120}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-128}"
GNN_EPOCHS="${GNN_EPOCHS:-${TOTAL_EPOCHS}}"
GNN_BATCH_SIZE="${GNN_BATCH_SIZE:-${BATCH_SIZE}}"

DATASETS=(
  dentate_gyrus
  dentategyrus_lamanno
)

for DATASET in "${DATASETS[@]}"; do
  runai-bgu submit cmd \
    -n "${JOB_NAME//_/-}-${DATASET//_/-}" \
    -c 64 \
    -m 120G \
    -g 1 \
    --conda "${CONDA_ENV}" \
    --working-dir "${WORKSPACE}" \
    -- "
      python -m scvi.experimental.velovi_improvements.runner \
        ${DATA_DIR} \
        --datasets ${DATASET} \
        --output-dir ${OUTPUT_DIR}/${DATASET} \
        --warmup-epochs ${WARMUP_EPOCHS} \
        --total-epochs ${TOTAL_EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --gnn-epochs ${GNN_EPOCHS} \
        --gnn-batch-size ${GNN_BATCH_SIZE} \
        --latent-dim 10 \
        --hidden-dim 128 \
        --gnn-hidden-dim 96 \
        --gnn-dropout 0.1 \
        --gnn-continuity-weight 0.2 \
        --gnn-continuity-horizon 1.0 \
        --num-workers 0 \
        --gnn-neighbor-source both \
        --gnn-attention \
        --gnn-gate \
        --velocity-laplacian-weight 0.05 \
        --velocity-angle-weight 0.02 \
        --stream-embed umap \
        --enable-transformer-refinement \
        --transformer-epochs 15 \
        --transformer-hidden-dim 192 \
        --transformer-layers 3 \
        --transformer-heads 6 \
        --transformer-dropout 0.1 \
        --transformer-batch-size 128 \
        --transformer-learning-rate 8e-4 \
        --transformer-weight-smooth 1e-2 \
        --transformer-weight-direction 0.4 \
        --transformer-weight-smooth-same 0.4 \
        --transformer-weight-boundary-align 0.6 \
        --transformer-weight-boundary-contrast 0.05 \
        --transformer-aux-cluster-loss-weight 0.15 \
        --transformer-max-neighbors 10 \
        --skip-preprocess \
        --checkpoint-dir ${WORKSPACE}/checkpoints \
        --disable-scvelo-dynamic \
        --use-paga-for-refinements \
        --plot-results \
        --use-wandb \
        --wandb-project \"RNA-Velocity\" \
        --wandb-run-group \"dentate_boundary_tivelo\" \
        $( [[ \"${SAVE_FIGURES_LOCALLY}\" == \"0\" ]] && echo \"--disable-local-figures\" )
    "
done
