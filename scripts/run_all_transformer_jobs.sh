#!/usr/bin/env bash

set -euo pipefail

WORKSPACE="${WORKSPACE:-/gpfs0/bgu-ofircohen/users/likhtepi/proj/scvi-tools}"
DATA_DIR="${DATA_DIR:-${SCVELO_CACHE_DIR:-/gpfs0/bgu-ofircohen/users/likhtepi/.cache/scvelo}}"
CONDA_ENV="${CONDA_ENV:-/gpfs0/bgu-ofircohen/users/likhtepi/conda_dirs/env/velovi-gnn}"
SAVE_FIGURES_LOCALLY="${SAVE_FIGURES_LOCALLY:-1}"

declare -A DATASET_LABELS=(
  ["pancreas_endocrinogenesis"]="pancreas"
  ["dentate_gyrus"]="dentate"
  ["dentategyrus_lamanno"]="dentate_lamanno"
  ["mouse_bone_marrow"]="mousebm"
  ["gastrulation"]="gastrulation"
  ["gastrulation_erythroid"]="gastrulation_erythroid"
  ["pbmc68k"]="pbmc"
)

ORDER=(
  "pancreas_endocrinogenesis"
  "dentate_gyrus"
  "dentategyrus_lamanno"
  "mouse_bone_marrow"
  "gastrulation"
  "gastrulation_erythroid"
  "pbmc68k"
)

for DATASET in "${ORDER[@]}"; do
  LABEL="${DATASET_LABELS[$DATASET]}"
  JOB_SAFE_LABEL="${LABEL//_/-}"
  JOB_NAME="${JOB_NAME_PREFIX:-velovi}-${JOB_SAFE_LABEL}-transformer"
  OUTPUT_DIR="${WORKSPACE}/results/velovi_${LABEL}_transformer"

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
        --datasets ${DATASET} \
        --output-dir ${OUTPUT_DIR} \
        --warmup-epochs 120 \
        --total-epochs 400 \
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
        --velocity-angle-weight 0.02 \
        --stream-embed umap \
        --enable-transformer-refinement \
        --transformer-epochs 15 \
        --transformer-hidden-dim 256 \
        --transformer-layers 3 \
        --transformer-heads 8 \
        --transformer-dropout 0.1 \
        --transformer-batch-size 128 \
        --transformer-learning-rate 1e-3 \
        --transformer-weight-smooth 0.2 \
        --transformer-weight-direction 0.2 \
        --transformer-weight-celltype 0.25 \
        --transformer-max-neighbors 16 \
        --skip-preprocess \
        --checkpoint-dir ${WORKSPACE}/checkpoints \
        --use-wandb \
        --wandb-project \"RNA-Velocity\" \
        --wandb-run-group \"${LABEL}_transformer\" \
        --plot-results \
        $( [[ \"${SAVE_FIGURES_LOCALLY}\" == \"0\" ]] && echo \"--disable-local-figures\" )
    "
done
