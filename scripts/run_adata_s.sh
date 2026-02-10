#!/usr/bin/env bash

set -euo pipefail

JOB_NAME="${JOB_NAME:-velovi-adata-s}"
SAVE_FIGURES_LOCALLY="${SAVE_FIGURES_LOCALLY:-1}"
WORKSPACE="${WORKSPACE:-/gpfs0/bgu-ofircohen/users/likhtepi/proj/scvi-tools}"
DATA_DIR="${DATA_DIR:-/gpfs0/bgu-ofircohen/users/likhtepi/files_adata}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE}/results/velovi_adata_s}"
CONDA_ENV="${CONDA_ENV:-/gpfs0/bgu-ofircohen/users/likhtepi/conda_dirs/env/velovi-gnn}"
SCVELO_N_JOBS="${SCVELO_N_JOBS:-32}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-150}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LATENT_METRIC_NN="${LATENT_METRIC_NN:-15}"
SMOOTH_SAME_WEIGHT="${SMOOTH_SAME_WEIGHT:-0.2e-2}"
BOUNDARY_ALIGN_WEIGHT="${BOUNDARY_ALIGN_WEIGHT:-0.3}"
BOUNDARY_CONTRAST_WEIGHT="${BOUNDARY_CONTRAST_WEIGHT:-0.05}"
RUN_GROUP="${RUN_GROUP:-runcustomdata}"

JOB_NAME_SAFE="${JOB_NAME//_/-}"
runai-bgu submit cmd \
  -n "${JOB_NAME_SAFE}" \
  -c 96 \
  -m 120G \
  -g 1 \
  --conda "${CONDA_ENV}" \
  --working-dir "${WORKSPACE}" \
  -- "
   python -m scvi.experimental.velovi_improvements.runner \
          ${DATA_DIR} \
          --datasets adata_S \
          --disable-checkpoints \
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
          --transformer-epochs 20 \
          --transformer-hidden-dim 256 \
          --transformer-layers 3 \
          --transformer-heads 8 \
          --transformer-dropout 0.1 \
          --transformer-batch-size 128 \
          --transformer-learning-rate 1e-3 \
          --transformer-weight-smooth 0.1e-2 \
          --transformer-weight-smooth-same ${SMOOTH_SAME_WEIGHT} \
          --transformer-weight-boundary-align ${BOUNDARY_ALIGN_WEIGHT} \
          --transformer-weight-boundary-contrast ${BOUNDARY_CONTRAST_WEIGHT} \
          --transformer-weight-direction 0.7 \
          --transformer-weight-celltype 0.0 \
          --transformer-weight-celltype-dir 0.35 \
          --transformer-weight-celltype-mag 0.0 \
          --transformer-weight-alignment 1.0 \
          --transformer-aux-cluster-loss-weight 0.2 \
          --transformer-neighbor-max-distance 4.0 \
          --transformer-max-neighbors 8 \
          --scvelo-n-jobs ${SCVELO_N_JOBS} \
          --latent-metric-n-neighbors ${LATENT_METRIC_NN} \
          --checkpoint-dir ${WORKSPACE}/checkpoints \
          --use-wandb \
          --wandb-project \"RNA-Velocity\" \
          --wandb-run-group \"${RUN_GROUP}\" \
          --plot-results \
          $( [[ \"${SAVE_FIGURES_LOCALLY}\" == \"0\" ]] && echo \"--disable-local-figures\" )
  "
