#!/usr/bin/env bash

set -euo pipefail

JOB_NAME="${JOB_NAME:-velovi-erythroid-transformer}"
SAVE_FIGURES_LOCALLY="${SAVE_FIGURES_LOCALLY:-1}"
WORKSPACE="${WORKSPACE:-/gpfs0/bgu-ofircohen/users/likhtepi/proj/scvi-tools}"
DATA_DIR="${DATA_DIR:-/gpfs0/bgu-ofircohen/users/likhtepi/proj/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE}/results/velovi_erythroid_transformer}"
CONDA_ENV="${CONDA_ENV:-/gpfs0/bgu-ofircohen/users/likhtepi/conda_dirs/env/velovi-gnn}"
TIVELO_PRIOR_STRENGTH="${TIVELO_PRIOR_STRENGTH:-0.35}"
SCVELO_N_JOBS="${SCVELO_N_JOBS:-32}"
LATENT_METRIC_NN="${LATENT_METRIC_NN:-15}"

JOB_NAME_SAFE="${JOB_NAME//_/-}"
runai-bgu submit cmd \
  -n "${JOB_NAME_SAFE}" \
  -c 128 \
  -m 256G \
  -g 2 \
  --conda "${CONDA_ENV}" \
  --working-dir "${WORKSPACE}" \
  -- "
    python -m scvi.experimental.velovi_improvements.runner \
      ${DATA_DIR} \
      --datasets mouse_bone_marrow \
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
      --transformer-use-tivelo \
      --transformer-epochs 15 \
      --transformer-hidden-dim 256 \
      --transformer-layers 3 \
      --transformer-heads 8 \
      --transformer-dropout 0.1 \
      --transformer-batch-size 128 \
      --transformer-learning-rate 1e-3 \
      --transformer-weight-smooth 0.2 \
      --transformer-weight-direction 0.7 \
      --transformer-weight-celltype 0.0 \
      --transformer-weight-celltype-dir 0.35 \
      --transformer-weight-celltype-mag 0.0 \
      --transformer-weight-alignment 1.0 \
      --transformer-weight-supervised 0.5 \
      --transformer-celltype-penalty cosine \
      --transformer-aux-cluster-loss-weight 0.2 \
      --transformer-neighbor-max-distance 4.0 \
      --transformer-max-neighbors 8 \
      --scvelo-n-jobs ${SCVELO_N_JOBS} \
      --tivelo-prior-strength ${TIVELO_PRIOR_STRENGTH} \
      --transformer-tivelo-threshold 0.1 \
      --transformer-tivelo-threshold-trans 1.0 \
      --latent-metric-n-neighbors ${LATENT_METRIC_NN} \
      --skip-preprocess \
      --checkpoint-dir ${WORKSPACE}/checkpoints \
      --plot-results \
      $( [[ "${SAVE_FIGURES_LOCALLY}" == "0" ]] && echo "--disable-local-figures" )
  "
