#!/usr/bin/env bash

set -euo pipefail

JOB_NAME="velovi-dentate"
SAVE_FIGURES_LOCALLY="${SAVE_FIGURES_LOCALLY:-1}"
WORKSPACE="/gpfs0/bgu-ofircohen/users/likhtepi/proj/scvi-tools"
DATA_DIR="/gpfs0/bgu-ofircohen/users/likhtepi/proj/velocity/scvi-tools/velovi_datasets"
OUTPUT_DIR="${WORKSPACE}/results/velovi_dentate"

CONDA_ENV="/gpfs0/bgu-ofircohen/users/likhtepi/conda_dirs/env/velovi-gnn"

runai-bgu submit cmd \
  -n "${JOB_NAME//_/-}" \
  -c 32 \
  -m 32G \
  -g 1 \
  --conda ${CONDA_ENV} \
  --working-dir "${WORKSPACE}" \
  -- "
    python -m scvi.experimental.velovi_improvements.runner \
      ${DATA_DIR} \
      --datasets dentate_gyrus \
      --output-dir ${OUTPUT_DIR} \
      --warmup-epochs 120 \
      --total-epochs 400 \
      --batch-size 256 \
      --latent-dim 10 \
      --hidden-dim 256 \
      --gnn-hidden-dim 128 \
      --gnn-dropout 0.1 \
      --num-workers 0 \
      --enable-gnn-latent \
      --plot-results \
          $( [[ "${SAVE_FIGURES_LOCALLY}" == "0" ]] && echo "--disable-local-figures" )
  "
