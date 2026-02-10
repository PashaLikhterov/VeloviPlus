#!/usr/bin/env bash

set -euo pipefail

JOB_NAME="${JOB_NAME:-tivelo-standalone}"
WORKSPACE="${WORKSPACE:-/gpfs0/bgu-ofircohen/users/likhtepi/proj/scvi-tools}"
SCVELO_LOADER="${SCVELO_LOADER:-pancreas}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE}/results/tivelo}"
CONDA_ENV="${CONDA_ENV:-/gpfs0/bgu-ofircohen/users/likhtepi/conda_dirs/env/velovi-gnn}"
DATASET_NAME="${DATASET_NAME:-${SCVELO_LOADER}}"
SPLICED_LAYER="${SPLICED_LAYER:-Ms}"
UNSPLICED_LAYER="${UNSPLICED_LAYER:-Mu}"
GROUP_KEY="${GROUP_KEY:-cell_type}"
EMBED_KEY="${EMBED_KEY:-X_umap}"

runai-bgu submit cmd \
  -n "${JOB_NAME//_/-}" \
  -c 16 \
  -m 32G \
  -g 1 \
  --conda "${CONDA_ENV}" \
  --working-dir "${WORKSPACE}" \
  -- "
    python -m scvi.experimental.velovi_improvements.run_tivelo \
      --output-dir ${OUTPUT_DIR} \
      --dataset-name ${DATASET_NAME} \
      --scvelo-loader ${SCVELO_LOADER} \
      --spliced-layer ${SPLICED_LAYER} \
      --unspliced-layer ${UNSPLICED_LAYER} \
      --group-key ${GROUP_KEY} \
      --embedding-key ${EMBED_KEY} \
      --resolution 0.6 \
      --njobs -1 \
      --start-mode stochastic \
      --rev-stat mean \
      --threshold 0.1 \
      --threshold-trans 1.0 \
      --t1 0.1 \
      --t2 1.0 \
      --loss-fun mse \
      --alpha-1 1.0 \
      --alpha-2 0.1 \
      --batch-size 1024 \
      --n-epochs 100 \
      --filter-genes \
      --min-shared-counts 30 \
      --n-top-genes 2000 \
      --n-pcs 30 \
      --n-neighbors 30 \
      --no-show-fig
  "
