#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/gpfs0/bgu-ofircohen/users/likhtepi/proj/scvi-tools}"
CONDA_ENV="${CONDA_ENV:-/gpfs0/bgu-ofircohen/users/likhtepi/conda_dirs/env/velovi-gnn}"

# Set your sweep ID here (from step 1)
SWEEP_ID="${SWEEP_ID:-likhtepi-ben-gurion-university-of-the-negev/RNA-Velocity/mog6ze9g}"

JOB_NAME="${JOB_NAME:-velovi-gazit-sweep-agent}"
JOB_NAME_SAFE="${JOB_NAME//_/-}"

runai-bgu submit cmd \
  -n "${JOB_NAME_SAFE}" \
  -c 96 \
  -m 128G \
  -g 1 \
  --conda "${CONDA_ENV}" \
  --working-dir "${WORKSPACE}" \
  -- "
    wandb agent ${SWEEP_ID}
  "
