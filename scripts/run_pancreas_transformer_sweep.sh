#!/usr/bin/env bash

set -euo pipefail

JOB_PREFIX="${JOB_PREFIX:-velovi-pancreas-transformer}"
SAVE_FIGURES_LOCALLY="${SAVE_FIGURES_LOCALLY:-1}"
WORKSPACE="${WORKSPACE:-/gpfs0/bgu-ofircohen/users/likhtepi/proj/scvi-tools}"
DATA_DIR="${DATA_DIR:-/gpfs0/bgu-ofircohen/users/likhtepi/proj/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE}/results/velovi_pancreas_transformer_sweep}"
CONDA_ENV="${CONDA_ENV:-/gpfs0/bgu-ofircohen/users/likhtepi/conda_dirs/env/velovi-gnn}"
TIVELO_PRIOR_STRENGTH="${TIVELO_PRIOR_STRENGTH:-0.35}"
SCVELO_N_JOBS="${SCVELO_N_JOBS:-32}"
LATENT_METRIC_NN="${LATENT_METRIC_NN:-30}"

SMOOTH_SAME_WEIGHTS=(${SMOOTH_SAME_WEIGHTS:-0.0 0.1 0.2 0.3})
BOUNDARY_ALIGN_WEIGHTS=(${BOUNDARY_ALIGN_WEIGHTS:-0.0 0.3 0.6})
BOUNDARY_CONTRAST_WEIGHTS=(${BOUNDARY_CONTRAST_WEIGHTS:-0.0 0.05 0.1})

WEIGHT_SMOOTH="${WEIGHT_SMOOTH:-0.1}"
WEIGHT_DIRECTION="${WEIGHT_DIRECTION:-0.7}"
WEIGHT_CELLTYPE_DIR="${WEIGHT_CELLTYPE_DIR:-0.35}"
WEIGHT_ALIGNMENT="${WEIGHT_ALIGNMENT:-1.0}"
WEIGHT_SUPERVISED="${WEIGHT_SUPERVISED:-0.5}"
WEIGHT_CELLTYPE="${WEIGHT_CELLTYPE:-0.0}"
WEIGHT_CELLTYPE_MAG="${WEIGHT_CELLTYPE_MAG:-0.0}"
TRANSFORMER_HEADS="${TRANSFORMER_HEADS:-8}"
TRANSFORMER_LAYERS="${TRANSFORMER_LAYERS:-3}"
TRANSFORMER_HIDDEN="${TRANSFORMER_HIDDEN:-256}"
TRANSFORMER_EPOCHS="${TRANSFORMER_EPOCHS:-15}"
TRANSFORMER_BATCH="${TRANSFORMER_BATCH:-128}"

run_job () {
  local smooth_same="$1"
  local boundary_align="$2"
  local boundary_contrast="$3"

  local tag="ss${smooth_same}_ba${boundary_align}_bc${boundary_contrast}"
  local job_name="${JOB_PREFIX}-${tag}"
  local job_safe="${job_name//_/-}"
  local output_dir="${OUTPUT_ROOT}/${tag}"
  local run_group="pancreas_transformer_${tag}"

  echo "[VELOVI][SWEEP] Submitting ${job_name}"

  runai-bgu submit cmd \
    -n "${job_safe}" \
    -c 128 \
    -m 256G \
    -g 1 \
    --conda "${CONDA_ENV}" \
    --working-dir "${WORKSPACE}" \
    -- "
      python -m scvi.experimental.velovi_improvements.runner \
        ${DATA_DIR} \
        --datasets pancreas_endocrinogenesis \
        --output-dir ${output_dir} \
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
        --transformer-epochs ${TRANSFORMER_EPOCHS} \
        --transformer-hidden-dim ${TRANSFORMER_HIDDEN} \
        --transformer-layers ${TRANSFORMER_LAYERS} \
        --transformer-heads ${TRANSFORMER_HEADS} \
        --transformer-dropout 0.1 \
        --transformer-batch-size ${TRANSFORMER_BATCH} \
        --transformer-learning-rate 1e-3 \
        --transformer-weight-smooth ${WEIGHT_SMOOTH} \
        --transformer-weight-smooth-same ${smooth_same} \
        --transformer-weight-boundary-align ${boundary_align} \
        --transformer-weight-boundary-contrast ${boundary_contrast} \
        --transformer-weight-direction ${WEIGHT_DIRECTION} \
        --transformer-weight-celltype ${WEIGHT_CELLTYPE} \
        --transformer-weight-celltype-dir ${WEIGHT_CELLTYPE_DIR} \
        --transformer-weight-celltype-mag ${WEIGHT_CELLTYPE_MAG} \
        --transformer-weight-alignment ${WEIGHT_ALIGNMENT} \
        --transformer-weight-supervised ${WEIGHT_SUPERVISED} \
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
        --use-wandb \
        --wandb-project \"RNA-Velocity\" \
        --wandb-run-group \"${run_group}\" \
        --plot-results \
          $( [[ "${SAVE_FIGURES_LOCALLY}" == "0" ]] && echo "--disable-local-figures" )
    "
}

for smooth_same in "${SMOOTH_SAME_WEIGHTS[@]}"; do
  for boundary_align in "${BOUNDARY_ALIGN_WEIGHTS[@]}"; do
    for boundary_contrast in "${BOUNDARY_CONTRAST_WEIGHTS[@]}"; do
      run_job "${smooth_same}" "${boundary_align}" "${boundary_contrast}"
    done
  done
done
