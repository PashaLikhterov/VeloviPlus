#!/usr/bin/env bash

set -euo pipefail

JOB_PREFIX="${JOB_PREFIX:-velovi-pancreas-gnn}"
SAVE_FIGURES_LOCALLY="${SAVE_FIGURES_LOCALLY:-1}"
WORKSPACE="${WORKSPACE:-/gpfs0/bgu-ofircohen/users/likhtepi/proj/scvi-tools}"
DATA_DIR="${DATA_DIR:-/gpfs0/bgu-ofircohen/users/likhtepi/proj/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE}/results/velovi_pancreas_gnn_sweep}"
CONDA_ENV="${CONDA_ENV:-/gpfs0/bgu-ofircohen/users/likhtepi/conda_dirs/env/velovi-gnn}"
SCVELO_N_JOBS="${SCVELO_N_JOBS:-32}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-120}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-256}"
GNN_EPOCHS="${GNN_EPOCHS:-${TOTAL_EPOCHS}}"
GNN_BATCH_SIZE="${GNN_BATCH_SIZE:-${BATCH_SIZE}}"

maybe_no_local=()
if [[ "${SAVE_FIGURES_LOCALLY}" == "0" ]]; then
  maybe_no_local+=("--disable-local-figures")
fi

BASE_ARGS=(
  --datasets pancreas_endocrinogenesis
  --warmup-epochs ${WARMUP_EPOCHS}
  --total-epochs ${TOTAL_EPOCHS}
  --batch-size ${BATCH_SIZE}
  --gnn-epochs ${GNN_EPOCHS}
  --gnn-batch-size ${GNN_BATCH_SIZE}
  --latent-dim 30
  --hidden-dim 256
  --gnn-hidden-dim 128
  --gnn-dropout 0.1
  --num-workers 0
  --stream-embed umap
  --checkpoint-dir "${WORKSPACE}/checkpoints"
  --use-wandb
  --wandb-project "RNA-Velocity"
  --plot-results
  --scvelo-n-jobs "${SCVELO_N_JOBS}"
)

CONFIGS=(
  "tag=expr_base source=expression attn=0 gate=0 diff=0 residual=1 lap=0.0 angle=0.0"
  "tag=expr_attn source=expression attn=1 gate=1 diff=0 residual=1 lap=0.0 angle=0.0"
  "tag=expr_diff source=expression attn=1 gate=1 diff=1 residual=1 lap=0.05 angle=0.02"
  "tag=latent_gate source=latent attn=0 gate=1 diff=0 residual=0 lap=0.05 angle=0.0"
  "tag=both_full source=both attn=1 gate=1 diff=1 residual=1 lap=0.05 angle=0.02"
)

submit_config() {
  local tag="$1"
  local source="$2"
  local attn="$3"
  local gate="$4"
  local diff="$5"
  local residual="$6"
  local lap="$7"
  local angle="$8"

  local job_name="${JOB_PREFIX}-${tag}"
  local job_safe="${job_name//_/-}"
  local output_dir="${OUTPUT_ROOT}/${tag}"
  local run_group="pancreas_gnn_${tag}"

  mkdir -p "${output_dir}"

  echo "[VELOVI][GNN-SWEEP] ${job_name} (source=${source}, attn=${attn}, gate=${gate}, diff=${diff}, residual=${residual}, lap=${lap}, angle=${angle})"

  # Build optional flags
  GNN_FLAGS=("--gnn-neighbor-source" "${source}")
  if [[ "${attn}" == "1" ]]; then
    GNN_FLAGS+=("--gnn-attention")
  fi
  if [[ "${gate}" == "1" ]]; then
    GNN_FLAGS+=("--gnn-gate")
  fi
  if [[ "${diff}" == "0" ]]; then
    GNN_FLAGS+=("--disable-gnn-differences")
  fi
  if [[ "${residual}" == "0" ]]; then
    GNN_FLAGS+=("--disable-gnn-residual")
  fi
  GNN_FLAGS+=("--velocity-laplacian-weight" "${lap}")
  GNN_FLAGS+=("--velocity-angle-weight" "${angle}")

  runai-bgu submit cmd \
    -n "${job_safe}" \
    -c 64 \
    -m 192G \
    -g 1 \
    --conda "${CONDA_ENV}" \
    --working-dir "${WORKSPACE}" \
    -- "
      python -m scvi.experimental.velovi_improvements.runner \
        ${DATA_DIR} \
        --output-dir ${output_dir} \
        --wandb-run-group \"${run_group}\" \
        ${BASE_ARGS[*]} \
        ${maybe_no_local[*]} \
        ${GNN_FLAGS[*]}
    "
}

for cfg in "${CONFIGS[@]}"; do
  eval "${cfg}"
  submit_config "${tag}" "${source}" "${attn}" "${gate}" "${diff}" "${residual}" "${lap}" "${angle}"
done
