#!/usr/bin/env bash

set -euo pipefail

JOB_PREFIX="${JOB_PREFIX:-velovi-pancreas-gnnlat}"
SAVE_FIGURES_LOCALLY="${SAVE_FIGURES_LOCALLY:-1}"
WORKSPACE="${WORKSPACE:-/gpfs0/bgu-ofircohen/users/likhtepi/proj/scvi-tools}"
DATA_DIR="${DATA_DIR:-/gpfs0/bgu-ofircohen/users/likhtepi/proj/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE}/results/velovi_pancreas_gnn_latent_sweep}"
CONDA_ENV="${CONDA_ENV:-/gpfs0/bgu-ofircohen/users/likhtepi/conda_dirs/env/velovi-gnn}"
SCVELO_N_JOBS="${SCVELO_N_JOBS:-32}"

BASE_ARGS=(
  --datasets pancreas_endocrinogenesis
  --warmup-epochs 120
  --total-epochs 400
  --batch-size 256
  --latent-dim 10
  --hidden-dim 256
  --gnn-hidden-dim 128
  --gnn-dropout 0.1
  --num-workers 0
  --stream-embed umap
  --enable-gnn-latent
  --checkpoint-dir "${WORKSPACE}/checkpoints"
  --use-wandb
  --wandb-project "RNA-Velocity"
  --plot-results \
          $( [[ "${SAVE_FIGURES_LOCALLY}" == "0" ]] && echo "--disable-local-figures" )
  --scvelo-n-jobs "${SCVELO_N_JOBS}"
)

CONFIGS=(
  "tag=latent_only source=latent attn=0 gate=1 diff=0 residual=1 lap=0.05 angle=0.0 latnn=20"
  "tag=latent_attn source=latent attn=1 gate=1 diff=0 residual=1 lap=0.05 angle=0.02 latnn=30"
  "tag=both_lat30 source=both attn=1 gate=1 diff=1 residual=1 lap=0.05 angle=0.02 latnn=30"
  "tag=both_lat60 source=both attn=1 gate=1 diff=1 residual=1 lap=0.05 angle=0.02 latnn=60"
  "tag=expr_latmix source=expression attn=1 gate=0 diff=0 residual=1 lap=0.02 angle=0.0 latnn=25"
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
  local latnn="$9"

  local job_name="${JOB_PREFIX}-${tag}"
  local job_safe="${job_name//_/-}"
  local output_dir="${OUTPUT_ROOT}/${tag}"
  local run_group="pancreas_gnn_latent_${tag}"

  mkdir -p "${output_dir}"

  echo "[VELOVI][GNN-LATENT] ${job_name} (source=${source}, latnn=${latnn}, attn=${attn}, gate=${gate}, diff=${diff}, lap=${lap}, angle=${angle})"

  FLAGS=("--gnn-neighbor-source" "${source}" "--latent-metric-n-neighbors" "${latnn}")
  if [[ "${attn}" == "1" ]]; then FLAGS+=("--gnn-attention"); fi
  if [[ "${gate}" == "1" ]]; then FLAGS+=("--gnn-gate"); fi
  if [[ "${diff}" == "0" ]]; then FLAGS+=("--disable-gnn-differences"); fi
  if [[ "${residual}" == "0" ]]; then FLAGS+=("--disable-gnn-residual"); fi
  FLAGS+=("--velocity-laplacian-weight" "${lap}")
  FLAGS+=("--velocity-angle-weight" "${angle}")

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
        ${FLAGS[*]}
    "
}

for cfg in "${CONFIGS[@]}"; do
  eval "${cfg}"
  submit_config "${tag}" "${source}" "${attn}" "${gate}" "${diff}" "${residual}" "${lap}" "${angle}" "${latnn}"
done
