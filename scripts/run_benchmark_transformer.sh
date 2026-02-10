#!/usr/bin/env bash

set -euo pipefail

JOB_NAME="${JOB_NAME:-velovi-benchmark}"
SAVE_FIGURES_LOCALLY="${SAVE_FIGURES_LOCALLY:-1}"
WORKSPACE="${WORKSPACE:-/gpfs0/bgu-ofircohen/users/likhtepi/proj/scvi-tools}"
DATA_DIR="${DATA_DIR:-${SCVELO_CACHE_DIR:-/gpfs0/bgu-ofircohen/users/likhtepi/.cache/scvelo}}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE}/results/velovi_benchmark}"
CONDA_ENV="${CONDA_ENV:-/gpfs0/bgu-ofircohen/users/likhtepi/conda_dirs/env/velovi-gnn}"
DATASETS="${DATASETS:-pancreas_endocrinogenesis}"

JOB_NAME_SAFE="${JOB_NAME//_/-}"
runai-bgu submit cmd \
  -n "${JOB_NAME_SAFE}" \
  -c "${CPUS:-128}" \
  -m "${MEMORY:-256G}" \
  -g "${GPUS:-2}" \
  --conda "${CONDA_ENV}" \
  --working-dir "${WORKSPACE}" \
  -- "
    python -m scvi.experimental.velovi_improvements.runner \
      ${DATA_DIR} \
      --datasets ${DATASETS} \
      --output-dir ${OUTPUT_DIR} \
      --warmup-epochs \${WARMUP_EPOCHS:-120} \
      --total-epochs \${TOTAL_EPOCHS:-400} \
      --batch-size \${BATCH_SIZE:-256} \
      --latent-dim \${LATENT_DIM:-10} \
      --hidden-dim \${HIDDEN_DIM:-256} \
      --gnn-hidden-dim \${GNN_HIDDEN_DIM:-128} \
      --gnn-dropout \${GNN_DROPOUT:-0.1} \
      --num-workers \${NUM_WORKERS:-0} \
      --gnn-neighbor-source \${GNN_NEIGHBOR_SOURCE:-both} \
      --gnn-attention \
      --gnn-gate \
      --enable-gnn-latent \
      --velocity-laplacian-weight \${VELOCITY_LAPLACIAN_WEIGHT:-0.05} \
      --velocity-angle-weight \${VELOCITY_ANGLE_WEIGHT:-0.02} \
      --stream-embed umap \
      --enable-transformer-refinement \
      --transformer-epochs \${TRANSFORMER_EPOCHS:-15} \
      --transformer-hidden-dim \${TRANSFORMER_HIDDEN_DIM:-256} \
      --transformer-layers \${TRANSFORMER_LAYERS:-3} \
      --transformer-heads \${TRANSFORMER_HEADS:-8} \
      --transformer-dropout \${TRANSFORMER_DROPOUT:-0.1} \
      --transformer-batch-size \${TRANSFORMER_BATCH_SIZE:-128} \
      --transformer-learning-rate \${TRANSFORMER_LR:-1e-3} \
      --transformer-weight-smooth \${TRANSFORMER_WEIGHT_SMOOTH:-0.2} \
      --transformer-weight-direction \${TRANSFORMER_WEIGHT_DIRECTION:-0.7} \
      --transformer-weight-celltype \${TRANSFORMER_WEIGHT_CELLTYPE:-0.0} \
      --transformer-weight-celltype-dir \${TRANSFORMER_WEIGHT_CELLTYPE_DIR:-0.35} \
      --transformer-weight-celltype-mag \${TRANSFORMER_WEIGHT_CELLTYPE_MAG:-0.0} \
      --transformer-weight-alignment \${TRANSFORMER_WEIGHT_ALIGNMENT:-1.0} \
      --transformer-weight-supervised \${TRANSFORMER_WEIGHT_SUPERVISED:-0.5} \
      --transformer-celltype-penalty \${TRANSFORMER_CELLTYPE_PENALTY:-cosine} \
      --transformer-aux-cluster-loss-weight \${TRANSFORMER_AUX_CLUSTER_LOSS_WEIGHT:-0.2} \
      --transformer-neighbor-max-distance \${TRANSFORMER_NEIGHBOR_MAX_DISTANCE:-4.0} \
      --transformer-max-neighbors \${TRANSFORMER_MAX_NEIGHBORS:-8} \
      --transformer-use-tivelo \
      --transformer-tivelo-threshold \${TRANSFORMER_TIVELO_THRESHOLD:-0.1} \
      --transformer-tivelo-threshold-trans \${TRANSFORMER_TIVELO_THRESHOLD_TRANS:-1.0} \
      --checkpoint-dir \${CHECKPOINT_DIR:-${WORKSPACE}/checkpoints} \
      --use-wandb \
      --wandb-project \${WANDB_PROJECT:-RNA-Velocity} \
      --wandb-run-group \${WANDB_RUN_GROUP:-benchmark} \
      --plot-results \
      $( [[ "${SAVE_FIGURES_LOCALLY}" == "0" ]] && echo "--disable-local-figures" )
  "
