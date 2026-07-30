#!/bin/bash
#SBATCH --job-name=prep_02_render
#SBATCH --output=%x_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-127

set -euo pipefail

module load apptainer

REPO_DIR="$HOME/SymTRELLIS"
SIF="$HOME/symtrellis.sif"

: "${DATA_DIR:?Set DATA_DIR}"

NUM_VIEWS="${NUM_VIEWS:-225}"
NUM_WORKERS="${NUM_WORKERS:-1}"
VERBOSE="${VERBOSE:-0}"
RECOMPUTE_FINISHED="${RECOMPUTE_FINISHED:-0}"

RANK="$SLURM_ARRAY_TASK_ID"
WORLD_SIZE="${WORLD_SIZE:-$((SLURM_ARRAY_TASK_MAX + 1))}"

RUN_ARGS=()
if [ "$VERBOSE" = "1" ]; then
    RUN_ARGS+=(--verbose)
fi
if [ "$RECOMPUTE_FINISHED" = "1" ]; then
    RUN_ARGS+=(--recompute-finished)
fi

mkdir -p "$DATA_DIR" "$HOME/.cache/cuda_compute_cache"
cd "$DATA_DIR"

apptainer exec --nv \
    --no-home \
    --no-mount cwd \
    --bind "$REPO_DIR/preprocess:/workspace/SymTRELLIS/preprocess:ro" \
    --bind "$HOME/.cache:$HOME/.cache" \
    --bind "$DATA_DIR:$DATA_DIR" \
    --env "CUDA_CACHE_PATH=$HOME/.cache/cuda_compute_cache" \
    --env "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt" \
    --pwd /workspace/SymTRELLIS \
    "$SIF" \
    python -m preprocess.trellis1.render \
    --dataset-dir "$DATA_DIR" \
    --num-views "$NUM_VIEWS" \
    --rank "$RANK" \
    --world-size "$WORLD_SIZE" \
    --num-workers "$NUM_WORKERS" \
    "${RUN_ARGS[@]}"
