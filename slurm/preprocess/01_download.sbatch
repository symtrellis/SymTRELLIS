#!/bin/bash
#SBATCH --job-name=prep_01_download
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=4:00:00

set -euo pipefail

module load apptainer

REPO_DIR="$HOME/SymTRELLIS"
SIF="$HOME/symtrellis.sif"

: "${DATA_DIR:?Set DATA_DIR}"

DATASET="${DATASET:-Toys4K}"
NUM_WORKERS="${NUM_WORKERS:-8}"
RECOMPUTE_FINISHED="${RECOMPUTE_FINISHED:-0}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

ARRAY_MAX="${SLURM_ARRAY_TASK_MAX:-0}"
RANK="${SLURM_ARRAY_TASK_ID:-0}"
WORLD_SIZE="${WORLD_SIZE:-$((ARRAY_MAX + 1))}"

RECOMPUTE_ARGS=()
if [ "$RECOMPUTE_FINISHED" = "1" ]; then
    RECOMPUTE_ARGS+=(--recompute-finished)
fi

mkdir -p "$DATA_DIR" "$HOME/.cache/huggingface"
cd "$DATA_DIR"

apptainer exec \
    --no-home \
    --no-mount cwd \
    --bind "$REPO_DIR/preprocess:/workspace/SymTRELLIS/preprocess:ro" \
    --bind "$HOME/.cache:$HOME/.cache" \
    --bind "$DATA_DIR:$DATA_DIR" \
    --env "HF_HOME=$HOME/.cache/huggingface" \
    --env "HF_HUB_CACHE=$HOME/.cache/huggingface/hub" \
    --env "HF_HUB_OFFLINE=$HF_HUB_OFFLINE" \
    --env "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt" \
    --pwd /workspace/SymTRELLIS \
    "$SIF" \
    python -m preprocess.download "$DATASET" \
    --dataset-dir "$DATA_DIR" \
    --rank "$RANK" \
    --world-size "$WORLD_SIZE" \
    --num-workers "$NUM_WORKERS" \
    "${RECOMPUTE_ARGS[@]}"
