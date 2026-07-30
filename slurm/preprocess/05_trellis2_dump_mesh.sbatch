#!/bin/bash
#SBATCH --job-name=prep_05_dump_mesh
#SBATCH --output=%x_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --array=0-31

set -euo pipefail

module load apptainer

REPO_DIR="$HOME/SymTRELLIS"
SIF="$HOME/symtrellis.sif"

: "${DATA_DIR:?Set DATA_DIR}"

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

mkdir -p "$DATA_DIR" "$HOME/.cache"
cd "$DATA_DIR"

apptainer exec \
    --no-home \
    --no-mount cwd \
    --bind "$REPO_DIR/preprocess:/workspace/SymTRELLIS/preprocess:ro" \
    --bind "$HOME/.cache:$HOME/.cache" \
    --bind "$DATA_DIR:$DATA_DIR" \
    --env "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt" \
    --pwd /workspace/SymTRELLIS \
    "$SIF" \
    python -m preprocess.trellis2.dump_mesh \
    --dataset-dir "$DATA_DIR" \
    --rank "$RANK" \
    --world-size "$WORLD_SIZE" \
    --num-workers "$NUM_WORKERS" \
    "${RUN_ARGS[@]}"
