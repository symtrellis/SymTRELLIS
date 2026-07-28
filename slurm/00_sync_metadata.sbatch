#!/bin/bash
#SBATCH --job-name=eval_00_metadata
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=30:00

set -euo pipefail

module load apptainer

REPO_DIR="$HOME/SymTRELLIS"
DATA_DIR="$SCRATCH/eval_workspace_after_acceptance"
SIF="$HOME/symtrellis.sif"
FULL_CHECK="${FULL_CHECK:-0}"
FOLDER="${FOLDER:-}"

SYNC_ARGS=()
if [ "$FULL_CHECK" = "1" ]; then
    SYNC_ARGS+=(--full-check)
fi
if [ -n "$FOLDER" ]; then
    SYNC_ARGS+=(--folders "$FOLDER")
fi

cd "$DATA_DIR"

apptainer exec \
    --no-home \
    --no-mount cwd \
    --bind "$REPO_DIR/evaluation:/workspace/SymTRELLIS/evaluation:ro" \
    --bind "$DATA_DIR:$DATA_DIR" \
    --pwd /workspace/SymTRELLIS \
    "$SIF" \
    python -m evaluation.sync_metadata \
    --workspace-dir "$DATA_DIR" \
    "${SYNC_ARGS[@]}"
