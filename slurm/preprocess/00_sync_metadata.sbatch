#!/bin/bash
#SBATCH --job-name=prep_00_metadata
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=30:00

set -euo pipefail

module load apptainer

REPO_DIR="$HOME/SymTRELLIS"
SIF="$HOME/symtrellis.sif"

: "${DATA_DIR:?Set DATA_DIR}"

DATASET="${DATASET:-Toys4K}"
FULL_CHECK="${FULL_CHECK:-0}"
FOLDERS="${FOLDERS:-}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

SYNC_ARGS=()
if [ "$FULL_CHECK" = "1" ]; then
    SYNC_ARGS+=(--full-check)
fi
if [ -n "$FOLDERS" ]; then
    read -r -a FOLDER_ARGS <<< "$FOLDERS"
    SYNC_ARGS+=(--folders "${FOLDER_ARGS[@]}")
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
    python -m preprocess.sync_metadata "$DATASET" \
    --dataset-dir "$DATA_DIR" \
    "${SYNC_ARGS[@]}"
