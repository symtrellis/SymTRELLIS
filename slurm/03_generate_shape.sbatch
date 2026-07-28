#!/bin/bash
#SBATCH --job-name=eval_03_shape
#SBATCH --output=%x_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --array=0-15

set -euo pipefail

module load apptainer

REPO_DIR="$HOME/SymTRELLIS"
DATA_DIR="$SCRATCH/eval_workspace_after_acceptance"
SIF="$HOME/symtrellis.sif"

: "${MAPPER_PATH:?Set MAPPER_PATH}"
: "${SPARSE_STRUCTURE_FOLDER:?Set SPARSE_STRUCTURE_FOLDER}"

SYMMETRY_PREDICTION_FOLDER="${SYMMETRY_PREDICTION_FOLDER:-}"
NOISE_STRENGTH="${NOISE_STRENGTH:-0.2}"
GUIDANCE_STRENGTH="${GUIDANCE_STRENGTH:-0.4}"
GUIDANCE_DURATION="${GUIDANCE_DURATION:-0.3}"
SEED="${SEED:-114514}"
STEPS="${STEPS:-32}"
RANK="$SLURM_ARRAY_TASK_ID"
WORLD_SIZE="${WORLD_SIZE:-$((SLURM_ARRAY_TASK_MAX + 1))}"

mkdir -p "$HOME/.cache/triton" "$HOME/.cache/cuda_compute_cache"

cd "$DATA_DIR"

apptainer exec --nv \
    --no-home \
    --no-mount cwd \
    --bind "$REPO_DIR/evaluation:/workspace/SymTRELLIS/evaluation:ro" \
    --bind "$REPO_DIR/inference:/workspace/SymTRELLIS/inference:ro" \
    --bind "$REPO_DIR/preprocess:/workspace/SymTRELLIS/preprocess:ro" \
    --bind "$HOME/.cache:$HOME/.cache" \
    --env "TRITON_CACHE_DIR=$HOME/.cache/triton" \
    --env "CUDA_CACHE_PATH=$HOME/.cache/cuda_compute_cache" \
    --env "HF_HOME=$HOME/.cache/huggingface" \
    --env "HF_HUB_CACHE=$HOME/.cache/huggingface/hub" \
    --env "HF_HUB_OFFLINE=1" \
    --env "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt" \
    --bind "$DATA_DIR:$DATA_DIR" \
    --pwd /workspace/SymTRELLIS \
    "$SIF" \
    python -m evaluation.03_generate_shape \
    --workspace-dir "$DATA_DIR" \
    --sparse-structure-folder "$SPARSE_STRUCTURE_FOLDER" \
    --mapper-path "$MAPPER_PATH" \
    --symmetry-prediction-folder "$SYMMETRY_PREDICTION_FOLDER" \
    --noise-strength "$NOISE_STRENGTH" \
    --guidance-strength "$GUIDANCE_STRENGTH" \
    --guidance-duration "$GUIDANCE_DURATION" \
    --seed "$SEED" \
    --steps "$STEPS" \
    --rank "$RANK" \
    --world-size "$WORLD_SIZE"
