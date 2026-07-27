#!/bin/bash
#SBATCH --job-name=eval_02_ss
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

SYMMETRY_PREDICTION_FOLDER="${SYMMETRY_PREDICTION_FOLDER:-}"
NOISE_STRENGTH="${NOISE_STRENGTH:-0.2}"
GUIDANCE_STRENGTH="${GUIDANCE_STRENGTH:-0.9}"
GUIDANCE_DURATION="${GUIDANCE_DURATION:-0.3}"
SEED="${SEED:-114514}"
STEPS="${STEPS:-32}"
MAX_RETRY="${MAX_RETRY:-10}"
VOXEL_COUNT_THRESHOLD="${VOXEL_COUNT_THRESHOLD:-0}"
RANK="$SLURM_ARRAY_TASK_ID"
WORLD_SIZE="${WORLD_SIZE:-$((SLURM_ARRAY_TASK_MAX + 1))}"

cd "$REPO_DIR"

apptainer exec --nv \
    --bind "$REPO_DIR:$REPO_DIR" \
    --bind "$DATA_DIR:$DATA_DIR" \
    "$SIF" \
    python -m evaluation.02_generate_sparse_structure \
        --workspace-dir "$DATA_DIR" \
        --mapper-path "$MAPPER_PATH" \
        --symmetry-prediction-folder "$SYMMETRY_PREDICTION_FOLDER" \
        --noise-strength "$NOISE_STRENGTH" \
        --guidance-strength "$GUIDANCE_STRENGTH" \
        --guidance-duration "$GUIDANCE_DURATION" \
        --seed "$SEED" \
        --steps "$STEPS" \
        --allow-retry \
        --max-retry "$MAX_RETRY" \
        --voxel-count-threshold "$VOXEL_COUNT_THRESHOLD" \
        --rank "$RANK" \
        --world-size "$WORLD_SIZE"
