#!/bin/bash
#SBATCH --job-name=eval_07_postprocess
#SBATCH --output=%x_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --array=0-15

set -euo pipefail

module load apptainer

REPO_DIR="$HOME/SymTRELLIS"
DATA_DIR="$SCRATCH/eval_workspace_after_acceptance"
SIF="$HOME/symtrellis.sif"

: "${SHAPE_FOLDER:?Set SHAPE_FOLDER}"

SYMMETRY_PREDICTION_FOLDER="${SYMMETRY_PREDICTION_FOLDER:-}"
RANK="$SLURM_ARRAY_TASK_ID"
WORLD_SIZE="${WORLD_SIZE:-$((SLURM_ARRAY_TASK_MAX + 1))}"

cd "$DATA_DIR"

apptainer exec --nv \
    --no-home \
    --no-mount cwd \
    --bind "$REPO_DIR/evaluation:/workspace/SymTRELLIS/evaluation:ro" \
    --bind "$REPO_DIR/preprocess:/workspace/SymTRELLIS/preprocess:ro" \
    --bind "$HOME/.cache:$HOME/.cache" \
    --bind "$DATA_DIR:$DATA_DIR" \
    --pwd /workspace/SymTRELLIS \
    "$SIF" \
    python -m evaluation.07_symmetry_postprocess \
    --workspace-dir "$DATA_DIR" \
    --shape-folder "$SHAPE_FOLDER" \
    --symmetry-prediction-folder "$SYMMETRY_PREDICTION_FOLDER" \
    --rank "$RANK" \
    --world-size "$WORLD_SIZE"
