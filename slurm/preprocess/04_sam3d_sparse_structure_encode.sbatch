#!/bin/bash
#SBATCH --job-name=prep_04_sam3d_ss
#SBATCH --output=%x_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=0-127

set -euo pipefail

module load apptainer

REPO_DIR="$HOME/SymTRELLIS"
SIF="$HOME/symtrellis.sif"

: "${DATA_DIR:?Set DATA_DIR}"

NUM_SCALE="${NUM_SCALE:-1}"
MIN_SCALE="${MIN_SCALE:-0.3}"
NUM_ROTS="${NUM_ROTS:-16}"
NUM_PERTS="${NUM_PERTS:-4}"
PERTURBATION_RAD_STD="${PERTURBATION_RAD_STD:-0.07}"
SEED="${SEED:-114514}"
SS_ENCODER="${SS_ENCODER:-facebook/sam-3d-objects/checkpoints/ss_encoder}"
RECOMPUTE_FINISHED="${RECOMPUTE_FINISHED:-0}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

RANK="$SLURM_ARRAY_TASK_ID"
WORLD_SIZE="${WORLD_SIZE:-$((SLURM_ARRAY_TASK_MAX + 1))}"

RECOMPUTE_ARGS=()
if [ "$RECOMPUTE_FINISHED" = "1" ]; then
    RECOMPUTE_ARGS+=(--recompute-finished)
fi

mkdir -p \
    "$DATA_DIR" \
    "$HOME/.cache/triton" \
    "$HOME/.cache/cuda_compute_cache" \
    "$HOME/.cache/huggingface"

cd "$DATA_DIR"

apptainer exec --nv \
    --no-home \
    --no-mount cwd \
    --bind "$REPO_DIR/preprocess:/workspace/SymTRELLIS/preprocess:ro" \
    --bind "$HOME/.cache:$HOME/.cache" \
    --bind "$DATA_DIR:$DATA_DIR" \
    --env "TRITON_CACHE_DIR=$HOME/.cache/triton" \
    --env "CUDA_CACHE_PATH=$HOME/.cache/cuda_compute_cache" \
    --env "HF_HOME=$HOME/.cache/huggingface" \
    --env "HF_HUB_CACHE=$HOME/.cache/huggingface/hub" \
    --env "HF_HUB_OFFLINE=$HF_HUB_OFFLINE" \
    --env "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt" \
    --pwd /workspace/SymTRELLIS \
    "$SIF" \
    python -m preprocess.sam3d.multi_pose_sparse_structure_encode \
    --dataset-dir "$DATA_DIR" \
    --num-scale "$NUM_SCALE" \
    --min-scale "$MIN_SCALE" \
    --num-rots "$NUM_ROTS" \
    --num-perts "$NUM_PERTS" \
    --perturbation-rad-std "$PERTURBATION_RAD_STD" \
    --seed "$SEED" \
    --ss-encoder "$SS_ENCODER" \
    --rank "$RANK" \
    --world-size "$WORLD_SIZE" \
    "${RECOMPUTE_ARGS[@]}"
