#!/bin/bash
#SBATCH --job-name=prep_07_trellis2_shape
#SBATCH --output=%x_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --array=0-255

set -euo pipefail

module load apptainer

REPO_DIR="$HOME/SymTRELLIS"
SIF="$HOME/symtrellis.sif"

: "${DATA_DIR:?Set DATA_DIR}"

NUM_SCALE="${NUM_SCALE:-3}"
MIN_SCALE="${MIN_SCALE:-0.3}"
NUM_ROTS="${NUM_ROTS:-6}"
NUM_PERTS="${NUM_PERTS:-3}"
SHAPE_LATENT_RESOLUTION="${SHAPE_LATENT_RESOLUTION:-32}"
PERTURBATION_RAD_STD="${PERTURBATION_RAD_STD:-0.07}"
SEED="${SEED:-114514}"
SHAPE_ENCODER="${SHAPE_ENCODER:-microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16}"
RECOMPUTE_FINISHED="${RECOMPUTE_FINISHED:-0}"
REMOVE_PICKLE="${REMOVE_PICKLE:-0}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

RANK="$SLURM_ARRAY_TASK_ID"
WORLD_SIZE="${WORLD_SIZE:-$((SLURM_ARRAY_TASK_MAX + 1))}"

RUN_ARGS=()
if [ "$RECOMPUTE_FINISHED" = "1" ]; then
    RUN_ARGS+=(--recompute-finished)
fi
if [ "$REMOVE_PICKLE" = "1" ]; then
    RUN_ARGS+=(--remove-pickle)
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
    python -m preprocess.trellis2.multi_pose_dual_grid_shape_encode \
    --dataset-dir "$DATA_DIR" \
    --num-scale "$NUM_SCALE" \
    --min-scale "$MIN_SCALE" \
    --num-rots "$NUM_ROTS" \
    --num-perts "$NUM_PERTS" \
    --shape-latent-resolution "$SHAPE_LATENT_RESOLUTION" \
    --perturbation-rad-std "$PERTURBATION_RAD_STD" \
    --seed "$SEED" \
    --shape-encoder "$SHAPE_ENCODER" \
    --rank "$RANK" \
    --world-size "$WORLD_SIZE" \
    "${RUN_ARGS[@]}"
