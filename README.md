# SymTRELLIS

Official code repository for **SymTRELLIS: Symmetry-Enforced Voxel Latents for 3D Generation**.

[Guangda Ji](https://quantaji.github.io/), [Qimin Chen](https://qiminchen.github.io/), [Qinchan Li](https://qinchanli.github.io/), [Mingrui Zhao](https://mingrui-zhao.github.io/), [Kai Wang](https://kwang-ether.github.io/), [Hao Zhang](https://www.cs.sfu.ca/~haoz/)

<p>
  <a href="https://arxiv.org/abs/2606.04108"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg" alt="Paper"></a>
  <a href="https://huggingface.co/symtrellis/SymTRELLIS"><img src="https://img.shields.io/badge/Hugging%20Face-Model-yellow" alt="Hugging Face Model"></a>
  <a href="https://huggingface.co/spaces/quantaji/SymTRELLIS"><img src="https://img.shields.io/badge/Hugging%20Face-Demo-blueviolet" alt="Hugging Face Demo"></a>
  <a href="https://symtrellis.github.io/"><img src="https://img.shields.io/badge/Project-Website-blue" alt="Project Website"></a>
  <a href="https://huggingface.co/datasets/symtrellis/SymTRELLIS-Latent-Transform-Pairs"><img src="https://img.shields.io/badge/Hugging%20Face-Training%20Data-orange" alt="Hugging Face Training Data"></a>
  <!-- <a href="https://huggingface.co/datasets/symtrellis/SymTRELLIS-Evaluation-Dataset"><img src="https://img.shields.io/badge/Hugging%20Face-Evaluation%20Dataset%20(TBD)-orange" alt="Hugging Face Evaluation Dataset"></a> -->
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License"></a>
</p>

## TL;DR

- We enforce symmetry during generation, not as post-processing.
- We support arbitrary 3D finite point group symmetries.
- We do not retrain the flow model or VAE, and do not use generation-time optimization.
- We train a lightweight **spatial-transform latent mapper** to average rotated voxel latents.
- The mapper does not require symmetric training data and is scalable.
- The idea is inspired by linear constraints in generative modeling, also known as visual anagrams ([Visual Anagrams](https://dangeng.github.io/visual_anagrams/), [LookingGlass](https://lookingglass-lpw.github.io/)).
- We also implement GPU-accelerated o-voxel representation encoding ([quantaji/o-voxel-gpu](https://github.com/quantaji/o-voxel-gpu)).

## News

- **2026-07-19**: Initial code repository setup.


## Abstract
Single-view 3D generative models have achieved impressive visual quality, yet they are not designed to satisfy structural or functional requirements, and in practice, often fall short. Symmetry is one such requirement: violations, even subtle ones, on symmetry can render a model physically unusable. We present **SymTRELLIS**, a method that enforces **arbitrary finite point group symmetries** (rotational, reflectional, and polyhedral) during the flow-based 3D generation of TRELLIS.2, **without retraining** the underlying VAE or flow model. Our key idea is to approximate the latent-space action of spatial transformations as a learned linear operator on voxel latents, implemented as a lightweight **spatial-transform latent mapper** trained on generic, non-symmetric 3D data. At generation time, we enforce symmetry by averaging predicted flow velocities across all symmetry-equivalent transformations at each ODE step, a process we call **velocity symmetrization**. The symmetry specification can be estimated automatically from an initial TRELLIS.2 generation or supplied by the user, enabling deliberate fold manipulation beyond what the input image suggests. On a curated benchmark of 266 strictly symmetric objects spanning 2- to 20-fold rotations and polyhedral symmetry groups, SymTRELLIS substantially reduces all symmetry error metrics compared to TRELLIS.2, Hunyuan3D-2.1, and TripoSG, while maintaining reconstruction accuracy comparable to the base model.

## Local demo implementation

```bash
ENV_TAG=torch2.9.0-cu130-py312
# Options:
# torch2.9.0-cu130-py312
# torch2.8.0-cu128-py312
# torch2.7.0-cu128-py311
# torch2.6.0-cu124-py310

mkdir -p \
  "$HOME/.cache/huggingface" \
  "$HOME/.config/huggingface" \
  "$HOME/.cache/triton"

docker run --rm --gpus all                                          \
  --user "$(id -u):$(id -g)"                                        \
  -p 7860:7860                                                      \
  -e SYMTRELLIS_WEBUI_SESSION_TIMEOUT_SECONDS=315360000             \
  -e SYMTRELLIS_WEBUI_CLEANUP_INTERVAL_SECONDS=300                  \
  -e HF_HOME=/root/.cache/huggingface                               \
  -e HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub             \
  -e TRITON_CACHE_DIR=/root/.cache/triton                           \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface"            \
  -v "$HOME/.config/huggingface:/root/.config/huggingface:ro"       \
  -v "$HOME/.cache/triton:/root/.cache/triton"                      \
  ghcr.io/symtrellis/symtrellis:inference-"$ENV_TAG"
```

## Training

### Data

Code for preparing the training data is provided in [`preprocess/`](preprocess/). The preprocessed data can also be downloaded from the [SymTRELLIS Latent Transform Pairs](https://huggingface.co/datasets/symtrellis/SymTRELLIS-Latent-Transform-Pairs) dataset.

### Sparse Structure Mapper Pretraining

Set `MODEL_BACKEND` to either `neighbor_graph` or `swin3d`.

```bash
MODEL_BACKEND="neighbor_graph"
DATASET_ROOT="<dataset-root>"
TRAIN_OUTPUT_DIR="<training-output-dir>"

mkdir -p \
  "$TRAIN_OUTPUT_DIR" \
  "$HOME/.cache/cuda_compute_cache" \
  "$HOME/.cache/triton" \
  "$HOME/.cache/huggingface" \
  "$HOME/.config/huggingface"

docker run --rm --gpus all --ipc=host \
  --user "$(id -u):$(id -g)" \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e CUDA_CACHE_PATH=/cuda-cache \
  -e CUDA_CACHE_MAXSIZE=2147483648 \
  -e TRITON_CACHE_DIR=/triton-cache \
  -e HF_HOME=/hf-cache \
  -e HF_HUB_CACHE=/hf-cache/hub \
  -e XDG_CONFIG_HOME=/tmp/.config \
  -e HOME=/tmp/user-home \
  -v "$HOME/.cache/cuda_compute_cache:/cuda-cache" \
  -v "$HOME/.cache/triton:/triton-cache" \
  -v "$HOME/.cache/huggingface:/hf-cache" \
  -v "$HOME/.config/huggingface:/tmp/.config/huggingface:ro" \
  -v "$PWD/dataset:/workspace/SymTRELLIS/dataset:ro" \
  -v "$PWD/trainer:/workspace/SymTRELLIS/trainer:ro" \
  -v "$DATASET_ROOT:/data:ro" \
  -v "$TRAIN_OUTPUT_DIR:/output" \
  -w /workspace/SymTRELLIS \
  ghcr.io/symtrellis/symtrellis:dev-torch2.9.0-cu130-py312 \
  bash -lc '
    mkdir -p "$HOME"

    PYTHONPATH=/workspace/SymTRELLIS \
    torchrun --standalone --nproc_per_node=1 trainer/train.py \
      --task trellis2_sparse_structure \
      --train-data-dir /data/objaversexl_sketchfab/trellis2/multi_slats/ss_enc_conv3d_16l8_fp16_sslatentres_16_occres_64_s1_r16_p4 \
      --eval-data-dir /data/toys4k/trellis2/multi_slats/ss_enc_conv3d_16l8_fp16_sslatentres_16_occres_64_s1_r16_p4 \
      --grid-size 16 \
      --latent-dim 8 \
      --num-scale 1 \
      --num-rots 16 \
      --num-perts 4 \
      --model-backend '"$MODEL_BACKEND"' \
      --attention-backend xformers \
      --model-scale base \
      --lowrank-rank 64 \
      --dst-input-norm-threshold 1.5 \
      --decoded-loss-weight 0.0 \
      --batch-size 8 \
      --accumulation-steps 32 \
      --steps-per-epoch 250 \
      --epochs 100 \
      --lr-scheduler one_cycle \
      --lr 1e-3 \
      --weight-decay 1e-2 \
      --log-interval 5 \
      --max-eval-batches 100 \
      --log-dir /output/tensorboard \
      --checkpoint-dir /output/checkpoints \
      2>&1 | tee /output/train.log
  '
```

This configuration uses a minibatch size of 8 and 32 accumulation steps, giving an effective batch size of 256.

### Shape Mapper Pretraining

The shape mapper uses the same Docker environment and cache layout, but has its own dataset format and model dimensions.

```bash
MODEL_BACKEND="neighbor_graph"
DATASET_ROOT="<dataset-root>"
TRAIN_OUTPUT_DIR="<training-output-dir>"

mkdir -p \
  "$TRAIN_OUTPUT_DIR" \
  "$HOME/.cache/cuda_compute_cache" \
  "$HOME/.cache/triton" \
  "$HOME/.cache/huggingface" \
  "$HOME/.config/huggingface"

docker run --rm --gpus all --ipc=host \
  --user "$(id -u):$(id -g)" \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e CUDA_CACHE_PATH=/cuda-cache \
  -e CUDA_CACHE_MAXSIZE=2147483648 \
  -e TRITON_CACHE_DIR=/triton-cache \
  -e HF_HOME=/hf-cache \
  -e HF_HUB_CACHE=/hf-cache/hub \
  -e XDG_CONFIG_HOME=/tmp/.config \
  -e HOME=/tmp/user-home \
  -v "$HOME/.cache/cuda_compute_cache:/cuda-cache" \
  -v "$HOME/.cache/triton:/triton-cache" \
  -v "$HOME/.cache/huggingface:/hf-cache" \
  -v "$HOME/.config/huggingface:/tmp/.config/huggingface:ro" \
  -v "$PWD/dataset:/workspace/SymTRELLIS/dataset:ro" \
  -v "$PWD/trainer:/workspace/SymTRELLIS/trainer:ro" \
  -v "$DATASET_ROOT:/data:ro" \
  -v "$TRAIN_OUTPUT_DIR:/output" \
  -w /workspace/SymTRELLIS \
  ghcr.io/symtrellis/symtrellis:dev-torch2.9.0-cu130-py312 \
  bash -lc '
    mkdir -p "$HOME"

    PYTHONPATH=/workspace/SymTRELLIS \
    torchrun --standalone --nproc_per_node=1 trainer/train.py \
      --task trellis2_shape \
      --train-data-dir /data/objaversexl_sketchfab/trellis2/multi_slats/shape_enc_next_dc_f16c32_fp16_shapelatentres_32_ovoxres_512_s3_r6_p3 \
      --eval-data-dir /data/toys4k/trellis2/multi_slats/shape_enc_next_dc_f16c32_fp16_shapelatentres_32_ovoxres_512_s3_r6_p3 \
      --grid-size 32 \
      --latent-dim 32 \
      --num-scale 3 \
      --num-rots 6 \
      --num-perts 3 \
      --model-backend '"$MODEL_BACKEND"' \
      --attention-backend xformers \
      --model-scale base \
      --lowrank-rank 256 \
      --decoded-output-weight 0.0 \
      --decoded-subdivision-weight 0.0 \
      --batch-size 16 \
      --accumulation-steps 16 \
      --steps-per-epoch 250 \
      --epochs 100 \
      --lr-scheduler one_cycle \
      --lr 1e-3 \
      --weight-decay 1e-2 \
      --log-interval 5 \
      --max-eval-batches 100 \
      --log-dir /output/tensorboard \
      --checkpoint-dir /output/checkpoints \
      2>&1 | tee /output/train.log
  '
```

This configuration uses a minibatch size of 16 and 16 accumulation steps, also giving an effective batch size of 256.

## BibTeX

```bibtex
@misc{ji2026symtrellis,
  title={{SymTRELLIS}: Symmetry-Enforced Voxel Latents for 3D Generation},
  author={Guangda Ji and Qimin Chen and Qinchan Li and Mingrui Zhao and Kai Wang and Hao Zhang},
  year={2026},
  eprint={2606.04108},
  archivePrefix={arXiv},
  primaryClass={cs.GR},
  url={https://arxiv.org/abs/2606.04108}
}
```
