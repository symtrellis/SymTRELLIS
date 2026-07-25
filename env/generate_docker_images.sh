#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly WHEELS_ROOT="${REPO_DIR}/wheels"
readonly LOCAL_INDEX_PORT=18080
readonly BLENDER_VERSION="5.1.2"
readonly BLENDER_SERIES="5.1"

readonly -a COMMON_APT_PACKAGES=(
    ca-certificates
    gcc
    libc6-dev
)

readonly -a COMMON_PIP_PACKAGES=(
    numpy
    scipy
    pillow
    tqdm
    click
    opencv-python-headless
    trimesh
    matplotlib
    imageio
    easydict
    scikit-learn
    "huggingface-hub>=0.30"
    "transformers<5.4.0"
    safetensors
    timm
    kornia
    rembg
    onnxruntime
    "hydra-core~=1.3.2"
    loguru
    optree
    astor
    pymeshfix
    pyvista
    xatlas
    igraph
    moderngl
    plyfile
    pygltflib
    zstandard
    einops
    iopath
    ninja
    jaxtyping
    rich
    nvidia-arch
    pccm
    ccimport
    pybind11
    fire
    sympy
    "spaces==0.51.0"
    "gradio[oauth,mcp]==6.20.0"
    fastapi
    "uvicorn>=0.14.0"
    "websockets>=10.4"
    python-multipart
    usd-core
    warp-lang
    wget
)

readonly -a DEV_APT_PACKAGES=(
    curl
    xz-utils
    libgl1
    libegl1
    libglib2.0-0
    libxcb1
    libgomp1
    libx11-6
    libxext6
    libxfixes3
    libxi6
    libxkbcommon0
    libxrender1
    libsm6
)

readonly -a DEV_PIP_PACKAGES=(
    black
    ipykernel
    notebook
    jupyterlab
    pandas
    imageio-ffmpeg
    open3d
    tensorboard
    lpips
    open-clip-torch
    "objaverse>=0.1.7"
    roma
    point-cloud-utils
    seaborn
    plotly
    scikit-image
    ipycanvas
    ipyevents
)

readonly -a BASE_COMMON_LOCAL_PACKAGES=(
    xformers
    o-voxel
    utils3d
    moge
    trellis
    trellis2
    sam3d-objects
    flash-attn
    pytorch3d
    kaolin
    gsplat
    diff-gaussian-rasterization
    nvdiffrast
    nvdiffrec-render
    cumesh
    flex-gemm
    symtrellis
)

readonly -a DEV_LOCAL_PACKAGES=(
    diffoctreerast
)

declare -Ar CUMM_SPCONV_TAG_BY_CUDA_TAG=(
    [cu124]=cu121
    [cu128]=cu128
    [cu130]=cu130
)

declare -Ar TORCHVISION_BY_TORCH=(
    [2.6.0]=0.21.0
    [2.7.0]=0.22.0
    [2.8.0]=0.23.0
    [2.9.0]=0.24.0
)

PYTHON_VERSION=""
TORCH_VERSION=""
TORCHVISION_VERSION=""
CUDA_VERSION=""
PYTHON_TAG=""
CUDA_TAG=""
ENV_TAG=""
WHEEL_DIR=""
LOCAL_INDEX_PID=""
LOCAL_INDEX_URL=""
TARGET=""
IMAGE_TAG=""
declare -a COMMON_LOCAL_PACKAGES=()

parse_arguments() {
    while (($#)); do
        case "$1" in
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --torch)
            TORCH_VERSION="$2"
            shift 2
            ;;
        --cuda)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        *)
            printf 'unknown argument: %s\n' "$1" >&2
            return 2
            ;;
        esac
    done

    : "${PYTHON_VERSION:?missing --python}"
    : "${TORCH_VERSION:?missing --torch}"
    : "${CUDA_VERSION:?missing --cuda}"
    : "${TARGET:?missing --target}"

    PYTHON_TAG="py${PYTHON_VERSION//./}"
    CUDA_TAG="cu${CUDA_VERSION//./}"
    ENV_TAG="torch${TORCH_VERSION}-${CUDA_TAG}-${PYTHON_TAG}"
    WHEEL_DIR="${WHEELS_ROOT}/${ENV_TAG}"
    TORCHVISION_VERSION="${TORCHVISION_BY_TORCH[$TORCH_VERSION]}"

    local cumm_spconv_tag="${CUMM_SPCONV_TAG_BY_CUDA_TAG[$CUDA_TAG]}"
    COMMON_LOCAL_PACKAGES=(
        "cumm-${cumm_spconv_tag}"
        "spconv-${cumm_spconv_tag}"
        "${BASE_COMMON_LOCAL_PACKAGES[@]}"
    )

    IMAGE_TAG="ghcr.io/symtrellis/symtrellis:${TARGET}-${ENV_TAG}"
}

start_local_index() {
    python3 -m http.server \
        "$LOCAL_INDEX_PORT" \
        --directory "$WHEEL_DIR" >&2 &

    LOCAL_INDEX_PID=$!
    LOCAL_INDEX_URL="http://host.docker.internal:${LOCAL_INDEX_PORT}/"
}

generate_common_dockerfile() {
    local package
    local pip_packages=""
    local local_packages=""

    for package in "${COMMON_PIP_PACKAGES[@]}"; do
        printf -v pip_packages '%s \\\n    "%s"' "$pip_packages" "$package"
    done

    for package in "${COMMON_LOCAL_PACKAGES[@]}"; do
        printf -v local_packages '%s \\\n    "%s"' "$local_packages" "$package"
    done

    cat <<EOF
FROM python:${PYTHON_VERSION}-slim

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG LOCAL_INDEX_URL

ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1

RUN apt-get update \\
 && apt-get install -y --no-install-recommends ${COMMON_APT_PACKAGES[*]} \\
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade \\
    pip \\
    setuptools \\
    wheel

RUN python -m pip install --no-cache-dir \\
    "torch==${TORCH_VERSION}" \\
    "torchvision==${TORCHVISION_VERSION}" \\
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

RUN python -m pip install --no-cache-dir${pip_packages}

RUN python -m pip install --no-cache-dir \\
    --no-deps \\
    --no-index \\
    --trusted-host host.docker.internal \\
    --find-links "\${LOCAL_INDEX_URL}"${local_packages}

ENV ATTN_BACKEND=flash_attn \\
    SPARSE_ATTN_BACKEND=flash_attn
EOF
}

generate_dev_dockerfile() {
    local package
    local pip_packages=""
    local local_packages=""

    for package in "${DEV_PIP_PACKAGES[@]}"; do
        printf -v pip_packages '%s \\\n    "%s"' "$pip_packages" "$package"
    done

    for package in "${DEV_LOCAL_PACKAGES[@]}"; do
        printf -v local_packages '%s \\\n    "%s"' "$local_packages" "$package"
    done

    cat <<EOF
RUN python -m pip install --no-cache-dir${pip_packages}

RUN python -m pip install --no-cache-dir \\
    --no-deps \\
    --no-index \\
    --trusted-host host.docker.internal \\
    --find-links "\${LOCAL_INDEX_URL}"${local_packages}

RUN apt-get update \\
 && apt-get install -y --no-install-recommends ${DEV_APT_PACKAGES[*]} \\
 && curl -fsSL \\
    "https://download.blender.org/release/Blender${BLENDER_SERIES}/blender-${BLENDER_VERSION}-linux-x64.tar.xz" \\
    -o /tmp/blender.tar.xz \\
 && tar -xJf /tmp/blender.tar.xz -C /opt \\
 && ln -sf "/opt/blender-${BLENDER_VERSION}-linux-x64/blender" /usr/local/bin/blender \\
 && rm -f /tmp/blender.tar.xz \\
 && rm -rf /var/lib/apt/lists/*
EOF
}

generate_inference_dockerfile() {
    cat <<'EOF'
WORKDIR /app

COPY --from=inference *.py /app/inference/
COPY --from=webui app.py /app/webui/
COPY --from=webui backend/*.py /app/webui/backend/
COPY --from=webui backend/loaders/*.py /app/webui/backend/loaders/
COPY --from=webui backend/operations/*.py /app/webui/backend/operations/
COPY --from=webui frontend/dist /app/webui/frontend/dist

RUN chmod -R a+rX /app/inference /app/webui

ENV PYTHONPATH=/app:/app/webui \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

EXPOSE 7860

CMD ["python", "/app/webui/app.py"]
EOF
}

build_image() {
    local image_tag="$1"
    local dockerfile="$2"

    printf '%s\n' "$dockerfile" |
        docker build \
            --tag "$image_tag" \
            --add-host host.docker.internal:host-gateway \
            --build-arg "LOCAL_INDEX_URL=${LOCAL_INDEX_URL}" \
            --build-context "inference=${REPO_DIR}/inference" \
            --build-context "webui=${REPO_DIR}/webui" \
            -
}

main() {
    local common_dockerfile
    local target_dockerfile

    parse_arguments "$@"

    case "$TARGET" in
    dev)
        target_dockerfile="$(generate_dev_dockerfile)"
        ;;
    inference)
        npm --prefix "${REPO_DIR}/webui/frontend" ci
        npm --prefix "${REPO_DIR}/webui/frontend" run build
        target_dockerfile="$(generate_inference_dockerfile)"
        ;;
    *)
        printf 'invalid target: %s\n' "$TARGET" >&2
        return 2
        ;;
    esac

    start_local_index
    trap 'kill "$LOCAL_INDEX_PID" 2>/dev/null || true' EXIT

    common_dockerfile="$(generate_common_dockerfile)"

    build_image \
        "$IMAGE_TAG" \
        "${common_dockerfile}"$'\n'"${target_dockerfile}"
}

main "$@"
