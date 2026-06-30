#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-dockers}"
IMAGE_REPO="${IMAGE_REPO:-}"
BLENDER_VERSION="${BLENDER_VERSION:-5.1.2}"
BLENDER_SERIES="${BLENDER_SERIES:-5.1}"
OVOXEL_VERSION="${OVOXEL_VERSION:-0.0.1}"
OVOXEL_RELEASE_TAG="${OVOXEL_RELEASE_TAG:-v${OVOXEL_VERSION}}"
OVOXEL_REPO="${OVOXEL_REPO:-quantaji/o-voxel-gpu}"
FLASH_ATTN_REPO="${FLASH_ATTN_REPO:-Dao-AILab/flash-attention}"

CUDA_VERSION=""
CUDA_MINOR_VERSION=""
TORCH_VERSION=""
TORCHVISION_VERSION=""
PYTHON_VERSION=""
CUDA_TAG=""
PYTHON_CP_TAG=""
PYTHON_SHORT_TAG=""
BASE_IMAGE=""
TORCH_INDEX_URL=""
SPCONV_PACKAGE=""
XFORMERS_VERSION="${XFORMERS_VERSION:-}"
KAOLIN_FIND_LINKS="${KAOLIN_FIND_LINKS:-}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}"
FLASH_ATTN_CUDA_KEY=""
FLASH_ATTN_TORCH_KEY=""
FLASH_ATTN_RELEASES_API=""
FLASH_ATTN_RELEASE_TAG=""
FLASH_ATTN_INSTALL_MODE=""
OVOXEL_WHEEL_NAME=""
OVOXEL_RELEASE_API=""
OVOXEL_WHEEL_URL=""
OVOXEL_INSTALL_MODE=""
IMAGE_TAG=""
IMAGE_FILE_TAG=""
IMAGE_VERSION_TAG=""
IMAGE_ARCHIVE_NAME=""
REMOTE_TAG=""
DOCKERFILE_PATH=""

PURE_PYTHON_PACKAGES=(
    black ipykernel notebook jupyterlab numpy scipy pandas tqdm pillow
    imageio imageio-ffmpeg opencv-python-headless trimesh open3d pymeshfix
    pyvista xatlas "huggingface_hub[cli]" transformers safetensors easydict
    tensorboard lpips rembg onnxruntime open_clip_torch objaverse astor
    optree roma point-cloud-utils seaborn==0.13.2 gradio==5.49.0
    matplotlib plotly kornia timm zstandard einops iopath scikit-image
    plyfile pygltflib ipycanvas ipyevents usd-core warp-lang
)

VCS_PYTHON_PACKAGES=(
    "git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900"
    "MoGe @ git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b"
)

PURE_PYTHON_IMPORTS=(
    black IPython ipykernel notebook jupyterlab numpy scipy pandas tqdm PIL
    imageio imageio_ffmpeg cv2 trimesh open3d pymeshfix pyvista xatlas
    huggingface_hub transformers safetensors easydict tensorboard lpips
    rembg onnxruntime open_clip objaverse astor optree roma point_cloud_utils
    seaborn gradio matplotlib plotly kornia timm zstandard einops iopath
    skimage plyfile pygltflib ipycanvas ipyevents pxr warp cairo utils3d moge
)

CORE_IMPORTS=(
    torch torchvision flash_attn xformers spconv pytorch3d gsplat
    nvdiffrast.torch nvdiffrec_render diffoctreerast cumesh
)

RUNTIME_GPU_IMPORTS=(
    kaolin flex_gemm o_voxel
)

SYMTRELLIS_IMPORTS=(
    symtrellis
    symtrellis.geometry.neighbors.sparse_lattice_ext._C
    symtrellis.mapper.attention.csr_attn_ext._C
)

DIST_ONLY_CHECKS=(
    kaolin
)

usage() {
    cat <<'EOF'
Usage:
  bash env/generate_dockerfile.sh --python PYTHON_VERSION --torch TORCH_VERSION --cuda CUDA_VERSION

Example:
  bash env/generate_dockerfile.sh --python 3.10 --torch 2.7.0 --cuda 12.8

What it does:
  - validates that the requested torch/cuda/python wheel exists in the PyTorch index
  - derives the matching torchvision version and validates its wheel
  - validates xformers, spconv, and kaolin binary wheels before writing anything
  - resolves flash-attn official release tag and builds it from source
  - writes dockers/symtrellis-torchX.Y.Z-cuXXX-pyXXX.dockerfile
  - prints docker build, tag, push, save, and SIF conversion commands

Environment overrides:
  IMAGE_REPO              Remote image repository used in printed tag/push commands.
  OUTPUT_DIR              Dockerfile output directory. Default: dockers
  BLENDER_VERSION         Official Blender binary version. Default: 5.1.2
  BLENDER_SERIES          Blender release series URL segment. Default: 5.1
  OVOXEL_VERSION          o-voxel package version. Default: 0.0.1
  OVOXEL_RELEASE_TAG      o-voxel GitHub release tag. Default: v0.0.1
  OVOXEL_REPO             o-voxel GitHub repository. Default: quantaji/o-voxel-gpu
  FLASH_ATTN_REPO         flash-attn GitHub repository. Default: Dao-AILab/flash-attention
  XFORMERS_VERSION        Override the torch-derived xformers version.
  KAOLIN_FIND_LINKS       Override the NVIDIA kaolin wheel index URL.
  TORCH_CUDA_ARCH_LIST    Override the generated CUDA architecture list.

Generated Docker build args:
  MAX_JOBS                Optional compile parallelism cap. Default inside Docker: 4.

Notes:
  - This script generates a Dockerfile only. It does not build it.
  - Docker builds should use BuildKit because the generated Dockerfile uses bind mounts.
  - Docker images do not use conda; they use the system Python installed in the image.
EOF
}

die() {
    echo "error: $*" >&2
    exit 2
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "$1 is required but was not found in PATH"
}

regex_escape() {
    printf '%s' "$1" | sed 's/[][\\.^$*+?{}|()]/\\&/g'
}

emit_shell_args() {
    local arg
    for arg in "$@"; do
        printf '    %q \\\n' "$arg"
    done
}

emit_python_strings() {
    local item
    for item in "$@"; do
        printf '    "%s",\n' "$item"
    done
}

python_list_literal() {
    local first=1 item
    printf '['
    for item in "$@"; do
        if [[ "$first" -eq 1 ]]; then
            first=0
        else
            printf ','
        fi
        printf '"%s"' "$item"
    done
    printf ']'
}

cuda_minor_version() {
    local major minor patch
    IFS='.' read -r major minor patch <<<"$1"
    [[ -n "${major:-}" && -n "${minor:-}" ]] || die "CUDA version must look like 12.8 or 12.8.0: $1"
    echo "${major}.${minor}"
}

torch_cuda_tag() {
    local version
    version="$(cuda_minor_version "$1")"
    echo "cu${version//./}"
}

python_cp_tag() {
    case "$1" in
    3.8) echo "cp38" ;;
    3.9) echo "cp39" ;;
    3.10) echo "cp310" ;;
    3.11) echo "cp311" ;;
    3.12) echo "cp312" ;;
    3.13) echo "cp313" ;;
    3.14) echo "cp314" ;;
    *) die "Unsupported Python version: $1. Add a mapping in python_cp_tag()." ;;
    esac
}

python_short_tag() {
    echo "py${1//./}"
}

docker_image_for_cuda() {
    case "$1" in
    11.8) echo "nvidia/cuda:11.8.0-devel-ubuntu22.04" ;;
    12.1) echo "nvidia/cuda:12.1.0-devel-ubuntu22.04" ;;
    12.4) echo "nvidia/cuda:12.4.0-devel-ubuntu22.04" ;;
    12.6) echo "nvidia/cuda:12.6.0-devel-ubuntu22.04" ;;
    12.8) echo "nvidia/cuda:12.8.0-devel-ubuntu22.04" ;;
    12.9) echo "nvidia/cuda:12.9.0-devel-ubuntu22.04" ;;
    *) die "Unsupported CUDA version: $1. Add a mapping in docker_image_for_cuda()." ;;
    esac
}

torchvision_for_torch() {
    case "$1" in
    2.0.0) echo "0.15.0" ;;
    2.0.1) echo "0.15.2" ;;
    2.1.0) echo "0.16.0" ;;
    2.1.1) echo "0.16.1" ;;
    2.1.2) echo "0.16.2" ;;
    2.2.0) echo "0.17.0" ;;
    2.2.1) echo "0.17.1" ;;
    2.2.2) echo "0.17.2" ;;
    2.3.0) echo "0.18.0" ;;
    2.3.1) echo "0.18.1" ;;
    2.4.0) echo "0.19.0" ;;
    2.4.1) echo "0.19.1" ;;
    2.5.0) echo "0.20.0" ;;
    2.5.1) echo "0.20.1" ;;
    2.6.0) echo "0.21.0" ;;
    2.7.0) echo "0.22.0" ;;
    2.7.1) echo "0.22.1" ;;
    2.8.0) echo "0.23.0" ;;
    2.9.0) echo "0.24.0" ;;
    2.9.1) echo "0.24.1" ;;
    2.10.0) echo "0.25.0" ;;
    2.11.0) echo "0.26.0" ;;
    2.12.0) echo "0.27.0" ;;
    2.12.1) echo "0.27.1" ;;
    *) die "Unsupported torch version: $1. Add a torchvision mapping first." ;;
    esac
}

xformers_for_torch() {
    case "$1" in
    2.2.2) echo "0.0.25.post1" ;;
    2.3.0) echo "0.0.26.post1" ;;
    2.3.1) echo "0.0.27" ;;
    2.4.0) echo "0.0.27.post2" ;;
    2.4.1) echo "0.0.28" ;;
    2.5.0) echo "0.0.28.post2" ;;
    2.5.1) echo "0.0.28.post3" ;;
    2.6.0) echo "0.0.29.post3" ;;
    2.7.0) echo "0.0.30" ;;
    2.7.1) echo "0.0.31.post1" ;;
    2.8.0) echo "0.0.32.post2" ;;
    2.9.0) echo "0.0.33.post1" ;;
    2.9.1) echo "0.0.33.post2" ;;
    2.10.0) echo "0.0.34" ;;
    2.11.0 | 2.12.0 | 2.12.1) echo "0.0.35" ;;
    *) die "Unsupported torch version for xformers: $1. Add an xformers mapping first." ;;
    esac
}

spconv_pkg_for_cuda() {
    case "$1" in
    11.8) echo "spconv-cu118" ;;
    12.*) echo "spconv-cu120" ;;
    *) die "Unsupported CUDA version for spconv: $1" ;;
    esac
}

default_arch_list_for_torch_cuda() {
    local torch_mm="${1%.*}"
    case "$2" in
    11.8 | 12.1 | 12.4 | 12.6)
        echo "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0+PTX"
        ;;
    12.8)
        case "$torch_mm" in
        2.7 | 2.8 | 2.9 | 2.10 | 2.11 | 2.12)
            echo "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;12.0+PTX"
            ;;
        *) echo "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0+PTX" ;;
        esac
        ;;
    12.9)
        case "$torch_mm" in
        2.8 | 2.9 | 2.10 | 2.11 | 2.12)
            echo "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.3;12.0;12.1+PTX"
            ;;
        *) echo "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0+PTX" ;;
        esac
        ;;
    *) die "Unsupported CUDA version for arch list: $2" ;;
    esac
}

parse_args() {
    [[ $# -gt 0 ]] || {
        usage >&2
        exit 2
    }

    while [[ $# -gt 0 ]]; do
        case "$1" in
        -h | --help)
            usage
            exit 0
            ;;
        --cuda | --torch | --python)
            [[ $# -ge 2 && "$2" != --* ]] || die "$1 requires a value"
            case "$1" in
            --cuda)
                [[ -z "$CUDA_VERSION" ]] || die "--cuda was specified more than once"
                CUDA_VERSION="$2"
                ;;
            --torch)
                [[ -z "$TORCH_VERSION" ]] || die "--torch was specified more than once"
                TORCH_VERSION="$2"
                ;;
            --python)
                [[ -z "$PYTHON_VERSION" ]] || die "--python was specified more than once"
                PYTHON_VERSION="$2"
                ;;
            esac
            shift 2
            ;;
        --cuda=* | --torch=* | --python=*)
            local key="${1%%=*}" value="${1#*=}"
            [[ -n "$value" ]] || die "$key requires a value"
            case "$key" in
            --cuda)
                [[ -z "$CUDA_VERSION" ]] || die "--cuda was specified more than once"
                CUDA_VERSION="$value"
                ;;
            --torch)
                [[ -z "$TORCH_VERSION" ]] || die "--torch was specified more than once"
                TORCH_VERSION="$value"
                ;;
            --python)
                [[ -z "$PYTHON_VERSION" ]] || die "--python was specified more than once"
                PYTHON_VERSION="$value"
                ;;
            esac
            shift
            ;;
        *) die "unknown argument: $1" ;;
        esac
    done

    [[ -n "$PYTHON_VERSION" ]] || die "missing required argument: --python"
    [[ -n "$TORCH_VERSION" ]] || die "missing required argument: --torch"
    [[ -n "$CUDA_VERSION" ]] || die "missing required argument: --cuda"
    CUDA_MINOR_VERSION="$(cuda_minor_version "$CUDA_VERSION")"
}

derive_versions() {
    CUDA_TAG="$(torch_cuda_tag "$CUDA_VERSION")"
    PYTHON_CP_TAG="$(python_cp_tag "$PYTHON_VERSION")"
    PYTHON_SHORT_TAG="$(python_short_tag "$PYTHON_VERSION")"
    BASE_IMAGE="$(docker_image_for_cuda "$CUDA_MINOR_VERSION")"
    TORCHVISION_VERSION="$(torchvision_for_torch "$TORCH_VERSION")"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/${CUDA_TAG}"
    SPCONV_PACKAGE="$(spconv_pkg_for_cuda "$CUDA_MINOR_VERSION")"
    XFORMERS_VERSION="${XFORMERS_VERSION:-$(xformers_for_torch "$TORCH_VERSION")}"
    KAOLIN_FIND_LINKS="${KAOLIN_FIND_LINKS:-https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-${TORCH_VERSION}_${CUDA_TAG}.html}"
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(default_arch_list_for_torch_cuda "$TORCH_VERSION" "$CUDA_MINOR_VERSION")}"
    FLASH_ATTN_CUDA_KEY="cu${CUDA_MINOR_VERSION%%.*}"
    FLASH_ATTN_TORCH_KEY="torch${TORCH_VERSION%.*}"
    FLASH_ATTN_RELEASES_API="https://api.github.com/repos/${FLASH_ATTN_REPO}/releases?per_page=100"
    OVOXEL_WHEEL_NAME="o_voxel-${OVOXEL_VERSION}+torch${TORCH_VERSION}.${CUDA_TAG}-${PYTHON_CP_TAG}-${PYTHON_CP_TAG}-linux_x86_64.whl"
    OVOXEL_RELEASE_API="https://api.github.com/repos/${OVOXEL_REPO}/releases/tags/${OVOXEL_RELEASE_TAG}"
    IMAGE_FILE_TAG="symtrellis-torch${TORCH_VERSION}-${CUDA_TAG}-${PYTHON_SHORT_TAG}"
    IMAGE_VERSION_TAG="torch${TORCH_VERSION}-${CUDA_TAG}-${PYTHON_SHORT_TAG}"
    IMAGE_TAG="symtrellis/env:${IMAGE_VERSION_TAG}"
    IMAGE_ARCHIVE_NAME="symtrellis-env-${IMAGE_VERSION_TAG}"
    REMOTE_TAG="${IMAGE_REPO:-registry.example.com/user/symtrellis-env}:${IMAGE_VERSION_TAG}"
    DOCKERFILE_PATH="${REPO_DIR}/${OUTPUT_DIR}/${IMAGE_FILE_TAG}.dockerfile"
}

github_release_asset_url() {
    local encoded_name="${3//+/%2B}"
    echo "https://github.com/${1}/releases/download/${2}/${encoded_name}"
}

resolve_flash_attn_plan() {
    local json current_tag="" current_prerelease=0 fallback_tag="" exact=0
    local asset exact_base exact_py
    json="$(mktemp)"
    curl -fsSL "$FLASH_ATTN_RELEASES_API" -o "$json"
    exact_base="${FLASH_ATTN_CUDA_KEY}${FLASH_ATTN_TORCH_KEY}cxx11abi"
    exact_py="-${PYTHON_CP_TAG}-${PYTHON_CP_TAG}-linux_x86_64.whl"

    while IFS= read -r line; do
        if [[ "$line" == *'"tag_name":'* ]]; then
            current_tag="$(printf '%s\n' "$line" | sed -E 's/.*"tag_name": "([^"]+)".*/\1/')"
            current_prerelease=0
            continue
        fi
        [[ "$line" != *'"prerelease": true'* ]] || {
            current_prerelease=1
            continue
        }
        [[ "$line" != *'"prerelease": false'* ]] || {
            current_prerelease=0
            continue
        }
        [[ "$current_prerelease" -eq 0 && "$line" == *'"name": "flash_attn-'* ]] || continue

        asset="$(printf '%s\n' "$line" | sed -E 's/.*"name": "([^"]+)".*/\1/')"
        if [[ -z "$fallback_tag" && "$asset" == flash_attn-*+"${exact_base}"*linux_x86_64.whl ]]; then
            fallback_tag="$current_tag"
        fi
        if [[ "$asset" == flash_attn-*+"${exact_base}"FALSE"${exact_py}" ||
            "$asset" == flash_attn-*+"${exact_base}"TRUE"${exact_py}" ]]; then
            FLASH_ATTN_RELEASE_TAG="$current_tag"
            exact=1
            break
        fi
    done <"$json"
    rm -f "$json"

    if [[ "$exact" -eq 1 ]]; then
        FLASH_ATTN_INSTALL_MODE="source build ${FLASH_ATTN_RELEASE_TAG} (${FLASH_ATTN_CUDA_KEY}, ${FLASH_ATTN_TORCH_KEY}, ${PYTHON_CP_TAG}; release has exact wheel metadata)"
    elif [[ -n "$fallback_tag" ]]; then
        FLASH_ATTN_RELEASE_TAG="$fallback_tag"
        FLASH_ATTN_INSTALL_MODE="source build ${FLASH_ATTN_RELEASE_TAG} (${FLASH_ATTN_CUDA_KEY}, ${FLASH_ATTN_TORCH_KEY}; no exact ${PYTHON_CP_TAG} wheel metadata)"
    else
        cat >&2 <<EOF
flash-attn release not found for:
  cuda key:  ${FLASH_ATTN_CUDA_KEY}
  torch key: ${FLASH_ATTN_TORCH_KEY}
  python:    ${PYTHON_VERSION} (${PYTHON_CP_TAG})
  checked:   ${FLASH_ATTN_RELEASES_API}
EOF
        exit 2
    fi
}

resolve_ovoxel_wheel() {
    local json
    json="$(mktemp)"
    OVOXEL_WHEEL_URL=""
    OVOXEL_INSTALL_MODE="fallback source build from ${OVOXEL_REPO}@${OVOXEL_RELEASE_TAG}"
    if curl -fsSL "$OVOXEL_RELEASE_API" -o "$json" && grep -Fq "$OVOXEL_WHEEL_NAME" "$json"; then
        OVOXEL_WHEEL_URL="$(github_release_asset_url "$OVOXEL_REPO" "$OVOXEL_RELEASE_TAG" "$OVOXEL_WHEEL_NAME")"
        OVOXEL_INSTALL_MODE="release wheel ${OVOXEL_WHEEL_NAME}"
    fi
    rm -f "$json"
}

preflight_index_wheel() {
    local package="$1" version="$2" url="$3" pattern="$4" html
    html="$(mktemp)"
    curl -fsSL "$url" -o "$html"
    if ! grep -Eq "$pattern" "$html"; then
        rm -f "$html"
        cat >&2 <<EOF
Binary wheel not found:
  package: ${package}${version:+==${version}}
  python:  ${PYTHON_VERSION} (${PYTHON_CP_TAG})
  cuda:    ${CUDA_MINOR_VERSION} (${CUDA_TAG})
  checked: ${url}
EOF
        exit 2
    fi
    rm -f "$html"
}

preflight_pytorch_wheel() {
    local pkg="$1" version="$2" path="$3"
    local pkg_re version_re cp_re cuda_re pattern
    pkg_re="$(regex_escape "$pkg")"
    version_re="$(regex_escape "$version")"
    cp_re="$(regex_escape "$PYTHON_CP_TAG")"
    cuda_re="$(regex_escape "$CUDA_TAG")"
    pattern="${pkg_re}-${version_re}(%2B|\\+)${cuda_re}-${cp_re}-${cp_re}-[^\"'<> ]*x86_64\\.whl"
    preflight_index_wheel "$pkg" "$version" "${TORCH_INDEX_URL}/${path}/" "$pattern"
}

preflight_plain_wheel() {
    local pkg="$1" version="$2" url="$3"
    local pkg_re version_re cp_re pattern
    pkg_re="$(regex_escape "$pkg" | sed 's/-/[-_]/g')"
    cp_re="$(regex_escape "$PYTHON_CP_TAG")"
    if [[ -n "$version" ]]; then
        version_re="$(regex_escape "$version")"
        pattern="${pkg_re}-${version_re}((%2B|\\+)[^\"'<> ]*)?-${cp_re}-${cp_re}-[^\"'<> ]*x86_64\\.whl"
    else
        pattern="${pkg_re}-[0-9][^\"'<> ]*-${cp_re}-${cp_re}-[^\"'<> ]*x86_64\\.whl"
    fi
    preflight_index_wheel "$pkg" "$version" "$url" "$pattern"
}

preflight_all() {
    require_cmd curl
    require_cmd git
    preflight_pytorch_wheel torch "$TORCH_VERSION" torch
    preflight_pytorch_wheel torchvision "$TORCHVISION_VERSION" torchvision
    preflight_plain_wheel xformers "$XFORMERS_VERSION" "${TORCH_INDEX_URL}/xformers/"
    preflight_plain_wheel "$SPCONV_PACKAGE" "" "https://pypi.org/simple/${SPCONV_PACKAGE}/"
    preflight_plain_wheel kaolin "" "$KAOLIN_FIND_LINKS"

    [[ -f "${REPO_DIR}/setup.py" ]] || die "missing setup.py"
    [[ -d "${REPO_DIR}/symtrellis" ]] || die "missing symtrellis/"
    [[ -d "${REPO_DIR}/inference" ]] || die "missing inference/"
    git -C "$REPO_DIR" rev-parse --verify HEAD >/dev/null || die "git HEAD is required for source path validation"
    for path in setup.py symtrellis inference; do
        [[ -n "$(git -C "$REPO_DIR" ls-tree -r --name-only HEAD -- "$path")" ]] || die "$path is not tracked in git HEAD"
    done
    mkdir -p "${REPO_DIR}/${OUTPUT_DIR}"
}

emit_clean() {
    cat <<'EOF'
 && (python -m pip cache purge || true) \
 && rm -rf /root/.cache/pip /root/.cache/torch_extensions /tmp/* /var/tmp/*

EOF
}

emit_ext_clean() {
    cat <<'EOF'
 && (python -m pip cache purge || true) \
 && rm -rf /tmp/extensions /root/.cache/pip /root/.cache/torch_extensions /tmp/* /var/tmp/*

EOF
}

emit_build_env() {
    cat <<'EOF'
RUN export BUILD_JOBS="${MAX_JOBS:-4}" \
 && export MAX_JOBS="$BUILD_JOBS" CMAKE_BUILD_PARALLEL_LEVEL="$BUILD_JOBS" MAKEFLAGS="-j${BUILD_JOBS}" \
EOF
}

emit_header() {
    cat <<EOF
# syntax=docker/dockerfile:1.7
# Generated by env/generate_dockerfile.sh
# CUDA=${CUDA_MINOR_VERSION}, torch=${TORCH_VERSION}, torchvision=${TORCHVISION_VERSION}, python=${PYTHON_VERSION}

FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG BLENDER_VERSION=${BLENDER_VERSION}
ARG BLENDER_SERIES=${BLENDER_SERIES}
ARG MAX_JOBS=4

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:\$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

EOF
}

emit_base_stage() {
    cat <<EOF
# Stage 1: base OS packages, Python, compiler toolchain, and official Blender binary.
RUN apt-get update \\
 && apt-get install -y --no-install-recommends \\
    ca-certificates curl wget git build-essential cmake ninja-build pkg-config \\
    software-properties-common xz-utils libx11-6 libxext6 libxi6 libxrender1 \\
    libxfixes3 libxxf86vm1 libxrandr2 libxinerama1 libxcursor1 libgl1 \\
    libglib2.0-0 libjpeg-dev libxkbcommon0 libsm6 python3-cairo \\
 && if [[ "${PYTHON_VERSION}" != "3.10" ]]; then add-apt-repository -y ppa:deadsnakes/ppa && apt-get update; fi \\
 && apt-get install -y --no-install-recommends python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \\
 && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python \\
 && curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \\
 && python /tmp/get-pip.py \\
 && python -m pip install --no-cache-dir --upgrade pip setuptools wheel \\
 && (python -m pip cache purge || true) \\
 && mkdir -p /opt/blender-\${BLENDER_VERSION} \\
 && wget -qO /tmp/blender.tar.xz https://download.blender.org/release/Blender\${BLENDER_SERIES}/blender-\${BLENDER_VERSION}-linux-x64.tar.xz \\
 && tar -xJf /tmp/blender.tar.xz -C /opt/blender-\${BLENDER_VERSION} --strip-components=1 \\
 && ln -sf /opt/blender-\${BLENDER_VERSION}/blender /usr/local/bin/blender \\
 && rm -rf /root/.cache/pip /tmp/* /var/tmp/* /var/lib/apt/lists/*

EOF
}

emit_torch_stage() {
    cat <<EOF
# Stage 2: PyTorch stack. The wheel combination was preflighted by the generator.
RUN python -m pip install --no-cache-dir \\
    torch==${TORCH_VERSION} \\
    torchvision==${TORCHVISION_VERSION} \\
    --index-url ${TORCH_INDEX_URL} \\
EOF
    emit_clean
}

emit_pure_python_stage() {
    cat <<'EOF'
# Stage 3: pure Python and guaranteed-binary runtime/tooling packages.
RUN python -m pip install --no-cache-dir --ignore-installed blinker \
 && (python -m pip cache purge || true) \
 && rm -rf /root/.cache/pip /root/.cache/torch_extensions /tmp/* /var/tmp/*

RUN python -m pip install --no-cache-dir --only-binary=:all: \
EOF
    emit_shell_args "${PURE_PYTHON_PACKAGES[@]}"
    cat <<'EOF'
 && (python -m pip cache purge || true) \
 && rm -rf /root/.cache/pip /root/.cache/torch_extensions /tmp/* /var/tmp/*

RUN python -m pip install --no-cache-dir \
EOF
    emit_shell_args "${VCS_PYTHON_PACKAGES[@]}"
    cat <<'EOF'
 && (python -m pip cache purge || true) \
 && rm -rf /root/.cache/pip /root/.cache/torch_extensions /tmp/* /var/tmp/*

EOF
}

emit_cuda_wheel_stage() {
    cat <<EOF
# Stage 4: CUDA/Torch packages that must install from wheels only.
RUN python -m pip install --no-cache-dir --only-binary=:all: --no-deps \\
    xformers==${XFORMERS_VERSION} \\
    --index-url ${TORCH_INDEX_URL} \\
EOF
    emit_clean
    cat <<EOF
RUN python -m pip install --no-cache-dir --only-binary=:all: \\
    ${SPCONV_PACKAGE} \\
EOF
    emit_clean
}

emit_flash_attn_stage() {
    cat <<EOF
# Stage 5: flash-attn source build from official release ${FLASH_ATTN_RELEASE_TAG}.
RUN export BUILD_JOBS="\${MAX_JOBS:-4}" \\
 && export MAX_JOBS="\$BUILD_JOBS" CMAKE_BUILD_PARALLEL_LEVEL="\$BUILD_JOBS" MAKEFLAGS="-j\${BUILD_JOBS}" \\
 && python -c 'import torch; print("torch", torch.__version__, "cxx11_abi", torch._C._GLIBCXX_USE_CXX11_ABI)' \\
 && git clone --depth 1 --branch "${FLASH_ATTN_RELEASE_TAG}" --recursive --shallow-submodules \\
    "https://github.com/${FLASH_ATTN_REPO}.git" /tmp/extensions/flash-attention \\
 && FLASH_ATTENTION_FORCE_BUILD=TRUE python -m pip install --no-cache-dir --no-deps --no-build-isolation /tmp/extensions/flash-attention \\
EOF
    emit_ext_clean
}

emit_pip_build_stage() {
    local stage="$1" label="$2" requirement="$3"
    cat <<EOF
# Stage ${stage}: ${label}.
EOF
    emit_build_env
    cat <<EOF
 && python -m pip install --no-cache-dir --no-build-isolation \\
    ${requirement} \\
EOF
    emit_clean
}

emit_git_build_stage() {
    local stage="$1" label="$2" clone_cmd="$3" install_path="$4" install_flags="${5:---no-build-isolation}"
    cat <<EOF
# Stage ${stage}: ${label}.
EOF
    emit_build_env
    cat <<EOF
 && ${clone_cmd} \\
 && python -m pip install --no-cache-dir ${install_path} ${install_flags} \\
EOF
    emit_ext_clean
}

emit_kaolin_stage() {
    cat <<EOF
# Stage 7: kaolin from NVIDIA wheel index. Dependencies are installed explicitly elsewhere.
RUN python -m pip install --no-cache-dir --only-binary=:all: --no-index --no-deps \\
    --find-links ${KAOLIN_FIND_LINKS} \\
    kaolin \\
EOF
    emit_clean
}

emit_ovoxel_stage() {
    if [[ -n "$OVOXEL_WHEEL_URL" ]]; then
        cat <<EOF
# Stage 14: o-voxel from quantaji/o-voxel-gpu release wheel.
RUN python -m pip install --no-cache-dir --no-deps \\
    "${OVOXEL_WHEEL_URL}" \\
EOF
        emit_clean
    else
        cat <<EOF
# Stage 14: o-voxel source build fallback from quantaji/o-voxel-gpu.
RUN export BUILD_JOBS="\${MAX_JOBS:-4}" \\
 && export MAX_JOBS="\$BUILD_JOBS" CMAKE_BUILD_PARALLEL_LEVEL="\$BUILD_JOBS" MAKEFLAGS="-j\${BUILD_JOBS}" \\
 && git clone --depth 1 --branch "${OVOXEL_RELEASE_TAG}" --recursive --shallow-submodules \\
    "https://github.com/${OVOXEL_REPO}.git" /tmp/extensions/o-voxel-gpu \\
 && python -m pip install --no-cache-dir --no-deps --no-build-isolation /tmp/extensions/o-voxel-gpu \\
EOF
        emit_ext_clean
    fi
}

emit_symtrellis_stage() {
    cat <<'EOF'
# Stage 15: install SymTRELLIS itself from the filtered Docker build context.
RUN --mount=type=bind,source=.,target=/mnt/repo,ro \
    export BUILD_JOBS="${MAX_JOBS:-4}" \
 && export MAX_JOBS="$BUILD_JOBS" CMAKE_BUILD_PARALLEL_LEVEL="$BUILD_JOBS" MAKEFLAGS="-j${BUILD_JOBS}" \
 && mkdir -p /tmp/symtrellis-src \
 && cd /mnt/repo \
 && tar -cf - setup.py symtrellis inference | tar -xf - -C /tmp/symtrellis-src \
 && cd /tmp/symtrellis-src \
 && python -m pip install --no-cache-dir . --no-build-isolation \
 && (python -m pip cache purge || true) \
 && rm -rf /tmp/symtrellis-src /root/.cache/pip /root/.cache/torch_extensions /tmp/* /var/tmp/*

WORKDIR /workspace

EOF
}

emit_verify_stage() {
    cat <<'EOF'
# Stage 16: build-time import verification that does not require a visible GPU driver.
RUN <<'EOF_VERIFY'
set -euo pipefail
cat > /tmp/verify_env.py <<'PY'
import importlib
import importlib.metadata as metadata
import torch

print("torch", torch.__version__, "cuda", torch.version.cuda)
if torch.version.cuda is None:
    raise SystemExit("torch was not built with CUDA support")

required_imports = [
EOF
    emit_python_strings "${PURE_PYTHON_IMPORTS[@]}" "${CORE_IMPORTS[@]}" "${SYMTRELLIS_IMPORTS[@]}"
    cat <<'EOF'
]

dist_only_checks = [
EOF
    emit_python_strings "${DIST_ONLY_CHECKS[@]}"
    cat <<'EOF'
]

failures = []

for name in required_imports:
    try:
        importlib.import_module(name)
        print("IMPORT_OK", name)
    except Exception as exc:
        failures.append(f"{name}: {type(exc).__name__}: {exc}")
        print("IMPORT_FAIL", failures[-1])

for name in dist_only_checks:
    try:
        dist = metadata.distribution(name)
        print("DIST_OK", name, dist.version)
    except Exception as exc:
        failures.append(f"{name}: {type(exc).__name__}: {exc}")
        print("DIST_FAIL", failures[-1])

print("SUMMARY", len(required_imports) + len(dist_only_checks) - len(failures), "ok", len(failures), "fail")
if failures:
    raise SystemExit(1)
PY
python /tmp/verify_env.py
(python -m pip cache purge || true)
rm -rf /tmp/verify_env.py /root/.cache/pip /root/.cache/torch_extensions /tmp/* /var/tmp/*
EOF_VERIFY
EOF
}

render_dockerfile() {
    {
        emit_header
        emit_base_stage
        emit_torch_stage
        emit_pure_python_stage
        emit_cuda_wheel_stage
        emit_flash_attn_stage
        emit_pip_build_stage 6 pytorch3d '"pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"'
        emit_kaolin_stage
        emit_pip_build_stage 8 gsplat '"gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7"'
        emit_git_build_stage 9 nvdiffrast "git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast" /tmp/extensions/nvdiffrast
        emit_git_build_stage 10 "nvdiffrec renderutils" "git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec" /tmp/extensions/nvdiffrec
        emit_git_build_stage 11 diffoctreerast "git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast" /tmp/extensions/diffoctreerast
        emit_git_build_stage 12 CuMesh "git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh" /tmp/extensions/CuMesh
        emit_git_build_stage 13 FlexGEMM "git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM" /tmp/extensions/FlexGEMM
        emit_ovoxel_stage
        emit_symtrellis_stage
        emit_verify_stage
    } >"$1"
}

write_dockerfile() {
    local tmp
    tmp="$(mktemp "${DOCKERFILE_PATH}.tmp.XXXXXX")"
    trap 'rm -f "$tmp"' EXIT
    render_dockerfile "$tmp"
    mv "$tmp" "$DOCKERFILE_PATH"
    trap - EXIT
}

print_summary() {
    local runtime_gpu_imports
    runtime_gpu_imports="$(python_list_literal "${RUNTIME_GPU_IMPORTS[@]}")"

    cat <<EOF
Generated Dockerfile:
  ${DOCKERFILE_PATH}

Resolved versions:
  cuda:        ${CUDA_MINOR_VERSION} (${CUDA_TAG})
  torch:       ${TORCH_VERSION}
  torchvision: ${TORCHVISION_VERSION}
  python:      ${PYTHON_VERSION} (${PYTHON_CP_TAG})
  xformers:    ${XFORMERS_VERSION}
  spconv:      ${SPCONV_PACKAGE}
  kaolin index: ${KAOLIN_FIND_LINKS}
  flash-attn:  ${FLASH_ATTN_INSTALL_MODE}
  o-voxel:     ${OVOXEL_INSTALL_MODE}
  base image:  ${BASE_IMAGE}
  blender:     ${BLENDER_VERSION}
  arch list:   ${TORCH_CUDA_ARCH_LIST}
  source install: filtered Docker context setup.py symtrellis inference

Next commands:

  DOCKER_BUILDKIT=1 docker build \\
    -f ${OUTPUT_DIR}/${IMAGE_FILE_TAG}.dockerfile \\
    -t ${IMAGE_TAG} \\
    .

  # Optional compile parallelism override:
  # DOCKER_BUILDKIT=1 docker build --build-arg MAX_JOBS=8 \\
  #   -f ${OUTPUT_DIR}/${IMAGE_FILE_TAG}.dockerfile \\
  #   -t ${IMAGE_TAG} \\
  #   .

  docker run --rm --gpus all ${IMAGE_TAG} \\
    python -c 'import importlib, torch; mods=${runtime_gpu_imports}; print("torch", torch.__version__, "cuda available", torch.cuda.is_available()); [importlib.import_module(m) for m in mods]; print("runtime gpu imports ok")'

EOF
    if [[ -n "$IMAGE_REPO" ]]; then
        printf '  docker tag %s %s\n  docker push %s\n\n' "$IMAGE_TAG" "$REMOTE_TAG" "$REMOTE_TAG"
    else
        printf '  # Upload image after setting IMAGE_REPO, for example:\n  docker tag %s %s\n  docker push %s\n\n' "$IMAGE_TAG" "$REMOTE_TAG" "$REMOTE_TAG"
    fi
    cat <<EOF
  docker save ${IMAGE_TAG} -o ${IMAGE_ARCHIVE_NAME}.tar
  apptainer build ${IMAGE_ARCHIVE_NAME}.sif docker-archive://${IMAGE_ARCHIVE_NAME}.tar
  # or:
  singularity build ${IMAGE_ARCHIVE_NAME}.sif docker-archive://${IMAGE_ARCHIVE_NAME}.tar

EOF
}

main() {
    parse_args "$@"
    derive_versions
    preflight_all
    resolve_flash_attn_plan
    resolve_ovoxel_wheel
    write_dockerfile
    print_summary
}

main "$@"
