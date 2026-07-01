#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# User-tunable defaults.
ENV_NAME="${ENV_NAME:-symtrellis}"
EXT_DIR="${EXT_DIR:-/tmp/symtrellis-conda-build}"
BLENDER_VERSION="${BLENDER_VERSION:-5.1.2}"
BLENDER_SERIES="${BLENDER_SERIES:-5.1}"
OVOXEL_VERSION="${OVOXEL_VERSION:-0.0.1}"
OVOXEL_RELEASE_TAG="${OVOXEL_RELEASE_TAG:-v${OVOXEL_VERSION}}"
OVOXEL_REPO="${OVOXEL_REPO:-quantaji/o-voxel-gpu}"
FLASH_ATTN_REPO="${FLASH_ATTN_REPO:-Dao-AILab/flash-attention}"
UTILS3D_COMMIT="${UTILS3D_COMMIT:-3913c65d81e05e47b9f367250cf8c0f7462a0900}"
MOGE_COMMIT="${MOGE_COMMIT:-a8c37341bc0325ca99b9d57981cc3bb2bd3e255b}"
PYTORCH3D_COMMIT="${PYTORCH3D_COMMIT:-75ebeeaea0908c5527e7b1e305fbc7681382db47}"
GSPLAT_COMMIT="${GSPLAT_COMMIT:-2323de5905d5e90e035f792fe65bad0fedd413e7}"
NVDIFFRAST_REF="${NVDIFFRAST_REF:-v0.4.0}"
NVDIFFREC_REF="${NVDIFFREC_REF:-renderutils}"

# Derived version plan.
CUDA_VERSION=""
CUDA_MINOR_VERSION=""
CUDA_LABEL_VERSION=""
CUDA_TAG=""
TORCH_VERSION=""
TORCHVISION_VERSION=""
PYTHON_VERSION=""
PYTHON_CP_TAG=""
TORCH_INDEX_URL=""
SPCONV_PACKAGE=""
XFORMERS_VERSION="${XFORMERS_VERSION:-}"
KAOLIN_FIND_LINKS="${KAOLIN_FIND_LINKS:-}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}"
FLASH_ATTN_CUDA_KEY=""
FLASH_ATTN_TORCH_KEY=""
FLASH_ATTN_RELEASE_TAG="${FLASH_ATTN_RELEASE_TAG:-}"
FLASH_ATTN_INSTALL_MODE=""
OVOXEL_WHEEL_NAME=""
OVOXEL_WHEEL_URL=""
OVOXEL_INSTALL_MODE=""

# Resume state, initialized after conda activation.
SETUP_STATE_DIR=""
PLAN_FILE=""
STAGE_DIR=""
LIST_STAGES=false
FROM_STAGE=""
ONLY_STAGE=""
FORCE_CURRENT_STAGE=false

CONDA_FORGE_PACKAGES=(gcc=11 gxx=11 cmake ninja pkg-config make git curl)
NVIDIA_CUDA_PACKAGES=(
    cuda-nvcc cuda-cudart-dev cuda-driver-dev cuda-cccl cuda-nvrtc-dev cuda-nvtx
    libcusparse-dev libcublas-dev libcusolver-dev
)

PURE_PYTHON_PACKAGES=(
    black ipykernel notebook jupyterlab numpy scipy pandas tqdm pillow
    imageio imageio-ffmpeg opencv-python-headless trimesh open3d pymeshfix
    pyvista xatlas "huggingface_hub[cli]" transformers safetensors easydict
    tensorboard lpips rembg open_clip_torch objaverse astor onnxruntime optree roma
    point-cloud-utils seaborn==0.13.2 gradio==5.49.0 matplotlib plotly
    kornia timm zstandard einops iopath scikit-image plyfile pygltflib
    ipycanvas ipyevents usd-core warp-lang
)

PURE_PYTHON_IMPORTS=(
    numpy scipy PIL imageio cv2 trimesh open3d xatlas huggingface_hub
    transformers safetensors easydict lpips rembg open_clip onnxruntime
    optree point_cloud_utils kornia timm skimage utils3d
)

STAGES=(
    00-conda-toolchain
    01-blender
    02-torch
    03-pure-python
    04-cuda-wheels
    05-flash-attn
    06-pytorch3d
    07-kaolin
    08-gsplat
    09-nvdiffrast
    10-nvdiffrec-render
    11-diffoctreerast
    12-cumesh
    13-flexgemm
    14-ovoxel
    15-symtrellis
)

usage() {
    cat <<'EOF'
Usage:
  bash env/setup_conda.sh --python PYTHON_VERSION --torch TORCH_VERSION --cuda CUDA_VERSION
  bash env/setup_conda.sh --list-stages

Example:
  bash env/setup_conda.sh --python 3.10 --torch 2.7.0 --cuda 12.8
  bash env/setup_conda.sh --python 3.10 --torch 2.7.0 --cuda 12.8 --from-stage 05-flash-attn

What it does:
  - validates the torch/torchvision/xformers/spconv/kaolin wheel combination first
  - resolves the official flash-attn release tag, then forces a source build
  - creates a new conda environment named symtrellis by default
  - resumes a matching existing environment from the first incomplete stage
  - installs a minimal conda CUDA build toolchain, not full cuda or cuda-toolkit
  - installs each required source-built CUDA extension as a separate step
  - installs this repository with `pip install . --no-build-isolation`, not editable mode
  - can force reinstall from one stage or only one stage

Environment overrides:
  ENV_NAME                Conda environment name. Default: symtrellis
  EXT_DIR                 Temporary build directory. Must be under /tmp. Default: /tmp/symtrellis-conda-build
  BLENDER_VERSION         Official Blender binary version. Default: 5.1.2
  BLENDER_SERIES          Blender release series URL segment. Default: 5.1
  OVOXEL_VERSION          o-voxel package version. Default: 0.0.1
  OVOXEL_RELEASE_TAG      o-voxel GitHub release tag. Default: v0.0.1
  OVOXEL_REPO             o-voxel GitHub repository. Default: quantaji/o-voxel-gpu
  FLASH_ATTN_REPO         flash-attn GitHub repository. Default: Dao-AILab/flash-attention
  FLASH_ATTN_RELEASE_TAG  Override flash-attn release tag, e.g. v2.7.4.post1
  XFORMERS_VERSION        Override the torch-derived xformers version.
  KAOLIN_FIND_LINKS       Override the NVIDIA kaolin wheel index URL.
  TORCH_CUDA_ARCH_LIST    Override the generated CUDA architecture list.
  MAX_JOBS                Optional compile parallelism cap. Default: 4.

Notes:
  - --from-stage STAGE forces STAGE and all later stages to reinstall.
  - --only-stage STAGE forces only STAGE to reinstall and skips the final full import check.
  - Existing conda environments are resumed only when their saved plan matches.
  - Resume state is stored under $CONDA_PREFIX/.symtrellis-setup and is removed with the env.
  - hydra-core and omegaconf are intentionally not installed.
  - pip is always run with --no-cache-dir and its script-local cache is purged.
EOF
}

die() {
    echo "error: $*" >&2
    exit 2
}

section() {
    printf '\n==> %s\n' "$*"
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "$1 is required but was not found in PATH"
}

join_by() {
    local sep="$1" first=1 item
    shift
    for item in "$@"; do
        if [[ "$first" -eq 1 ]]; then
            printf '%s' "$item"
            first=0
        else
            printf '%s%s' "$sep" "$item"
        fi
    done
}

regex_escape() {
    printf '%s' "$1" | sed 's/[][\\.^$*+?{}|()]/\\&/g'
}

safe_stage_name() {
    echo "${1//[^A-Za-z0-9_.-]/_}"
}

print_stages() {
    printf '%s\n' "${STAGES[@]}"
}

stage_exists() {
    local stage
    for stage in "${STAGES[@]}"; do
        [[ "$stage" == "$1" ]] && return 0
    done
    return 1
}

require_stage() {
    local stage="$1"
    if stage_exists "$stage"; then
        return 0
    fi

    echo "error: unknown stage: ${stage}" >&2
    echo "valid stages:" >&2
    printf '  %s\n' "${STAGES[@]}" >&2
    exit 2
}

cuda_minor_version() {
    local major minor patch
    IFS='.' read -r major minor patch <<<"$1"
    [[ -n "${major:-}" && -n "${minor:-}" ]] || die "CUDA version must look like 12.8 or 12.8.0: $1"
    echo "${major}.${minor}"
}

cuda_label_version() {
    local minor
    minor="$(cuda_minor_version "$1")"
    echo "${minor}.0"
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

supported_cuda_or_die() {
    case "$1" in
    11.8 | 12.1 | 12.4 | 12.6 | 12.8 | 12.9) ;;
    *) die "Unsupported CUDA version: $1. Add a mapping in supported_cuda_or_die()." ;;
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

flash_attn_tag_compatible_with_xformers() {
    local tag="$1" min_version max_version
    local major minor patch min_major min_minor min_patch max_major max_minor max_patch
    local version_score min_score max_score

    [[ "$tag" =~ ^v([0-9]+)\.([0-9]+)\.([0-9]+) ]] || return 1
    major="${BASH_REMATCH[1]}"
    minor="${BASH_REMATCH[2]}"
    patch="${BASH_REMATCH[3]}"

    case "$XFORMERS_VERSION" in
    0.0.25.post1 | 0.0.26.post1) min_version="2.5.2"; max_version="2.5.6" ;;
    0.0.27 | 0.0.27.post2) min_version="2.5.7"; max_version="2.5.7" ;;
    0.0.28 | 0.0.28.post2 | 0.0.28.post3) min_version="2.6.3"; max_version="2.6.3" ;;
    0.0.29.post3) min_version="2.7.1"; max_version="2.7.2" ;;
    0.0.30) min_version="2.7.1"; max_version="2.7.4" ;;
    0.0.31.post1) min_version="2.7.1"; max_version="2.8.0" ;;
    0.0.32.post2) min_version="2.7.1"; max_version="2.8.2" ;;
    0.0.33.post1 | 0.0.33.post2 | 0.0.34 | 0.0.35) min_version="2.7.1"; max_version="2.8.4" ;;
    *) die "Unknown flash-attn compatibility range for xformers $XFORMERS_VERSION" ;;
    esac

    IFS='.' read -r min_major min_minor min_patch <<<"$min_version"
    IFS='.' read -r max_major max_minor max_patch <<<"$max_version"
    version_score=$((10#$major * 10000 + 10#$minor * 100 + 10#$patch))
    min_score=$((10#$min_major * 10000 + 10#$min_minor * 100 + 10#$min_patch))
    max_score=$((10#$max_major * 10000 + 10#$max_minor * 100 + 10#$max_patch))

    ((version_score >= min_score && version_score <= max_score))
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
        --list-stages)
            LIST_STAGES=true
            shift
            ;;
        --from-stage | --only-stage)
            [[ $# -ge 2 && "$2" != --* ]] || die "$1 requires a value"
            case "$1" in
            --from-stage)
                [[ -z "$FROM_STAGE" ]] || die "--from-stage was specified more than once"
                FROM_STAGE="$2"
                ;;
            --only-stage)
                [[ -z "$ONLY_STAGE" ]] || die "--only-stage was specified more than once"
                ONLY_STAGE="$2"
                ;;
            esac
            shift 2
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
        --from-stage=* | --only-stage=*)
            local key="${1%%=*}" value="${1#*=}"
            [[ -n "$value" ]] || die "$key requires a value"
            case "$key" in
            --from-stage)
                [[ -z "$FROM_STAGE" ]] || die "--from-stage was specified more than once"
                FROM_STAGE="$value"
                ;;
            --only-stage)
                [[ -z "$ONLY_STAGE" ]] || die "--only-stage was specified more than once"
                ONLY_STAGE="$value"
                ;;
            esac
            shift
            ;;
        *) die "unknown argument: $1" ;;
        esac
    done

    if [[ "$LIST_STAGES" == true ]]; then
        print_stages
        exit 0
    fi

    [[ -z "$FROM_STAGE" || -z "$ONLY_STAGE" ]] || die "--from-stage and --only-stage cannot be used together"
    [[ -n "$PYTHON_VERSION" ]] || die "missing required argument: --python"
    [[ -n "$TORCH_VERSION" ]] || die "missing required argument: --torch"
    [[ -n "$CUDA_VERSION" ]] || die "missing required argument: --cuda"
    [[ -z "$FROM_STAGE" ]] || require_stage "$FROM_STAGE"
    [[ -z "$ONLY_STAGE" ]] || require_stage "$ONLY_STAGE"

    CUDA_MINOR_VERSION="$(cuda_minor_version "$CUDA_VERSION")"
    supported_cuda_or_die "$CUDA_MINOR_VERSION"
}

derive_versions() {
    CUDA_LABEL_VERSION="$(cuda_label_version "$CUDA_VERSION")"
    CUDA_TAG="$(torch_cuda_tag "$CUDA_VERSION")"
    PYTHON_CP_TAG="$(python_cp_tag "$PYTHON_VERSION")"
    TORCHVISION_VERSION="$(torchvision_for_torch "$TORCH_VERSION")"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/${CUDA_TAG}"
    SPCONV_PACKAGE="$(spconv_pkg_for_cuda "$CUDA_MINOR_VERSION")"
    XFORMERS_VERSION="${XFORMERS_VERSION:-$(xformers_for_torch "$TORCH_VERSION")}"
    KAOLIN_FIND_LINKS="${KAOLIN_FIND_LINKS:-https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-${TORCH_VERSION}_${CUDA_TAG}.html}"
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(default_arch_list_for_torch_cuda "$TORCH_VERSION" "$CUDA_MINOR_VERSION")}"
    FLASH_ATTN_CUDA_KEY="cu${CUDA_MINOR_VERSION%%.*}"
    FLASH_ATTN_TORCH_KEY="torch${TORCH_VERSION%.*}"
    OVOXEL_WHEEL_NAME="o_voxel-${OVOXEL_VERSION}+torch${TORCH_VERSION}.${CUDA_TAG}-${PYTHON_CP_TAG}-${PYTHON_CP_TAG}-linux_x86_64.whl"
}

conda_env_exists() {
    conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"
}

preflight_paths() {
    [[ -f "${REPO_DIR}/setup.py" ]] || die "missing setup.py"
    [[ -d "${REPO_DIR}/symtrellis" ]] || die "missing symtrellis/"
    [[ "$EXT_DIR" == /tmp/* ]] || die "EXT_DIR must be under /tmp: $EXT_DIR"
}

preflight_tools() {
    require_cmd conda
    require_cmd curl
    require_cmd git
    require_cmd tar
    require_cmd sed
    require_cmd grep
    require_cmd awk
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

preflight_wheels() {
    section "Preflight Python/CUDA wheel availability"
    preflight_pytorch_wheel torch "$TORCH_VERSION" torch
    preflight_pytorch_wheel torchvision "$TORCHVISION_VERSION" torchvision
    preflight_plain_wheel xformers "$XFORMERS_VERSION" "${TORCH_INDEX_URL}/xformers/"
    preflight_plain_wheel "$SPCONV_PACKAGE" "" "https://pypi.org/simple/${SPCONV_PACKAGE}/"
    preflight_plain_wheel kaolin "" "$KAOLIN_FIND_LINKS"
}

github_release_asset_url() {
    local encoded_name="${3//+/%2B}"
    echo "https://github.com/${1}/releases/download/${2}/${encoded_name}"
}

resolve_flash_attn_plan() {
    local api json current_tag="" current_prerelease=0 fallback_tag="" exact=0
    local asset exact_base exact_py
    section "Resolve flash-attn source release"
    exact_base="${FLASH_ATTN_CUDA_KEY}${FLASH_ATTN_TORCH_KEY}cxx11abi"
    exact_py="-${PYTHON_CP_TAG}-${PYTHON_CP_TAG}-linux_x86_64.whl"

    if [[ -n "$FLASH_ATTN_RELEASE_TAG" ]]; then
        if ! flash_attn_tag_compatible_with_xformers "$FLASH_ATTN_RELEASE_TAG"; then
            cat >&2 <<EOF
FLASH_ATTN_RELEASE_TAG=${FLASH_ATTN_RELEASE_TAG} is incompatible with xformers ${XFORMERS_VERSION}
EOF
            exit 2
        fi

        api="https://api.github.com/repos/${FLASH_ATTN_REPO}/releases/tags/${FLASH_ATTN_RELEASE_TAG}"
        json="$(mktemp)"
        if ! curl -fsSL "$api" -o "$json"; then
            rm -f "$json"
            die "flash-attn release tag not found: ${FLASH_ATTN_RELEASE_TAG}"
        fi
        while IFS= read -r line; do
            [[ "$line" == *'"name": "flash_attn-'* ]] || continue
            asset="$(printf '%s\n' "$line" | sed -E 's/.*"name": "([^"]+)".*/\1/')"
            if [[ "$asset" == flash_attn-*+"${exact_base}"FALSE"${exact_py}" ||
                "$asset" == flash_attn-*+"${exact_base}"TRUE"${exact_py}" ]]; then
                exact=1
                break
            fi
        done <"$json"
        rm -f "$json"

        if [[ "$exact" -ne 1 ]]; then
            cat >&2 <<EOF
flash-attn release asset not found:
  tag:       ${FLASH_ATTN_RELEASE_TAG}
  cuda key:  ${FLASH_ATTN_CUDA_KEY}
  torch key: ${FLASH_ATTN_TORCH_KEY}
  python:    ${PYTHON_VERSION} (${PYTHON_CP_TAG})
EOF
            exit 2
        fi

        FLASH_ATTN_INSTALL_MODE="source build ${FLASH_ATTN_RELEASE_TAG} (user-specified; ${FLASH_ATTN_CUDA_KEY}, ${FLASH_ATTN_TORCH_KEY}, ${PYTHON_CP_TAG}; compatible with xformers ${XFORMERS_VERSION})"
        return
    fi

    api="https://api.github.com/repos/${FLASH_ATTN_REPO}/releases?per_page=100"
    json="$(mktemp)"
    curl -fsSL "$api" -o "$json"

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
        flash_attn_tag_compatible_with_xformers "$current_tag" || continue

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
  checked:   ${api}
EOF
        exit 2
    fi
}

resolve_ovoxel_wheel() {
    local api json
    section "Resolve o-voxel release wheel"
    api="https://api.github.com/repos/${OVOXEL_REPO}/releases/tags/${OVOXEL_RELEASE_TAG}"
    json="$(mktemp)"
    OVOXEL_WHEEL_URL=""
    OVOXEL_INSTALL_MODE="fallback source build from ${OVOXEL_REPO}@${OVOXEL_RELEASE_TAG}"
    if curl -fsSL "$api" -o "$json" && grep -Fq "$OVOXEL_WHEEL_NAME" "$json"; then
        OVOXEL_WHEEL_URL="$(github_release_asset_url "$OVOXEL_REPO" "$OVOXEL_RELEASE_TAG" "$OVOXEL_WHEEL_NAME")"
        OVOXEL_INSTALL_MODE="release wheel ${OVOXEL_WHEEL_NAME}"
    fi
    rm -f "$json"
}

symtrellis_git_head() {
    git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo "unknown"
}

plan_content() {
    cat <<EOF
SETUP_SCHEMA=1
ENV_NAME=${ENV_NAME}
PYTHON_VERSION=${PYTHON_VERSION}
PYTHON_CP_TAG=${PYTHON_CP_TAG}
CUDA_MINOR_VERSION=${CUDA_MINOR_VERSION}
CUDA_LABEL_VERSION=${CUDA_LABEL_VERSION}
CUDA_TAG=${CUDA_TAG}
TORCH_VERSION=${TORCH_VERSION}
TORCHVISION_VERSION=${TORCHVISION_VERSION}
TORCH_INDEX_URL=${TORCH_INDEX_URL}
XFORMERS_VERSION=${XFORMERS_VERSION}
SPCONV_PACKAGE=${SPCONV_PACKAGE}
KAOLIN_FIND_LINKS=${KAOLIN_FIND_LINKS}
TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
BLENDER_VERSION=${BLENDER_VERSION}
BLENDER_SERIES=${BLENDER_SERIES}
FLASH_ATTN_REPO=${FLASH_ATTN_REPO}
FLASH_ATTN_RELEASE_TAG=${FLASH_ATTN_RELEASE_TAG}
OVOXEL_REPO=${OVOXEL_REPO}
OVOXEL_VERSION=${OVOXEL_VERSION}
OVOXEL_RELEASE_TAG=${OVOXEL_RELEASE_TAG}
OVOXEL_WHEEL_NAME=${OVOXEL_WHEEL_NAME}
OVOXEL_WHEEL_URL=${OVOXEL_WHEEL_URL}
UTILS3D_COMMIT=${UTILS3D_COMMIT}
MOGE_COMMIT=${MOGE_COMMIT}
PYTORCH3D_COMMIT=${PYTORCH3D_COMMIT}
GSPLAT_COMMIT=${GSPLAT_COMMIT}
NVDIFFRAST_REF=${NVDIFFRAST_REF}
NVDIFFREC_REF=${NVDIFFREC_REF}
EOF
}

stage_stamp_content() {
    local stage="$1" fingerprint="$2"
    plan_content
    printf 'STAGE_SCHEMA=1\n'
    printf 'STAGE=%s\n' "$stage"
    printf '%s\n' "$fingerprint"
}

stage_stamp_path() {
    echo "${STAGE_DIR}/$(safe_stage_name "$1").done"
}

init_setup_state() {
    local expected
    SETUP_STATE_DIR="${CONDA_PREFIX}/.symtrellis-setup"
    PLAN_FILE="${SETUP_STATE_DIR}/plan.env"
    STAGE_DIR="${SETUP_STATE_DIR}/stages"
    mkdir -p "$STAGE_DIR"

    expected="$(mktemp)"
    plan_content >"$expected"
    if [[ -f "$PLAN_FILE" ]]; then
        if ! cmp -s "$PLAN_FILE" "$expected"; then
            cat >&2 <<EOF
error: existing conda environment has a different SymTRELLIS setup plan:
  env:      ${ENV_NAME}
  plan:     ${PLAN_FILE}
  expected: ${expected}

Remove the environment explicitly or use a different ENV_NAME.
EOF
            exit 2
        fi
        rm -f "$expected"
        return
    fi

    mv "$expected" "$PLAN_FILE"
}

run_stage() {
    local stage="$1" verify_fn="$2" install_fn="$3" fingerprint="$4"
    local stamp expected
    stamp="$(stage_stamp_path "$stage")"
    expected="$(mktemp)"
    stage_stamp_content "$stage" "$fingerprint" >"$expected"

    if [[ "$FORCE_CURRENT_STAGE" != true && -f "$stamp" ]] && cmp -s "$stamp" "$expected" && "$verify_fn" >/dev/null 2>&1; then
        section "Skip ${stage}"
        rm -f "$expected"
        return 0
    fi

    if [[ "$FORCE_CURRENT_STAGE" == true ]]; then
        section "Force ${stage}"
    fi
    rm -f "$stamp"
    if ! "$install_fn"; then
        rm -f "$expected"
        return 1
    fi
    if ! "$verify_fn"; then
        rm -f "$expected"
        return 1
    fi
    mv "$expected" "$stamp"
}

run_registered_stage() {
    local stage="$1"
    case "$stage" in
    00-conda-toolchain)
        run_stage "$stage" verify_toolchain install_toolchain \
            "CONDA_FORGE_PACKAGES=$(join_by ' ' "${CONDA_FORGE_PACKAGES[@]}")"$'\n'"NVIDIA_CUDA_PACKAGES=$(join_by ' ' "${NVIDIA_CUDA_PACKAGES[@]}")"
        ;;
    01-blender)
        run_stage "$stage" verify_blender install_blender \
            "BLENDER_VERSION=${BLENDER_VERSION}"$'\n'"BLENDER_SERIES=${BLENDER_SERIES}"
        ;;
    02-torch)
        run_stage "$stage" verify_torch_stack install_torch_stack \
            "TORCH=${TORCH_VERSION}"$'\n'"TORCHVISION=${TORCHVISION_VERSION}"$'\n'"CUDA=${CUDA_MINOR_VERSION}"
        ;;
    03-pure-python)
        run_stage "$stage" verify_pure_python_stack install_pure_python_stack \
            "PURE_PYTHON_PACKAGES=$(join_by ' ' "${PURE_PYTHON_PACKAGES[@]}")"$'\n'"UTILS3D_COMMIT=${UTILS3D_COMMIT}"$'\n'"MOGE_COMMIT=${MOGE_COMMIT}"
        ;;
    04-cuda-wheels)
        run_stage "$stage" verify_cuda_wheels install_cuda_wheels \
            "XFORMERS_VERSION=${XFORMERS_VERSION}"$'\n'"SPCONV_PACKAGE=${SPCONV_PACKAGE}"$'\n'"TORCH_INDEX_URL=${TORCH_INDEX_URL}"
        ;;
    05-flash-attn)
        run_stage "$stage" verify_flash_attn install_flash_attn \
            "FLASH_ATTN_REPO=${FLASH_ATTN_REPO}"$'\n'"FLASH_ATTN_RELEASE_TAG=${FLASH_ATTN_RELEASE_TAG}"$'\n'"FLASH_ATTENTION_FORCE_BUILD=TRUE"
        ;;
    06-pytorch3d)
        run_stage "$stage" verify_pytorch3d install_pytorch3d \
            "PYTORCH3D_COMMIT=${PYTORCH3D_COMMIT}"
        ;;
    07-kaolin)
        run_stage "$stage" verify_kaolin install_kaolin \
            "KAOLIN_FIND_LINKS=${KAOLIN_FIND_LINKS}"
        ;;
    08-gsplat)
        run_stage "$stage" verify_gsplat install_gsplat \
            "GSPLAT_COMMIT=${GSPLAT_COMMIT}"
        ;;
    09-nvdiffrast)
        run_stage "$stage" verify_nvdiffrast install_nvdiffrast \
            "NVDIFFRAST_REF=${NVDIFFRAST_REF}"
        ;;
    10-nvdiffrec-render)
        run_stage "$stage" verify_nvdiffrec install_nvdiffrec \
            "NVDIFFREC_REF=${NVDIFFREC_REF}"
        ;;
    11-diffoctreerast)
        run_stage "$stage" verify_diffoctreerast install_diffoctreerast \
            "REPO=https://github.com/JeffreyXiang/diffoctreerast.git"
        ;;
    12-cumesh)
        run_stage "$stage" verify_cumesh install_cumesh \
            "REPO=https://github.com/JeffreyXiang/CuMesh.git"
        ;;
    13-flexgemm)
        run_stage "$stage" verify_flexgemm install_flexgemm \
            "REPO=https://github.com/JeffreyXiang/FlexGEMM.git"
        ;;
    14-ovoxel)
        run_stage "$stage" verify_ovoxel install_ovoxel \
            "OVOXEL_REPO=${OVOXEL_REPO}"$'\n'"OVOXEL_RELEASE_TAG=${OVOXEL_RELEASE_TAG}"$'\n'"OVOXEL_WHEEL_URL=${OVOXEL_WHEEL_URL}"
        ;;
    15-symtrellis)
        run_stage "$stage" verify_symtrellis install_symtrellis \
            "SYMTRELLIS_GIT_HEAD=$(symtrellis_git_head)"
        ;;
    *) die "unknown stage: $stage" ;;
    esac
}

run_all_stages() {
    local stage force_from=false

    if [[ -n "$ONLY_STAGE" ]]; then
        FORCE_CURRENT_STAGE=true
        run_registered_stage "$ONLY_STAGE"
        FORCE_CURRENT_STAGE=false
        return
    fi

    for stage in "${STAGES[@]}"; do
        if [[ -n "$FROM_STAGE" && "$stage" == "$FROM_STAGE" ]]; then
            force_from=true
        fi
        FORCE_CURRENT_STAGE="$force_from"
        run_registered_stage "$stage"
    done
    FORCE_CURRENT_STAGE=false
}

reset_ext_dir() {
    [[ "$EXT_DIR" == /tmp/* ]] || die "refusing to clean unsafe EXT_DIR: $EXT_DIR"
    rm -rf "$EXT_DIR"
    mkdir -p "$EXT_DIR"
}

pip_cleanup() {
    python -m pip cache purge >/dev/null 2>&1 || true
    if [[ -n "${PIP_CACHE_DIR:-}" && "$PIP_CACHE_DIR" == /tmp/* ]]; then
        rm -rf "$PIP_CACHE_DIR"
    fi
    if [[ -n "${TORCH_EXTENSIONS_DIR:-}" && "$TORCH_EXTENSIONS_DIR" == /tmp/* ]]; then
        rm -rf "$TORCH_EXTENSIONS_DIR"
    fi
}

pip_install_command() {
    python -m pip install --no-cache-dir "$@"
}

pip_install() {
    pip_install_command "$@"
    pip_cleanup
}

prepare_build_env() {
    local jobs
    jobs="${MAX_JOBS:-4}"
    export BUILD_JOBS="$jobs"
    export MAX_JOBS="$jobs"
    export CMAKE_BUILD_PARALLEL_LEVEL="$jobs"
    export MAKEFLAGS="-j${jobs}"
    export TORCH_EXTENSIONS_DIR="${EXT_DIR}/torch_extensions"
    mkdir -p "$TORCH_EXTENSIONS_DIR"
}

write_activation_hook() {
    local hook_dir hook
    hook_dir="${CONDA_PREFIX}/etc/conda/activate.d"
    hook="${hook_dir}/symtrellis.sh"
    mkdir -p "$hook_dir"
    {
        printf '%s\n' 'export CC="${CONDA_PREFIX}/bin/gcc"'
        printf '%s\n' 'export CXX="${CONDA_PREFIX}/bin/g++"'
        printf '%s\n' 'export CUDA_HOST_COMPILER="${CONDA_PREFIX}/bin/g++"'
        printf '%s\n' 'export CUDAHOSTCXX="${CONDA_PREFIX}/bin/g++"'
        printf '%s\n' 'export CUDACXX="${CONDA_PREFIX}/bin/nvcc"'
        printf '%s\n' 'export CUDA_HOME="${CONDA_PREFIX}"'
        printf '%s\n' 'export CUDA_PATH="${CONDA_PREFIX}"'
        printf '%s\n' 'export FORCE_CUDA=1'
        printf '%s\n' 'export BUILD_WITH_CUDA=1'
        printf '%s\n' 'export PIP_NO_CACHE_DIR=1'
        printf '%s\n' 'export PIP_DISABLE_PIP_VERSION_CHECK=1'
        printf 'export TORCH_CUDA_ARCH_LIST=%q\n' "$TORCH_CUDA_ARCH_LIST"
        printf '%s\n' 'export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib64:${LD_LIBRARY_PATH:-}"'
        printf '%s\n' 'export LIBRARY_PATH="${CONDA_PREFIX}/lib/stubs:${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"'
    } >"$hook"
}

export_current_build_env() {
    export CC="${CONDA_PREFIX}/bin/gcc"
    export CXX="${CONDA_PREFIX}/bin/g++"
    export CUDA_HOST_COMPILER="${CONDA_PREFIX}/bin/g++"
    export CUDAHOSTCXX="${CONDA_PREFIX}/bin/g++"
    export CUDACXX="${CONDA_PREFIX}/bin/nvcc"
    export CUDA_HOME="${CONDA_PREFIX}"
    export CUDA_PATH="${CONDA_PREFIX}"
    export FORCE_CUDA=1
    export BUILD_WITH_CUDA=1
    export TORCH_CUDA_ARCH_LIST
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib64:${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="${CONDA_PREFIX}/lib/stubs:${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"
    export PIP_NO_CACHE_DIR=1
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    export PIP_CACHE_DIR="${EXT_DIR}/pip-cache"
}

verify_active_python_version() {
    PYTHON_EXPECTED="$PYTHON_VERSION" python <<'PY'
import os
import sys

expected = os.environ["PYTHON_EXPECTED"]
actual = f"{sys.version_info.major}.{sys.version_info.minor}"
if actual != expected:
    raise SystemExit(f"python version mismatch: expected {expected}, got {actual}")
PY
}

prepare_conda_env() {
    eval "$(conda shell.bash hook)"
    if conda_env_exists; then
        section "Activate existing conda environment ${ENV_NAME}"
        conda activate "$ENV_NAME"
    else
        section "Create conda environment ${ENV_NAME}"
        conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}"
        conda activate "$ENV_NAME"
    fi

    verify_active_python_version
    write_activation_hook
    export_current_build_env
    init_setup_state
}

install_toolchain() {
    section "Install minimal conda compiler and CUDA build packages"
    conda install -y -n "$ENV_NAME" -c conda-forge "${CONDA_FORGE_PACKAGES[@]}"
    conda install -y -n "$ENV_NAME" -c "nvidia/label/cuda-${CUDA_LABEL_VERSION}" "${NVIDIA_CUDA_PACKAGES[@]}"
    write_activation_hook
    export_current_build_env
}

install_blender() {
    local dst tarball url
    section "Install official Blender ${BLENDER_VERSION}"
    dst="${CONDA_PREFIX}/opt/blender-${BLENDER_VERSION}"
    tarball="${EXT_DIR}/blender.tar.xz"
    url="https://download.blender.org/release/Blender${BLENDER_SERIES}/blender-${BLENDER_VERSION}-linux-x64.tar.xz"

    reset_ext_dir
    rm -rf "$dst"
    mkdir -p "${CONDA_PREFIX}/opt" "$dst"
    curl -fsSL "$url" -o "$tarball"
    tar -xJf "$tarball" -C "$dst" --strip-components=1
    ln -sf "${dst}/blender" "${CONDA_PREFIX}/bin/blender"
    reset_ext_dir
}

install_torch_stack() {
    section "Install torch and torchvision"
    pip_install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url "$TORCH_INDEX_URL"
}

install_pure_python_stack() {
    section "Install pure Python and binary-only packages"
    pip_install --upgrade pip setuptools wheel packaging
    pip_install --ignore-installed blinker
    pip_install --only-binary=:all: "${PURE_PYTHON_PACKAGES[@]}"
    pip_install \
        "git+https://github.com/EasternJournalist/utils3d.git@${UTILS3D_COMMIT}" \
        "MoGe @ git+https://github.com/microsoft/MoGe.git@${MOGE_COMMIT}"
}

install_cuda_wheels() {
    section "Install CUDA/Torch wheels"
    pip_install --only-binary=:all: --no-deps "xformers==${XFORMERS_VERSION}" --index-url "$TORCH_INDEX_URL"
    pip_install --only-binary=:all: "$SPCONV_PACKAGE"
}

install_flash_attn() {
    section "Build flash-attn from ${FLASH_ATTN_RELEASE_TAG}"
    reset_ext_dir
    prepare_build_env
    git clone --depth 1 --branch "$FLASH_ATTN_RELEASE_TAG" --recursive --shallow-submodules \
        "https://github.com/${FLASH_ATTN_REPO}.git" "${EXT_DIR}/flash-attention"
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
        pip_install_command --no-deps --no-build-isolation "${EXT_DIR}/flash-attention"
    pip_cleanup
    reset_ext_dir
}

install_pytorch3d() {
    install_pip_source_build pytorch3d "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@${PYTORCH3D_COMMIT}"
}

install_gsplat() {
    install_pip_source_build gsplat "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@${GSPLAT_COMMIT}"
}

install_pip_source_build() {
    local label="$1" requirement="$2"
    section "Build ${label}"
    prepare_build_env
    pip_install_command --no-deps --no-build-isolation "$requirement"
    pip_cleanup
}

install_git_source_build() {
    local label="$1" repo="$2" install_path="$3"
    shift 3
    section "Build ${label}"
    reset_ext_dir
    prepare_build_env
    git clone "$@" "$repo" "$install_path"
    pip_install_command --no-deps --no-build-isolation "$install_path"
    pip_cleanup
    reset_ext_dir
}

install_kaolin() {
    section "Install kaolin wheel"
    pip_install --only-binary=:all: --no-index --no-deps --find-links "$KAOLIN_FIND_LINKS" kaolin
}

install_nvdiffrast() {
    install_git_source_build nvdiffrast https://github.com/NVlabs/nvdiffrast.git "${EXT_DIR}/nvdiffrast" -b "$NVDIFFRAST_REF"
}

install_nvdiffrec() {
    install_git_source_build "nvdiffrec renderutils" https://github.com/JeffreyXiang/nvdiffrec.git "${EXT_DIR}/nvdiffrec" -b "$NVDIFFREC_REF"
}

install_diffoctreerast() {
    install_git_source_build diffoctreerast https://github.com/JeffreyXiang/diffoctreerast.git "${EXT_DIR}/diffoctreerast" --recurse-submodules
}

install_cumesh() {
    install_git_source_build CuMesh https://github.com/JeffreyXiang/CuMesh.git "${EXT_DIR}/CuMesh" --recursive
}

install_flexgemm() {
    install_git_source_build FlexGEMM https://github.com/JeffreyXiang/FlexGEMM.git "${EXT_DIR}/FlexGEMM" --recursive
}

install_ovoxel() {
    section "Install o-voxel"
    if [[ -n "$OVOXEL_WHEEL_URL" ]]; then
        pip_install --no-deps "$OVOXEL_WHEEL_URL"
        return
    fi

    reset_ext_dir
    prepare_build_env
    git clone --depth 1 --branch "$OVOXEL_RELEASE_TAG" --recursive --shallow-submodules \
        "https://github.com/${OVOXEL_REPO}.git" "${EXT_DIR}/o-voxel-gpu"
    pip_install_command --no-deps --no-build-isolation "${EXT_DIR}/o-voxel-gpu"
    pip_cleanup
    reset_ext_dir
}

install_symtrellis() {
    section "Install SymTRELLIS from current repository"
    prepare_build_env
    cd "$REPO_DIR"
    pip_install_command --no-deps --no-build-isolation .
    pip_cleanup
}

verify_imports() {
    python - "$@" <<'PY'
import importlib
import sys

for name in sys.argv[1:]:
    importlib.import_module(name)
PY
}

verify_any_distribution() {
    python - "$@" <<'PY'
import sys
from importlib.metadata import PackageNotFoundError, version

for name in sys.argv[1:]:
    try:
        version(name)
        raise SystemExit(0)
    except PackageNotFoundError:
        pass

raise SystemExit(f"none of these distributions are installed: {', '.join(sys.argv[1:])}")
PY
}

verify_toolchain() {
    [[ -x "${CONDA_PREFIX}/bin/gcc" ]]
    [[ -x "${CONDA_PREFIX}/bin/g++" ]]
    [[ -x "${CONDA_PREFIX}/bin/nvcc" ]]
    [[ -f "${CONDA_PREFIX}/include/cuda_runtime.h" ]]
    [[ -f "${CONDA_PREFIX}/include/cusparse.h" ]]
    [[ -f "${CONDA_PREFIX}/include/cublas_v2.h" ]]
    [[ -f "${CONDA_PREFIX}/include/cusolverDn.h" ]]
    [[ -f "${CONDA_PREFIX}/lib/libcudart.so" ]]
    "${CONDA_PREFIX}/bin/gcc" -dumpfullversion -dumpversion | grep -Eq '^11\.'
}

verify_blender() {
    [[ -x "${CONDA_PREFIX}/bin/blender" ]]
    [[ -d "${CONDA_PREFIX}/opt/blender-${BLENDER_VERSION}" ]]
}

verify_torch_stack() {
    PYTHON_EXPECTED="$PYTHON_VERSION" \
        TORCH_EXPECTED="$TORCH_VERSION" \
        TORCHVISION_EXPECTED="$TORCHVISION_VERSION" \
        CUDA_EXPECTED="$CUDA_MINOR_VERSION" \
        python <<'PY'
import os
import sys
import torch
import torchvision

def base_version(value: str) -> str:
    return value.split("+", 1)[0]

python_expected = os.environ["PYTHON_EXPECTED"]
python_actual = f"{sys.version_info.major}.{sys.version_info.minor}"
if python_actual != python_expected:
    raise SystemExit(f"python version mismatch: expected {python_expected}, got {python_actual}")

torch_expected = os.environ["TORCH_EXPECTED"]
if base_version(torch.__version__) != torch_expected:
    raise SystemExit(f"torch version mismatch: expected {torch_expected}, got {torch.__version__}")

torchvision_expected = os.environ["TORCHVISION_EXPECTED"]
if base_version(torchvision.__version__) != torchvision_expected:
    raise SystemExit(f"torchvision version mismatch: expected {torchvision_expected}, got {torchvision.__version__}")

cuda_expected = os.environ["CUDA_EXPECTED"]
if torch.version.cuda != cuda_expected:
    raise SystemExit(f"torch cuda mismatch: expected {cuda_expected}, got {torch.version.cuda}")
PY
}

verify_pure_python_stack() {
    verify_torch_stack
    verify_imports "${PURE_PYTHON_IMPORTS[@]}"
}

verify_cuda_wheels() {
    verify_torch_stack
    XFORMERS_EXPECTED="$XFORMERS_VERSION" python <<'PY'
import os
import spconv
import xformers

expected = os.environ["XFORMERS_EXPECTED"]
actual = xformers.__version__.split("+", 1)[0]
if actual != expected:
    raise SystemExit(f"xformers version mismatch: expected {expected}, got {xformers.__version__}")
PY
}

verify_flash_attn() {
    verify_torch_stack
    verify_imports flash_attn
}

verify_pytorch3d() {
    verify_torch_stack
    verify_imports pytorch3d
}

verify_kaolin() {
    verify_torch_stack
    verify_imports kaolin
}

verify_gsplat() {
    verify_torch_stack
    verify_imports gsplat
}

verify_nvdiffrast() {
    verify_torch_stack
    verify_imports nvdiffrast.torch
}

verify_nvdiffrec() {
    verify_torch_stack
    verify_imports nvdiffrec_render
}

verify_diffoctreerast() {
    verify_torch_stack
    verify_imports diffoctreerast
}

verify_cumesh() {
    verify_torch_stack
    verify_imports cumesh
}

verify_flexgemm() {
    verify_torch_stack
    verify_imports flex_gemm
}

verify_ovoxel() {
    verify_torch_stack
    verify_imports o_voxel
}

verify_symtrellis() {
    verify_torch_stack
    verify_imports \
        symtrellis \
        symtrellis.geometry.neighbors.sparse_lattice_ext._C \
        symtrellis.mapper.attention.csr_attn_ext._C
}

verify_install() {
    section "Verify imports that do not require a visible GPU driver"
    verify_torch_stack
    python <<'PY'
import importlib
import torch

print("torch", torch.__version__, "cuda", torch.version.cuda)
if torch.version.cuda is None:
    raise SystemExit("torch was not built with CUDA support")

modules = [
    "flash_attn",
    "xformers",
    "spconv",
    "nvdiffrast.torch",
    "nvdiffrec_render",
    "diffoctreerast",
    "cumesh",
    "symtrellis",
    "symtrellis.geometry.neighbors.sparse_lattice_ext._C",
    "symtrellis.mapper.attention.csr_attn_ext._C",
]

for name in modules:
    importlib.import_module(name)
    print("ok", name)
PY
    pip_cleanup
}

stage_mode_summary() {
    if [[ -n "$ONLY_STAGE" ]]; then
        echo "only ${ONLY_STAGE}"
    elif [[ -n "$FROM_STAGE" ]]; then
        echo "from ${FROM_STAGE}"
    else
        echo "normal"
    fi
}

print_summary() {
    cat <<EOF

Conda environment installed:
  env name:    ${ENV_NAME}
  python:      ${PYTHON_VERSION} (${PYTHON_CP_TAG})
  cuda:        ${CUDA_MINOR_VERSION} (${CUDA_TAG}; conda label cuda-${CUDA_LABEL_VERSION})
  torch:       ${TORCH_VERSION}
  torchvision: ${TORCHVISION_VERSION}
  xformers:    ${XFORMERS_VERSION}
  spconv:      ${SPCONV_PACKAGE}
  kaolin index: ${KAOLIN_FIND_LINKS}
  flash-attn:  ${FLASH_ATTN_INSTALL_MODE}
  o-voxel:     ${OVOXEL_INSTALL_MODE}
  arch list:   ${TORCH_CUDA_ARCH_LIST}
  state dir:   ${SETUP_STATE_DIR}
  stage mode:  $(stage_mode_summary)

Activate:
  conda activate ${ENV_NAME}

Optional GPU-driver import check:
  python -c 'import importlib, torch; mods=["kaolin","gsplat","flex_gemm","o_voxel"]; print("torch", torch.__version__, "cuda available", torch.cuda.is_available()); [importlib.import_module(m) for m in mods]; print("runtime gpu imports ok")'

EOF
}

main() {
    parse_args "$@"
    derive_versions
    preflight_paths
    preflight_tools
    preflight_wheels
    resolve_flash_attn_plan
    resolve_ovoxel_wheel
    prepare_conda_env
    run_all_stages
    if [[ -z "$ONLY_STAGE" ]]; then
        verify_install
    fi
    print_summary
}

main "$@"
