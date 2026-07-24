#!/usr/bin/env bash
set -euo pipefail

# Host paths and fixed container mount points.
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly WHEELS_ROOT="${REPO_DIR}/wheels"

readonly CONTAINER_REPO_DIR="/repo"
readonly CONTAINER_WHEELS_DIR="/wheels"
readonly CONTAINER_PIP_CACHE_DIR="/pip-cache"
readonly CONTAINER_BUILD_DIR="/tmp/package-build"

# Package groups also define the execution order within each stage.
readonly -a DOWNLOAD_PACKAGES=(
    cumm
    spconv
    xformers
    o-voxel-gpu
)

readonly -a PURE_PYTHON_PACKAGES=(
    utils3d
    moge
    trellis
    trellis2
    sam3d-objects
)

readonly -a NATIVE_PACKAGES=(
    flash-attn
    pytorch3d
    kaolin
    gsplat
    diff-gaussian-rasterization
    nvdiffrast
    nvdiffrec-render
    diffoctreerast
    cumesh
    flex-gemm
    symtrellis
)

# Package indexes and source repositories.
readonly PYTORCH_INDEX_BASE_URL="https://download.pytorch.org/whl"
readonly DEFAULT_SPCONV_INDEX_URL="https://ratharog.github.io/cumm-spconv/"

readonly FLASH_ATTN_REPOSITORY="Dao-AILab/flash-attention"
readonly OVOXEL_REPOSITORY="quantaji/o-voxel-gpu"
readonly UTILS3D_REPOSITORY="EasternJournalist/utils3d"
readonly MOGE_REPOSITORY="microsoft/MoGe"
readonly TRELLIS_REPOSITORY="microsoft/TRELLIS"
readonly TRELLIS2_REPOSITORY="microsoft/TRELLIS.2"
readonly SAM3D_OBJECTS_REPOSITORY="facebookresearch/sam-3d-objects"
readonly PYTORCH3D_REPOSITORY="facebookresearch/pytorch3d"
readonly KAOLIN_REPOSITORY="NVIDIAGameWorks/kaolin"
readonly GSPLAT_REPOSITORY="nerfstudio-project/gsplat"
readonly MIP_SPLATTING_REPOSITORY="autonomousvision/mip-splatting"
readonly NVDIFFRAST_REPOSITORY="NVlabs/nvdiffrast"
readonly NVDIFFREC_REPOSITORY="JeffreyXiang/nvdiffrec"
readonly DIFFOCTREERAST_REPOSITORY="JeffreyXiang/diffoctreerast"
readonly CUMESH_REPOSITORY="JeffreyXiang/CuMesh"
readonly FLEXGEMM_REPOSITORY="JeffreyXiang/FlexGEMM"

# Source revisions inherited from generate_dockerfile.sh.
readonly DEFAULT_UTILS3D_COMMIT="3913c65d81e05e47b9f367250cf8c0f7462a0900"
readonly DEFAULT_MOGE_COMMIT="a8c37341bc0325ca99b9d57981cc3bb2bd3e255b"
readonly DEFAULT_TRELLIS_COMMIT="442aa1e1afb9014e80681d3bf604e8d728a86ee7"
readonly DEFAULT_TRELLIS2_COMMIT="75fbf0183001ed9876c8dbb35de6b68552ee08bd"
readonly DEFAULT_SAM3D_OBJECTS_COMMIT="f91db411c50efee93d8db7aeb323885650f6f722"
readonly DEFAULT_PYTORCH3D_COMMIT="f5f6b78e70e0a1b70f3be9a09b5b001e9b3a7a03"
readonly DEFAULT_GSPLAT_COMMIT="2323de5905d5e90e035f792fe65bad0fedd413e7"
readonly DEFAULT_KAOLIN_REF="ad43ffd3ed9bb11fb4acc29e5b848712cdb53ce1"
readonly DEFAULT_MIP_SPLATTING_REF="dda02ab5ecf45d6edb8c540d9bb65c7e451345a9"
readonly DEFAULT_NVDIFFRAST_REF="v0.4.0"
readonly DEFAULT_NVDIFFREC_REF="b296927cc7fd01c2ac1087c8065c4d7248f72da4"
readonly DEFAULT_DIFFOCTREERAST_REF="b09c20b84ec3aace4729e6e18a613112320eca3a"
readonly DEFAULT_CUMESH_REF="12289e1062f0603f2f0d0771b02e1395d247f26f"
readonly DEFAULT_FLEXGEMM_REF="6dd94a859c26ee8246888502eada3dd8ad85532e"
readonly DEFAULT_TRELLIS_VERSION="0.0.1"
readonly DEFAULT_TRELLIS2_VERSION="0.0.1"
readonly DEFAULT_OVOXEL_VERSION="0.0.1"

# Direct target-version mappings inherited from generate_dockerfile.sh.
declare -Ar CUDA_IMAGE_BY_VERSION=(
    ["11.8"]="nvidia/cuda:11.8.0-devel-ubuntu22.04"
    ["12.1"]="nvidia/cuda:12.1.0-devel-ubuntu22.04"
    ["12.4"]="nvidia/cuda:12.4.0-devel-ubuntu22.04"
    ["12.6"]="nvidia/cuda:12.6.0-devel-ubuntu22.04"
    ["12.8"]="nvidia/cuda:12.8.0-devel-ubuntu22.04"
    ["12.9"]="nvidia/cuda:12.9.0-devel-ubuntu22.04"
    ["13.0"]="nvidia/cuda:13.0.0-devel-ubuntu22.04"
)

declare -Ar PYTHON_CP_TAG_BY_VERSION=(
    ["3.8"]="cp38"
    ["3.9"]="cp39"
    ["3.10"]="cp310"
    ["3.11"]="cp311"
    ["3.12"]="cp312"
    ["3.13"]="cp313"
    ["3.14"]="cp314"
)

declare -Ar TORCHVISION_BY_TORCH=(
    ["2.0.0"]="0.15.0"
    ["2.0.1"]="0.15.2"
    ["2.1.0"]="0.16.0"
    ["2.1.1"]="0.16.1"
    ["2.1.2"]="0.16.2"
    ["2.2.0"]="0.17.0"
    ["2.2.1"]="0.17.1"
    ["2.2.2"]="0.17.2"
    ["2.3.0"]="0.18.0"
    ["2.3.1"]="0.18.1"
    ["2.4.0"]="0.19.0"
    ["2.4.1"]="0.19.1"
    ["2.5.0"]="0.20.0"
    ["2.5.1"]="0.20.1"
    ["2.6.0"]="0.21.0"
    ["2.7.0"]="0.22.0"
    ["2.7.1"]="0.22.1"
    ["2.8.0"]="0.23.0"
    ["2.9.0"]="0.24.0"
    ["2.9.1"]="0.24.1"
    ["2.10.0"]="0.25.0"
    ["2.11.0"]="0.26.0"
    ["2.12.0"]="0.27.0"
    ["2.12.1"]="0.27.1"
)

declare -Ar XFORMERS_BY_TORCH=(
    ["2.2.2"]="0.0.25.post1"
    ["2.3.0"]="0.0.26.post1"
    ["2.3.1"]="0.0.27"
    ["2.4.0"]="0.0.27.post2"
    ["2.4.1"]="0.0.28"
    ["2.5.0"]="0.0.28.post2"
    ["2.5.1"]="0.0.28.post3"
    ["2.6.0"]="0.0.29.post3"
    ["2.7.0"]="0.0.30"
    ["2.7.1"]="0.0.31.post1"
    ["2.8.0"]="0.0.32.post2"
    ["2.9.0"]="0.0.33.post1"
    ["2.9.1"]="0.0.33.post2"
    ["2.10.0"]="0.0.34"
    ["2.11.0"]="0.0.35"
    ["2.12.0"]="0.0.35"
    ["2.12.1"]="0.0.35"
)

declare -Ar CUMM_SPCONV_PLAN_BY_CUDA=(
    ["11.8"]="cumm-cu113 0.7.14 spconv-cu113 2.4.1"
    ["12.1"]="cumm-cu121 0.7.14 spconv-cu121 2.4.1"
    ["12.4"]="cumm-cu121 0.7.14 spconv-cu121 2.4.1"
    ["12.6"]="cumm-cu126 0.9.1 spconv-cu126 2.4.1"
    ["12.8"]="cumm-cu128 0.9.1 spconv-cu128 2.4.1"
    ["12.9"]="cumm-cu128 0.9.1 spconv-cu128 2.4.1"
    ["13.0"]="cumm-cu130 0.9.1 spconv-cu130 2.4.1"
)

declare -Ar CUMM_SPCONV_PYTHON_BY_CUDA=(
    ["11.8"]="cp39 cp310 cp311"
    ["12.1"]="cp39 cp310 cp311"
    ["12.4"]="cp39 cp310 cp311"
    ["12.6"]="cp311 cp312 cp313 cp314"
    ["12.8"]="cp311 cp312 cp313 cp314"
    ["12.9"]="cp311 cp312 cp313 cp314"
    ["13.0"]="cp311 cp312 cp313 cp314"
)

readonly -a CUDA_ARCH_PLANS=(
    "12.8:2.[78]|7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;12.0+PTX"
    "12.8:2.9|7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;12.0+PTX"
    "12.8:2.1[0-4]|7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;12.0+PTX"
    "12.8:*|7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0+PTX"
    "12.9:2.8|7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;10.3;12.0;12.1+PTX"
    "12.9:2.9|7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.3;12.0;12.1+PTX"
    "12.9:2.1[0-4]|7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.3;12.0;12.1+PTX"
    "12.9:*|7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0+PTX"
    "13.0:2.9|7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.3;11.0;12.0;12.1+PTX"
    "13.0:2.1[0-4]|7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.3;11.0;12.0;12.1+PTX"
    "13.0:*|7.5;8.0;8.6;8.7;8.9;9.0+PTX"
    "11.8:*|7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0+PTX"
    "12.[146]:*|7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0+PTX"
)

declare -Ar FLASH_ATTN_PLAN_BY_XFORMERS=(
    ["0.0.25.post1"]="2.5.2 2.5.6 tuple"
    ["0.0.26.post1"]="2.5.2 2.5.6 tuple"
    ["0.0.27"]="2.5.7 2.5.7 tuple"
    ["0.0.27.post2"]="2.5.7 2.5.7 tuple"
    ["0.0.28"]="2.6.3 2.6.3 tuple"
    ["0.0.28.post2"]="2.6.3 2.6.3 tuple"
    ["0.0.28.post3"]="2.6.3 2.6.3 tuple"
    ["0.0.29.post3"]="2.7.1 2.7.2 tuple"
    ["0.0.30"]="2.7.1 2.7.4 tuple"
    ["0.0.31.post1"]="2.7.1 2.8.0 tuple"
    ["0.0.32.post2"]="2.7.1 2.8.2 packaging"
    ["0.0.33.post1"]="2.7.1 2.8.4 packaging"
    ["0.0.33.post2"]="2.7.1 2.8.4 packaging"
    ["0.0.34"]="2.7.1 2.8.4 packaging"
    ["0.0.35"]="2.7.1 2.8.4 packaging"
)

# User-configurable environment inputs.
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/pip}"
MAX_JOBS="${MAX_JOBS:-}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}"

XFORMERS_VERSION="${XFORMERS_VERSION:-}"
XFORMERS_WHEEL_PREFIX=""
FLASH_ATTN_RELEASE_TAG="${FLASH_ATTN_RELEASE_TAG:-}"
TRELLIS_VERSION="${TRELLIS_VERSION:-${DEFAULT_TRELLIS_VERSION}}"
TRELLIS2_VERSION="${TRELLIS2_VERSION:-${DEFAULT_TRELLIS2_VERSION}}"
OVOXEL_VERSION="${OVOXEL_VERSION:-${DEFAULT_OVOXEL_VERSION}}"
OVOXEL_RELEASE_TAG="${OVOXEL_RELEASE_TAG:-v${OVOXEL_VERSION}}"
SPCONV_INDEX_URL="${SPCONV_INDEX_URL:-${DEFAULT_SPCONV_INDEX_URL}}"

UTILS3D_COMMIT="${UTILS3D_COMMIT:-${DEFAULT_UTILS3D_COMMIT}}"
MOGE_COMMIT="${MOGE_COMMIT:-${DEFAULT_MOGE_COMMIT}}"
TRELLIS_COMMIT="${TRELLIS_COMMIT:-${DEFAULT_TRELLIS_COMMIT}}"
TRELLIS2_COMMIT="${TRELLIS2_COMMIT:-${DEFAULT_TRELLIS2_COMMIT}}"
SAM3D_OBJECTS_COMMIT="${SAM3D_OBJECTS_COMMIT:-${DEFAULT_SAM3D_OBJECTS_COMMIT}}"
PYTORCH3D_COMMIT="${PYTORCH3D_COMMIT:-${DEFAULT_PYTORCH3D_COMMIT}}"
GSPLAT_COMMIT="${GSPLAT_COMMIT:-${DEFAULT_GSPLAT_COMMIT}}"
KAOLIN_REF="${KAOLIN_REF:-${DEFAULT_KAOLIN_REF}}"
MIP_SPLATTING_REF="${MIP_SPLATTING_REF:-${DEFAULT_MIP_SPLATTING_REF}}"
NVDIFFRAST_REF="${NVDIFFRAST_REF:-${DEFAULT_NVDIFFRAST_REF}}"
NVDIFFREC_REF="${NVDIFFREC_REF:-${DEFAULT_NVDIFFREC_REF}}"
DIFFOCTREERAST_REF="${DIFFOCTREERAST_REF:-${DEFAULT_DIFFOCTREERAST_REF}}"
CUMESH_REF="${CUMESH_REF:-${DEFAULT_CUMESH_REF}}"
FLEXGEMM_REF="${FLEXGEMM_REF:-${DEFAULT_FLEXGEMM_REF}}"

# Command-line inputs and values derived from the selected target.
PYTHON_VERSION=""
TORCH_VERSION=""
CUDA_VERSION=""
CUDA_MINOR_VERSION=""
CUDA_TAG=""
PYTHON_CP_TAG=""
PYTHON_SHORT_TAG=""

BASE_IMAGE=""
TORCHVISION_VERSION=""
TORCH_INDEX_URL=""
WHEEL_OUTPUT_DIR=""

CUMM_PACKAGE=""
CUMM_VERSION=""
SPCONV_PACKAGE=""
SPCONV_VERSION=""

OVOXEL_WHEEL_NAME=""
OVOXEL_WHEEL_URL=""

declare -A SELECTED_PACKAGES=()
HAS_DOWNLOAD_PACKAGES=0
HAS_PURE_PYTHON_PACKAGES=0
HAS_NATIVE_PACKAGES=0
CONTAINER_SCRIPT=""

# Print command usage, version arguments, package selectors, and --all.
usage() {
    :
}

# Print an error message and terminate the script.
die() {
    printf 'error: %s\n' "$1" >&2
    exit 1
}

# Parse target versions and package-selection arguments.
parse_arguments() {
    local argument value package
    local -a packages=(
        "${DOWNLOAD_PACKAGES[@]}"
        "${PURE_PYTHON_PACKAGES[@]}"
        "${NATIVE_PACKAGES[@]}"
    )

    while (($#)); do
        argument="$1"
        shift

        case "$argument" in
        -h | --help)
            usage
            exit 0
            ;;

        --python | --torch | --cuda)
            (($#)) || die "$argument requires a value"
            value="$1"
            shift
            ;;

        --python=* | --torch=* | --cuda=*)
            value="${argument#*=}"
            argument="${argument%%=*}"
            ;;

        --all)
            for package in "${packages[@]}"; do
                SELECTED_PACKAGES["$package"]=1
            done
            continue
            ;;

        --*)
            package="${argument#--}"
            [[ " ${packages[*]} " == *" $package "* ]] ||
                die "unknown argument: $argument"
            SELECTED_PACKAGES["$package"]=1
            continue
            ;;

        *)
            die "unexpected positional argument: $argument"
            ;;
        esac

        case "$argument" in
        --python) PYTHON_VERSION="$value" ;;
        --torch) TORCH_VERSION="$value" ;;
        --cuda) CUDA_VERSION="$value" ;;
        esac
    done

    [[ -n "$PYTHON_VERSION" ]] || die "missing --python"
    [[ -n "$TORCH_VERSION" ]] || die "missing --torch"
    [[ -n "$CUDA_VERSION" ]] || die "missing --cuda"
    ((${#SELECTED_PACKAGES[@]})) || die "no package selected"

    for package in "${!SELECTED_PACKAGES[@]}"; do
        [[ " ${DOWNLOAD_PACKAGES[*]} " == *" $package "* ]] &&
            HAS_DOWNLOAD_PACKAGES=1
        [[ " ${PURE_PYTHON_PACKAGES[*]} " == *" $package "* ]] &&
            HAS_PURE_PYTHON_PACKAGES=1
        [[ " ${NATIVE_PACKAGES[*]} " == *" $package "* ]] &&
            HAS_NATIVE_PACKAGES=1
    done

    return 0
}

# Resolve target tags, output paths, CUDA image, and architecture list.
resolve_target_versions() {
    local cuda_major cuda_minor cuda_patch
    local target arch_plan arch_pattern

    IFS='.' read -r cuda_major cuda_minor cuda_patch <<<"$CUDA_VERSION"
    [[ -n "$cuda_major" && -n "$cuda_minor" ]] ||
        die "invalid CUDA version: $CUDA_VERSION"

    CUDA_MINOR_VERSION="${cuda_major}.${cuda_minor}"
    CUDA_TAG="cu${CUDA_MINOR_VERSION//./}"
    PYTHON_SHORT_TAG="py${PYTHON_VERSION//./}"

    BASE_IMAGE="${CUDA_IMAGE_BY_VERSION[$CUDA_MINOR_VERSION]:-}"
    PYTHON_CP_TAG="${PYTHON_CP_TAG_BY_VERSION[$PYTHON_VERSION]:-}"

    [[ -n "$BASE_IMAGE" ]] || die "unsupported CUDA: $CUDA_MINOR_VERSION"
    [[ -n "$PYTHON_CP_TAG" ]] || die "unsupported Python: $PYTHON_VERSION"

    TORCH_INDEX_URL="${PYTORCH_INDEX_BASE_URL}/${CUDA_TAG}"
    WHEEL_OUTPUT_DIR="${WHEELS_ROOT}/torch${TORCH_VERSION}-${CUDA_TAG}-${PYTHON_SHORT_TAG}"

    ((HAS_NATIVE_PACKAGES)) || return 0

    TORCHVISION_VERSION="${TORCHVISION_BY_TORCH[$TORCH_VERSION]:-}"
    [[ -n "$TORCHVISION_VERSION" ]] ||
        die "no torchvision mapping for torch $TORCH_VERSION"

    [[ -z "$TORCH_CUDA_ARCH_LIST" ]] || return 0

    target="${CUDA_MINOR_VERSION}:${TORCH_VERSION%.*}"

    for arch_plan in "${CUDA_ARCH_PLANS[@]}"; do
        arch_pattern="${arch_plan%%|*}"
        [[ "$target" == $arch_pattern ]] || continue
        TORCH_CUDA_ARCH_LIST="${arch_plan#*|}"
        return 0
    done

    die "no CUDA architecture mapping for $target"
}

# Resolve versions, repositories, refs, and indexes for selected packages.
resolve_package_versions() {
    local plan allowed_python encoded_wheel_name
    local xformers_wheel_version xformers_python_tag

    if [[ -n "${SELECTED_PACKAGES[cumm]+selected}" ||
        -n "${SELECTED_PACKAGES[spconv]+selected}" ]]; then
        plan="${CUMM_SPCONV_PLAN_BY_CUDA[$CUDA_MINOR_VERSION]:-}"
        allowed_python="${CUMM_SPCONV_PYTHON_BY_CUDA[$CUDA_MINOR_VERSION]:-}"

        if [[ -z "$plan" ||
            " $allowed_python " != *" $PYTHON_CP_TAG "* ]]; then
            die "no cumm/spconv wheels for CUDA $CUDA_MINOR_VERSION and Python $PYTHON_VERSION"
        fi

        read -r CUMM_PACKAGE CUMM_VERSION SPCONV_PACKAGE SPCONV_VERSION \
            <<<"$plan"
    fi

    if [[ -n "${SELECTED_PACKAGES[xformers]+selected}" ||
        -n "${SELECTED_PACKAGES["flash-attn"]+selected}" ]]; then
        XFORMERS_VERSION="${XFORMERS_VERSION:-${XFORMERS_BY_TORCH[$TORCH_VERSION]:-}}"
        [[ -n "$XFORMERS_VERSION" ]] ||
            die "no xformers mapping for torch $TORCH_VERSION"
    fi

    if [[ -n "${SELECTED_PACKAGES[xformers]+selected}" ]]; then
        xformers_wheel_version="${XFORMERS_VERSION//+/%2B}"

        case "$XFORMERS_VERSION" in
        0.0.31* | 0.0.32* | 0.0.33* | 0.0.34*)
            xformers_python_tag="cp39-abi3"
            ;;
        0.0.35*)
            xformers_python_tag="py39-none"
            ;;
        *)
            xformers_python_tag="${PYTHON_CP_TAG}-${PYTHON_CP_TAG}"
            ;;
        esac

        if [[ "$PYTHON_CP_TAG" == "cp38" &&
            "$xformers_python_tag" != "cp38-cp38" ]]; then
            die "xformers $XFORMERS_VERSION requires Python 3.9 or newer"
        fi

        case "${CUDA_TAG}:${XFORMERS_VERSION}" in
        cu118:0.0.25.post1 | cu118:0.0.26.post1 | cu118:0.0.27 | cu118:0.0.27.post2)
            xformers_wheel_version="${XFORMERS_VERSION}%2Bcu118"
            ;;
        esac

        XFORMERS_WHEEL_PREFIX="xformers-${xformers_wheel_version}-${xformers_python_tag}"
    fi

    if [[ -n "${SELECTED_PACKAGES["o-voxel-gpu"]+selected}" ]]; then
        OVOXEL_WHEEL_NAME="o_voxel-${OVOXEL_VERSION}+torch${TORCH_VERSION}.${CUDA_TAG}-${PYTHON_CP_TAG}-${PYTHON_CP_TAG}-linux_x86_64.whl"
        encoded_wheel_name="${OVOXEL_WHEEL_NAME//+/%2B}"
        OVOXEL_WHEEL_URL="https://github.com/${OVOXEL_REPOSITORY}/releases/download/${OVOXEL_RELEASE_TAG}/${encoded_wheel_name}"
    fi

    [[ -z "${SELECTED_PACKAGES["flash-attn"]+selected}" ]] ||
        resolve_flash_attn_version
}

# Resolve and validate the flash-attn release for the selected target.
resolve_flash_attn_version() {
    local plan min_version max_version compare_mode
    local major minor patch min_key max_key candidate_key
    local requested_tag="$FLASH_ATTN_RELEASE_TAG"
    local release_path="releases?per_page=100"
    local response line asset
    local tag=""
    local prerelease=0
    local asset_base="cu${CUDA_MINOR_VERSION%%.*}torch${TORCH_VERSION%.*}cxx11abi"
    local asset_suffix="-${PYTHON_CP_TAG}-${PYTHON_CP_TAG}-linux_x86_64.whl"

    plan="${FLASH_ATTN_PLAN_BY_XFORMERS[$XFORMERS_VERSION]:-}"
    [[ -n "$plan" ]] ||
        die "no flash-attn mapping for xformers $XFORMERS_VERSION"

    read -r min_version max_version compare_mode <<<"$plan"

    IFS='.' read -r major minor patch <<<"$min_version"
    min_key=$((10#$major * 1000000 + 10#$minor * 1000 + 10#$patch))

    IFS='.' read -r major minor patch <<<"$max_version"
    max_key=$((10#$major * 1000000 + 10#$minor * 1000 + 10#$patch))

    [[ -z "$requested_tag" ]] ||
        release_path="releases/tags/${requested_tag}"

    response="$(
        curl -fsSL \
            "https://api.github.com/repos/${FLASH_ATTN_REPOSITORY}/${release_path}"
    )" || die "cannot query flash-attn releases"

    while IFS= read -r line; do
        case "$line" in
        *'"tag_name":'*)
            tag="$(sed -E 's/.*"tag_name": "([^"]+)".*/\1/' <<<"$line")"
            prerelease=0
            ;;

        *'"prerelease": true'*)
            prerelease=1
            ;;

        *'"prerelease": false'*)
            prerelease=0
            ;;

        *'"name": "flash_attn-'*)
            [[ -n "$tag" ]] || continue
            [[ -n "$requested_tag" || "$prerelease" -eq 0 ]] || continue
            [[ -z "$requested_tag" || "$tag" == "$requested_tag" ]] || continue
            [[ "$tag" =~ ^v([0-9]+)\.([0-9]+)\.([0-9]+) ]] || continue

            major="${BASH_REMATCH[1]}"
            minor="${BASH_REMATCH[2]}"
            patch="${BASH_REMATCH[3]}"
            candidate_key=$((\
                10#$major * 1000000 + \
                10#$minor * 1000 + \
                10#$patch))

            ((candidate_key >= min_key && candidate_key <= max_key)) ||
                continue

            [[ "$compare_mode" != "packaging" ||
                "$candidate_key" -ne "$max_key" ||
                "$tag" == "v${max_version}" ]] ||
                continue

            asset="$(sed -E 's/.*"name": "([^"]+)".*/\1/' <<<"$line")"

            [[ "$asset" == flash_attn-*+"${asset_base}"FALSE"${asset_suffix}" ||
                "$asset" == flash_attn-*+"${asset_base}"TRUE"${asset_suffix}" ]] ||
                continue

            FLASH_ATTN_RELEASE_TAG="$tag"
            return 0
            ;;
        esac
    done <<<"$response"

    die "no compatible flash-attn release for torch $TORCH_VERSION, CUDA $CUDA_MINOR_VERSION, Python $PYTHON_VERSION"
}

# Emit the ca-certificates and curl preparation needed by download-only tasks.
emit_download_setup() {
    cat <<EOF
apt-get update &&
apt-get install -y --no-install-recommends ca-certificates curl

download_wheel() {
    local url="\$1" filename="\$2"
    local temporary_file="${CONTAINER_BUILD_DIR}/\${filename}.part"

    curl -fsSL "\$url" -o "\$temporary_file"
    chown "\${HOST_UID}:\${HOST_GID}" "\$temporary_file"
    mv "\$temporary_file" "${CONTAINER_WHEELS_DIR}/\${filename}"
}

download_index_wheel() {
    local index_url="\$1" wheel_prefix="\$2"
    local index_html line wheel_url="" filename

    index_html="\$(curl -fsSL "\$index_url")"

    while IFS= read -r line; do
        case "\$line" in
        *'href="'*"/\${wheel_prefix}-"*x86_64.whl*)
            wheel_url="\${line#*href=\\"}"
            wheel_url="\${wheel_url%%\\"*}"
            break
            ;;
        esac
    done <<<"\$index_html"

    if [[ -z "\$wheel_url" ]]; then
        printf 'wheel not found: %s in %s\\n' "\$wheel_prefix" "\$index_url" >&2
        return 1
    fi

    wheel_url="\${wheel_url%%#*}"
    if [[ "\$wheel_url" == /* ]]; then
        wheel_url="\${index_url%%/whl/*}\${wheel_url}"
    fi

    filename="\${wheel_url##*/}"
    filename="\${filename//%2B/+}"
    download_wheel "\$wheel_url" "\$filename"
}

EOF
}

# Emit the cumm wheel download commands.
emit_cumm_commands() {
    local wheel_prefix="${CUMM_PACKAGE//-/_}-${CUMM_VERSION}-${PYTHON_CP_TAG}-${PYTHON_CP_TAG}"

    cat <<EOF
download_index_wheel "${SPCONV_INDEX_URL%/}/${CUMM_PACKAGE}/" "${wheel_prefix}"

EOF
}

# Emit the spconv wheel download commands.
emit_spconv_commands() {
    local wheel_prefix="${SPCONV_PACKAGE//-/_}-${SPCONV_VERSION}-${PYTHON_CP_TAG}-${PYTHON_CP_TAG}"

    cat <<EOF
download_index_wheel "${SPCONV_INDEX_URL%/}/${SPCONV_PACKAGE}/" "${wheel_prefix}"

EOF
}

# Emit the xformers wheel download commands.
emit_xformers_commands() {
    cat <<EOF
download_index_wheel "${TORCH_INDEX_URL}/xformers/" "${XFORMERS_WHEEL_PREFIX}"

EOF
}

# Emit the o-voxel-gpu release wheel download commands.
emit_o_voxel_gpu_commands() {
    cat <<EOF
download_wheel "${OVOXEL_WHEEL_URL}" "${OVOXEL_WHEEL_NAME}"

EOF
}

# Emit root-level apt, Python, pip, and build-identity preparation commands.
emit_apt_setup() {
    if ((!HAS_DOWNLOAD_PACKAGES)); then
        cat <<'EOF'
apt-get update
EOF
    fi

    cat <<'EOF'
apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    software-properties-common \
    util-linux
EOF

    if [[ "$PYTHON_VERSION" != "3.10" ]]; then
        cat <<'EOF'
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
EOF
    fi

    cat <<EOF
apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv

mkdir -p \
    "${CONTAINER_BUILD_DIR}/home"
python${PYTHON_VERSION} -m venv "${CONTAINER_BUILD_DIR}/venv"
chown -R "\${HOST_UID}:\${HOST_GID}" "${CONTAINER_BUILD_DIR}"

setpriv \
    --reuid "\${HOST_UID}" \
    --regid "\${HOST_GID}" \
    --clear-groups \
    env \
        HOME="${CONTAINER_BUILD_DIR}/home" \
        PIP_CACHE_DIR="${CONTAINER_PIP_CACHE_DIR}" \
    "${CONTAINER_BUILD_DIR}/venv/bin/python" -m pip install --upgrade \
        pip \
        setuptools \
        wheel \
        packaging \
        build

EOF
}

# Emit the utils3d source preparation and pure Python packaging commands.
emit_utils3d_commands() {
    cat <<EOF
git clone \
    "https://github.com/${UTILS3D_REPOSITORY}.git" \
    "${CONTAINER_BUILD_DIR}/utils3d"
git -C "${CONTAINER_BUILD_DIR}/utils3d" \
    checkout --detach "${UTILS3D_COMMIT}"

python -m build \
    --wheel \
    --no-isolation \
    --outdir "${CONTAINER_WHEELS_DIR}" \
    "${CONTAINER_BUILD_DIR}/utils3d"

EOF
}

# Emit the MoGe source preparation and pure Python packaging commands.
emit_moge_commands() {
    cat <<EOF
git clone \
    "https://github.com/${MOGE_REPOSITORY}.git" \
    "${CONTAINER_BUILD_DIR}/moge"
git -C "${CONTAINER_BUILD_DIR}/moge" \
    checkout --detach "${MOGE_COMMIT}"

sed -i \
    's|"utils3d @ git+https://github.com/${UTILS3D_REPOSITORY}.git@${UTILS3D_COMMIT}"|"utils3d==0.0.2"|' \
    "${CONTAINER_BUILD_DIR}/moge/pyproject.toml"

python -m build \
    --wheel \
    --no-isolation \
    --outdir "${CONTAINER_WHEELS_DIR}" \
    "${CONTAINER_BUILD_DIR}/moge"

EOF
}

# Emit the TRELLIS source preparation and pure Python packaging commands.
emit_trellis_commands() {
    cat <<EOF
git clone \
    "https://github.com/${TRELLIS_REPOSITORY}.git" \
    "${CONTAINER_BUILD_DIR}/trellis"
git -C "${CONTAINER_BUILD_DIR}/trellis" \
    checkout --detach "${TRELLIS_COMMIT}"
git -C "${CONTAINER_BUILD_DIR}/trellis" \
    submodule update --init --recursive

sed -i \
    '/^from \. import pipelines$/d' \
    "${CONTAINER_BUILD_DIR}/trellis/trellis/__init__.py"

cat > "${CONTAINER_BUILD_DIR}/trellis/pyproject.toml" <<'TRELLIS_PYPROJECT'
[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "trellis"
version = "${TRELLIS_VERSION}"

[tool.setuptools.packages.find]
where = ["."]
include = ["trellis*"]
namespaces = true
TRELLIS_PYPROJECT

python -m build \
    --wheel \
    --no-isolation \
    --outdir "${CONTAINER_WHEELS_DIR}" \
    "${CONTAINER_BUILD_DIR}/trellis"

EOF
}

# Emit the TRELLIS.2 source preparation and pure Python packaging commands.
emit_trellis2_commands() {
    cat <<EOF
git clone \
    "https://github.com/${TRELLIS2_REPOSITORY}.git" \
    "${CONTAINER_BUILD_DIR}/trellis2"
git -C "${CONTAINER_BUILD_DIR}/trellis2" \
    checkout --detach "${TRELLIS2_COMMIT}"

cat > "${CONTAINER_BUILD_DIR}/trellis2/pyproject.toml" <<'TRELLIS2_PYPROJECT'
[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "trellis2"
version = "${TRELLIS2_VERSION}"

[tool.setuptools.packages.find]
where = ["."]
include = ["trellis2*"]
namespaces = true
TRELLIS2_PYPROJECT

python -m build \
    --wheel \
    --no-isolation \
    --outdir "${CONTAINER_WHEELS_DIR}" \
    "${CONTAINER_BUILD_DIR}/trellis2"

EOF
}

# Emit the local SAM 3D Objects pure Python packaging commands.
emit_sam3d_objects_commands() {
    cat <<EOF
git clone \
    "https://github.com/${SAM3D_OBJECTS_REPOSITORY}.git" \
    "${CONTAINER_BUILD_DIR}/sam3d-objects"
git -C "${CONTAINER_BUILD_DIR}/sam3d-objects" \
    checkout --detach "${SAM3D_OBJECTS_COMMIT}"

cat > "${CONTAINER_BUILD_DIR}/sam3d-objects/sam3d_objects/__init__.py" <<'SAM3D_INIT'
# Copyright (c) Meta Platforms, Inc. and affiliates.
SAM3D_INIT

cat > "${CONTAINER_BUILD_DIR}/sam3d-objects/pyproject.toml" <<'SAM3D_PYPROJECT'
[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "sam3d_objects"
version = "0.0.1"

[tool.setuptools.packages.find]
where = ["."]
include = ["sam3d_objects*"]
namespaces = true
SAM3D_PYPROJECT

python -m build \
    --wheel \
    --no-isolation \
    --outdir "${CONTAINER_WHEELS_DIR}" \
    "${CONTAINER_BUILD_DIR}/sam3d-objects"

EOF
}

# Emit Torch, torchvision, and CUDA build environment commands.
emit_torch_setup() {
    cat <<'EOF'
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export FORCE_CUDA=1
EOF

    if [[ -n "$MAX_JOBS" ]]; then
        cat <<'EOF'
export CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS}"
export MAKEFLAGS="-j${MAX_JOBS}"
EOF
    fi

    cat <<EOF

python -m pip install \
    ninja \
    psutil

python -m pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"

EOF
}

# Emit the flash-attn source preparation and wheel build commands.
emit_flash_attn_commands() {
    cat <<EOF
git clone \
    --depth 1 \
    --branch "${FLASH_ATTN_RELEASE_TAG}" \
    --recursive \
    --shallow-submodules \
    "https://github.com/${FLASH_ATTN_REPOSITORY}.git" \
    "${CONTAINER_BUILD_DIR}/flash-attn"

FLASH_ATTENTION_FORCE_BUILD=TRUE \
python -m build \
    --wheel \
    --no-isolation \
    --outdir "${CONTAINER_WHEELS_DIR}" \
    "${CONTAINER_BUILD_DIR}/flash-attn"

EOF
}

# Emit the PyTorch3D source preparation and wheel build commands.
emit_pytorch3d_commands() {
    cat <<EOF
git clone \
    "https://github.com/${PYTORCH3D_REPOSITORY}.git" \
    "${CONTAINER_BUILD_DIR}/pytorch3d"
git -C "${CONTAINER_BUILD_DIR}/pytorch3d" \
    checkout --detach "${PYTORCH3D_COMMIT}"

python -m build \
    --wheel \
    --no-isolation \
    --outdir "${CONTAINER_WHEELS_DIR}" \
    "${CONTAINER_BUILD_DIR}/pytorch3d"

EOF
}

# Emit the Kaolin source preparation and wheel build commands.
emit_kaolin_commands() {
    cat <<EOF
python -m pip install "setuptools==75.8.2"

git clone \
    "https://github.com/${KAOLIN_REPOSITORY}.git" \
    "${CONTAINER_BUILD_DIR}/kaolin"
git -C "${CONTAINER_BUILD_DIR}/kaolin" \
    checkout --detach "${KAOLIN_REF}"

grep -q \
    "viz_requirements.txt" \
    "${CONTAINER_BUILD_DIR}/kaolin/setup.py"
sed -i \
    "/viz_requirements.txt/,+1d" \
    "${CONTAINER_BUILD_DIR}/kaolin/setup.py"

python -m build \
    --wheel \
    --no-isolation \
    --outdir "${CONTAINER_WHEELS_DIR}" \
    "${CONTAINER_BUILD_DIR}/kaolin"

EOF
}

# Emit the gsplat source preparation and wheel build commands.
emit_gsplat_commands() {
    cat <<EOF
git clone \
    "https://github.com/${GSPLAT_REPOSITORY}.git" \
    "${CONTAINER_BUILD_DIR}/gsplat"
git -C "${CONTAINER_BUILD_DIR}/gsplat" \
    checkout --detach "${GSPLAT_COMMIT}"
git -C "${CONTAINER_BUILD_DIR}/gsplat" \
    submodule update \
    --init \
    --recursive \
    gsplat/cuda/csrc/third_party/glm

python -m build \
    --wheel \
    --no-isolation \
    --outdir "${CONTAINER_WHEELS_DIR}" \
    "${CONTAINER_BUILD_DIR}/gsplat"

EOF
}

# Emit the diff-gaussian-rasterization source preparation and wheel build commands.
emit_diff_gaussian_rasterization_commands() {
    cat <<EOF
git clone \
    "https://github.com/${MIP_SPLATTING_REPOSITORY}.git" \
    "${CONTAINER_BUILD_DIR}/mip-splatting"
git -C "${CONTAINER_BUILD_DIR}/mip-splatting" \
    checkout --detach "${MIP_SPLATTING_REF}"

python -m build \
    --wheel \
    --no-isolation \
    --outdir "${CONTAINER_WHEELS_DIR}" \
    "${CONTAINER_BUILD_DIR}/mip-splatting/submodules/diff-gaussian-rasterization"

EOF
}

# Emit the nvdiffrast source preparation and wheel build commands.
emit_nvdiffrast_commands() {
    cat <<EOF
git clone \
    "https://github.com/${NVDIFFRAST_REPOSITORY}.git" \
    "${CONTAINER_BUILD_DIR}/nvdiffrast"
git -C "${CONTAINER_BUILD_DIR}/nvdiffrast" \
    checkout --detach "${NVDIFFRAST_REF}"

python -m build \
    --wheel \
    --no-isolation \
    --outdir "${CONTAINER_WHEELS_DIR}" \
    "${CONTAINER_BUILD_DIR}/nvdiffrast"

EOF
}

# Emit the nvdiffrec-render source preparation and wheel build commands.
emit_nvdiffrec_render_commands() {
    cat <<EOF
git clone \
    "https://github.com/${NVDIFFREC_REPOSITORY}.git" \
    "${CONTAINER_BUILD_DIR}/nvdiffrec-render"
git -C "${CONTAINER_BUILD_DIR}/nvdiffrec-render" \
    checkout --detach "${NVDIFFREC_REF}"

sed -i \
    "/name='nvdiffrec_render',/a\\    version='0.0.1'," \
    "${CONTAINER_BUILD_DIR}/nvdiffrec-render/setup.py"

python -m build \
    --wheel \
    --no-isolation \
    --outdir "${CONTAINER_WHEELS_DIR}" \
    "${CONTAINER_BUILD_DIR}/nvdiffrec-render"

EOF
}

# Emit the diffoctreerast source preparation and wheel build commands.
emit_diffoctreerast_commands() {
    cat <<EOF
git clone \
    "https://github.com/${DIFFOCTREERAST_REPOSITORY}.git" \
    "${CONTAINER_BUILD_DIR}/diffoctreerast"
git -C "${CONTAINER_BUILD_DIR}/diffoctreerast" \
    checkout --detach "${DIFFOCTREERAST_REF}"
git -C "${CONTAINER_BUILD_DIR}/diffoctreerast" \
    submodule update \
    --init \
    --recursive \
    lib/glm

sed -i \
    '/name="diffoctreerast",/a\    version="0.0.1",' \
    "${CONTAINER_BUILD_DIR}/diffoctreerast/setup.py"

python -m build \
    --wheel \
    --no-isolation \
    --outdir "${CONTAINER_WHEELS_DIR}" \
    "${CONTAINER_BUILD_DIR}/diffoctreerast"

EOF
}

# Emit the CuMesh source preparation and wheel build commands.
emit_cumesh_commands() {
    cat <<EOF
git clone \
    "https://github.com/${CUMESH_REPOSITORY}.git" \
    "${CONTAINER_BUILD_DIR}/cumesh"
git -C "${CONTAINER_BUILD_DIR}/cumesh" \
    checkout --detach "${CUMESH_REF}"
git -C "${CONTAINER_BUILD_DIR}/cumesh" \
    submodule update --init --recursive

python -m build \
    --wheel \
    --no-isolation \
    --outdir "${CONTAINER_WHEELS_DIR}" \
    "${CONTAINER_BUILD_DIR}/cumesh"

EOF
}

# Emit the FlexGEMM source preparation and wheel build commands.
emit_flex_gemm_commands() {
    cat <<EOF
git clone \
    "https://github.com/${FLEXGEMM_REPOSITORY}.git" \
    "${CONTAINER_BUILD_DIR}/flex-gemm"
git -C "${CONTAINER_BUILD_DIR}/flex-gemm" \
    checkout --detach "${FLEXGEMM_REF}"

python - \
    "${CONTAINER_BUILD_DIR}/flex-gemm/flex_gemm/ops/spconv/submanifold_conv3d.py" \
    <<'FLEXGEMM_PATCH'
from pathlib import Path
import sys

source_path = Path(sys.argv[1])
old = "            grad_weight = grad_weight.reshape(Co, Kw, Kh, Kd, Ci)"
new = (
    "            if grad_weight is not None:\n"
    "                grad_weight = grad_weight.reshape(Co, Kw, Kh, Kd, Ci)"
)

source = source_path.read_text()
count = source.count(old)
if count != 4:
    raise RuntimeError(f"expected 4 patch targets, found {count}: {source_path}")

source_path.write_text(source.replace(old, new))
FLEXGEMM_PATCH

python -m build \
    --wheel \
    --no-isolation \
    --outdir "${CONTAINER_WHEELS_DIR}" \
    "${CONTAINER_BUILD_DIR}/flex-gemm"

EOF
}

# Emit the local SymTRELLIS wheel build commands.
emit_symtrellis_commands() {
    cat <<EOF
mkdir -p "${CONTAINER_BUILD_DIR}/symtrellis"
cp -R \
    "${CONTAINER_REPO_DIR}/setup.py" \
    "${CONTAINER_REPO_DIR}/symtrellis" \
    "${CONTAINER_BUILD_DIR}/symtrellis/"

python -m build \
    --wheel \
    --no-isolation \
    --outdir "${CONTAINER_WHEELS_DIR}" \
    "${CONTAINER_BUILD_DIR}/symtrellis"

EOF
}

# Dispatch a selected package name to its command-emitter function.
emit_selected_packages() {
    local package

    for package in "$@"; do
        [[ -n "${SELECTED_PACKAGES["$package"]+selected}" ]] || continue

        case "$package" in
        cumm) emit_cumm_commands ;;
        spconv) emit_spconv_commands ;;
        xformers) emit_xformers_commands ;;
        o-voxel-gpu) emit_o_voxel_gpu_commands ;;
        utils3d) emit_utils3d_commands ;;
        moge) emit_moge_commands ;;
        trellis) emit_trellis_commands ;;
        trellis2) emit_trellis2_commands ;;
        sam3d-objects) emit_sam3d_objects_commands ;;
        flash-attn) emit_flash_attn_commands ;;
        pytorch3d) emit_pytorch3d_commands ;;
        kaolin) emit_kaolin_commands ;;
        gsplat) emit_gsplat_commands ;;
        diff-gaussian-rasterization) emit_diff_gaussian_rasterization_commands ;;
        nvdiffrast) emit_nvdiffrast_commands ;;
        nvdiffrec-render) emit_nvdiffrec_render_commands ;;
        diffoctreerast) emit_diffoctreerast_commands ;;
        cumesh) emit_cumesh_commands ;;
        flex-gemm) emit_flex_gemm_commands ;;
        symtrellis) emit_symtrellis_commands ;;
        esac
    done
}

# Collect the complete apt, download, Torch, and package command stream in memory.
collect_container_script() {
    local build_script

    build_script="$(
        if ((HAS_PURE_PYTHON_PACKAGES)); then
            emit_selected_packages "${PURE_PYTHON_PACKAGES[@]}"
        fi

        if ((HAS_NATIVE_PACKAGES)); then
            emit_torch_setup
            emit_selected_packages "${NATIVE_PACKAGES[@]}"
        fi
    )"

    CONTAINER_SCRIPT="$(
        cat <<EOF
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
mkdir -p "${CONTAINER_BUILD_DIR}"

EOF

        if ((HAS_DOWNLOAD_PACKAGES)); then
            emit_download_setup
            emit_selected_packages "${DOWNLOAD_PACKAGES[@]}"
        fi

        if [[ -n "$build_script" ]]; then
            emit_apt_setup

            cat <<EOF
setpriv \
    --reuid "\${HOST_UID}" \
    --regid "\${HOST_GID}" \
    --clear-groups \
    env \
        HOME="${CONTAINER_BUILD_DIR}/home" \
        PATH="${CONTAINER_BUILD_DIR}/venv/bin:\${PATH}" \
        PIP_CACHE_DIR="${CONTAINER_PIP_CACHE_DIR}" \
    bash <<'PACKAGE_BUILD_SCRIPT'
set -euo pipefail
cd "${CONTAINER_BUILD_DIR}"

EOF
            printf '%s\n' "$build_script"
            cat <<'EOF'
PACKAGE_BUILD_SCRIPT
EOF
        fi
    )"
}

# Run the CUDA builder container with cache, source, output, and UID/GID mappings.
run_builder_container() {
    local host_uid host_gid
    local -a docker_arguments

    mkdir -p "$WHEEL_OUTPUT_DIR" "$PIP_CACHE_DIR"

    host_uid="$(id -u)"
    host_gid="$(id -g)"

    docker_arguments=(
        run
        --rm
        -i
        --runtime runc
        --mount "type=bind,source=${REPO_DIR},target=${CONTAINER_REPO_DIR},readonly"
        --mount "type=bind,source=${WHEEL_OUTPUT_DIR},target=${CONTAINER_WHEELS_DIR}"
        --mount "type=bind,source=${PIP_CACHE_DIR},target=${CONTAINER_PIP_CACHE_DIR}"
        --env "HOST_UID=${host_uid}"
        --env "HOST_GID=${host_gid}"
    )

    if ((HAS_NATIVE_PACKAGES)); then
        docker_arguments+=(
            --env "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
        )
    fi

    if [[ -n "$MAX_JOBS" ]]; then
        docker_arguments+=(
            --env "MAX_JOBS=${MAX_JOBS}"
        )
    fi

    printf '%s\n' "$CONTAINER_SCRIPT" |
        docker "${docker_arguments[@]}" "$BASE_IMAGE" bash -s
}

# Coordinate argument parsing, version resolution, command collection, and execution.
main() {
    parse_arguments "$@"

    [[ -z "$MAX_JOBS" || "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]] ||
        die "MAX_JOBS must be a positive integer"

    resolve_target_versions
    resolve_package_versions
    collect_container_script
    run_builder_container
}

main "$@"
