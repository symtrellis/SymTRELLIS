#!/usr/bin/env bash
set -euo pipefail

readonly WHEELS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/wheels"
readonly RELEASES_URL="https://github.com/symtrellis/SymTRELLIS/releases/download"

declare -Ar MATRIX=(
    ["torch2.6.0-cu124-py310"]='
        wheels-cp310-torch2.6.0-cu124
        cumesh-0.0.1-cp310-cp310-linux_x86_64.whl
        cumm_cu121-0.7.14-cp310-cp310-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
        diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl
        diffoctreerast-0.0.1-cp310-cp310-linux_x86_64.whl
        flash_attn-2.7.2.post1-cp310-cp310-linux_x86_64.whl
        flex_gemm-1.0.0-cp310-cp310-linux_x86_64.whl
        gsplat-1.5.3-cp310-cp310-linux_x86_64.whl
        kaolin-0.18.0-cp310-cp310-linux_x86_64.whl
        moge-1.0.0-py3-none-any.whl
        nvdiffrast-0.4.0-cp310-cp310-linux_x86_64.whl
        nvdiffrec_render-0.0.1-cp310-cp310-linux_x86_64.whl
        o_voxel-0.0.1+torch2.6.0.cu124-cp310-cp310-linux_x86_64.whl
        pytorch3d-0.7.9-cp310-cp310-linux_x86_64.whl
        sam3d_objects-0.0.1-py3-none-any.whl
        spconv_cu121-2.4.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
        symtrellis-0.0.1-cp310-cp310-linux_x86_64.whl
        trellis-0.0.1-py3-none-any.whl
        trellis2-0.0.1-py3-none-any.whl
        utils3d-0.0.2-py3-none-any.whl
        xformers-0.0.29.post3-cp310-cp310-manylinux_2_28_x86_64.whl
    '
    ["torch2.7.0-cu128-py311"]='
        wheels-cp311-torch2.7.0-cu128
        cumesh-0.0.1-cp311-cp311-linux_x86_64.whl
        cumm_cu128-0.9.1-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
        diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl
        diffoctreerast-0.0.1-cp311-cp311-linux_x86_64.whl
        flash_attn-2.7.4.post1-cp311-cp311-linux_x86_64.whl
        flex_gemm-1.0.0-cp311-cp311-linux_x86_64.whl
        gsplat-1.5.3-cp311-cp311-linux_x86_64.whl
        kaolin-0.18.0-cp311-cp311-linux_x86_64.whl
        moge-1.0.0-py3-none-any.whl
        nvdiffrast-0.4.0-cp311-cp311-linux_x86_64.whl
        nvdiffrec_render-0.0.1-cp311-cp311-linux_x86_64.whl
        o_voxel-0.0.1+torch2.7.0.cu128-cp311-cp311-linux_x86_64.whl
        pytorch3d-0.7.9-cp311-cp311-linux_x86_64.whl
        sam3d_objects-0.0.1-py3-none-any.whl
        spconv_cu128-2.4.1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
        symtrellis-0.0.1-cp311-cp311-linux_x86_64.whl
        trellis-0.0.1-py3-none-any.whl
        trellis2-0.0.1-py3-none-any.whl
        utils3d-0.0.2-py3-none-any.whl
        xformers-0.0.30-cp311-cp311-manylinux_2_28_x86_64.whl
    '
    ["torch2.8.0-cu128-py312"]='
        wheels-cp312-torch2.8.0-cu128
        cumesh-0.0.1-cp312-cp312-linux_x86_64.whl
        cumm_cu128-0.9.1-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
        diff_gaussian_rasterization-0.0.0-cp312-cp312-linux_x86_64.whl
        diffoctreerast-0.0.1-cp312-cp312-linux_x86_64.whl
        flash_attn-2.8.1-cp312-cp312-linux_x86_64.whl
        flex_gemm-1.0.0-cp312-cp312-linux_x86_64.whl
        gsplat-1.5.3-cp312-cp312-linux_x86_64.whl
        kaolin-0.18.0-cp312-cp312-linux_x86_64.whl
        moge-1.0.0-py3-none-any.whl
        nvdiffrast-0.4.0-cp312-cp312-linux_x86_64.whl
        nvdiffrec_render-0.0.1-cp312-cp312-linux_x86_64.whl
        o_voxel-0.0.1+torch2.8.0.cu128-cp312-cp312-linux_x86_64.whl
        pytorch3d-0.7.9-cp312-cp312-linux_x86_64.whl
        sam3d_objects-0.0.1-py3-none-any.whl
        spconv_cu128-2.4.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
        symtrellis-0.0.1-cp312-cp312-linux_x86_64.whl
        trellis-0.0.1-py3-none-any.whl
        trellis2-0.0.1-py3-none-any.whl
        utils3d-0.0.2-py3-none-any.whl
        xformers-0.0.32.post2-cp39-abi3-manylinux_2_28_x86_64.whl
    '
    ["torch2.9.0-cu130-py312"]='
        wheels-cp312-torch2.9.0-cu130
        cumesh-0.0.1-cp312-cp312-linux_x86_64.whl
        cumm_cu130-0.9.1-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
        diff_gaussian_rasterization-0.0.0-cp312-cp312-linux_x86_64.whl
        diffoctreerast-0.0.1-cp312-cp312-linux_x86_64.whl
        flash_attn-2.8.3.post1-cp312-cp312-linux_x86_64.whl
        flex_gemm-1.0.0-cp312-cp312-linux_x86_64.whl
        gsplat-1.5.3-cp312-cp312-linux_x86_64.whl
        kaolin-0.18.0-cp312-cp312-linux_x86_64.whl
        moge-1.0.0-py3-none-any.whl
        nvdiffrast-0.4.0-cp312-cp312-linux_x86_64.whl
        nvdiffrec_render-0.0.1-cp312-cp312-linux_x86_64.whl
        o_voxel-0.0.1+torch2.9.0.cu130-cp312-cp312-linux_x86_64.whl
        pytorch3d-0.7.9-cp312-cp312-linux_x86_64.whl
        sam3d_objects-0.0.1-py3-none-any.whl
        spconv_cu130-2.4.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
        symtrellis-0.0.1-cp312-cp312-linux_x86_64.whl
        trellis-0.0.1-py3-none-any.whl
        trellis2-0.0.1-py3-none-any.whl
        utils3d-0.0.2-py3-none-any.whl
        xformers-0.0.33.post1-cp39-abi3-manylinux_2_28_x86_64.whl
    '
)

[[ "$#" -eq 1 ]] || {
    echo "Usage: $0 torch2.9.0-cu130-py312" >&2
    exit 2
}
command -v curl >/dev/null 2>&1 || {
    echo "error: curl is required" >&2
    exit 2
}
target="$1"
entry="${MATRIX[$target]:-}"
[[ -n "$entry" ]] || {
    echo "error: unsupported wheel set: $target" >&2
    exit 2
}
read -ra row <<<"${entry//$'\n'/ }"
destination="${WHEELS_ROOT}/${target}"
mkdir -p "$destination"

for filename in "${row[@]:1}"; do
    [[ -e "${destination}/${filename}" ]] && {
        echo "skip: ${filename}"
        continue
    }
    echo "download: ${filename}"
    curl -fL --retry 3 -o "${destination}/${filename}.part" \
        "${RELEASES_URL}/${row[0]}/${filename//+/%2B}"
    mv "${destination}/${filename}.part" "${destination}/${filename}"
done
