import os

from setuptools import find_packages, setup

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError as exc:
    raise RuntimeError("Building SymTRELLIS CUDA extensions must use the PyTorch installed in " "the current conda environment. pip build isolation hides that PyTorch. " "Run `pip install -e . --no-build-isolation` instead.") from exc


os.environ.setdefault("MAX_JOBS", str(os.cpu_count() or 1))


setup(
    name="symtrellis",
    packages=find_packages(),
    install_requires=[
        "torch",
    ],
    ext_modules=[
        CUDAExtension(
            name="symtrellis.geometry.neighbors._sparse_lattice_ext",
            sources=[
                "symtrellis/geometry/neighbors/sparse_lattice_ext/bindings.cpp",
                "symtrellis/geometry/neighbors/sparse_lattice_ext/radius_nbr_edges_sparse_lattice_cpu.cpp",
                "symtrellis/geometry/neighbors/sparse_lattice_ext/radius_nbr_edges_sparse_lattice.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
