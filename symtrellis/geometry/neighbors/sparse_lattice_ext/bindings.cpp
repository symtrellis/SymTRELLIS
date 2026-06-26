#include <torch/extension.h>

#include "radius_nbr_edges_sparse_lattice.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "radius_nbr_kids_by_offset_sparse_lattice_cuda",
      &radius_nbr_kids_by_offset_sparse_lattice_cuda,
      "Sparse lattice radius-neighbor kids-by-offset, CUDA backend");
}
