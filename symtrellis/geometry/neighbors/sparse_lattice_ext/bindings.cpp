#include <torch/extension.h>

#include "radius_nbr_edges_sparse_lattice.h"

// Thin pybind layer for the installed sparse-lattice extension module:
// symtrellis.geometry.neighbors.sparse_lattice_ext._C.
//
// The Python wrapper owns validation, contiguous conversion, and conversion
// from the dense [Nq, L] kids-by-offset table to COO edge lists. These bindings
// expose only the CPU/CUDA implementations that compute that table.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "radius_nbr_kids_by_offset_sparse_lattice_cpu",
        &radius_nbr_kids_by_offset_sparse_lattice_cpu,
        R"doc(
Return the CPU kids-by-offset table for sparse lattice radius neighbors.

Inputs follow radius_nbr_edges_sparse_lattice(...). The result is an int64
tensor with shape [Nq, L]; each entry is a key row id, or -1 when the offset
candidate has no valid key within radius.
)doc");
    m.def(
        "radius_nbr_kids_by_offset_sparse_lattice_cuda",
        &radius_nbr_kids_by_offset_sparse_lattice_cuda,
        R"doc(
Return the CUDA kids-by-offset table for sparse lattice radius neighbors.

Inputs follow radius_nbr_edges_sparse_lattice(...). The result is an int64
tensor with shape [Nq, L]; each entry is a key row id, or -1 when the offset
candidate has no valid key within radius.
)doc");
}
