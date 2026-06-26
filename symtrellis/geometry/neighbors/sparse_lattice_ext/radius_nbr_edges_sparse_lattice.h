#pragma once

#include <torch/types.h>

torch::Tensor radius_nbr_kids_by_offset_sparse_lattice_cuda(
    torch::Tensor query_pos,
    torch::Tensor query_bid,
    torch::Tensor key_coords,
    torch::Tensor key_bid,
    double radius,
    torch::Tensor nbr_offsets,
    torch::Tensor coord_min);
