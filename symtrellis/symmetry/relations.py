from typing import Sequence, Tuple

import torch

from ..geometry import grid2pos, t_abs2grid


def build_symmetry_relation_inputs(
    coords: torch.Tensor,
    relations: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    grid_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert sample-level symmetry relations to coords-wise mapper inputs.

    This function is an input-format adapter. It takes complete symmetry
    relations for each batch sample and expands them into coords-wise source
    and destination entries used by the mapper and symmetry projector.

    Grid convention: integer latent-grid coordinate `g` denotes the voxel
    center `(g + 0.5) / grid_size - 0.5` in the canonical cube [-0.5, 0.5].
    This convention is used by the latent grids in TRELLIS, TRELLIS.2, and
    SAM3D-Objects. When adapting this code to another voxel-based generation
    model, verify that the model uses the same center offset and normalized
    coordinate range.

    Args:
        coords: Integer tensor with shape [N, 4]. `coords[:, 0]` is the sample
            id, and `coords[:, 1:]` are integer latent-grid coordinates.
        relations: One `(O, t, s)` tuple per sample. `O` has shape [R, 3, 3],
            `t` has shape [R, 3], and `s` has shape [R]. `O` and `t` are
            destination-to-source transforms in normalized position units.
        grid_size: Number of latent-grid cells per axis.

    Returns:
        coords_src: [num_src_entries, 4]. The first column is flattened
            relation id; the last three columns are source grid coordinates.
        coords_dst: [num_dst_entries, 4]. The first column is flattened
            relation id; the last three columns are destination grid
            coordinates.
        rows_src: [num_src_entries]. Row ids into the input `coords` tensor.
        rows_dst: [num_dst_entries]. Row ids into the input `coords` tensor.
        O_dst2src: [num_relations, 3, 3]. Flattened transforms.
        t_dst2src: [num_relations, 3]. Flattened grid-space translations.
        s: [num_relations]. Flattened symmetry signs/types.
    """
    coords_src_list = []
    coords_dst_list = []
    rows_src_list = []
    rows_dst_list = []
    O_dst2src_list = []
    t_dst2src_list = []
    s_list = []

    relation_offset = 0
    for sample_id, (O_sample, t_sample, s_sample) in enumerate(relations):
        # Select rows belonging to this batch sample; these row ids are kept for projection.
        sample_rows = (coords[:, 0] == sample_id).nonzero(as_tuple=False).flatten()
        sample_grid = coords[sample_rows, 1:]
        sample_pos = grid2pos(sample_grid, grid_size)

        for local_relation_id in range(O_sample.shape[0]):
            # Flatten (sample_id, local_relation_id) into one mapper condition id.
            relation_id = relation_offset + local_relation_id
            O_dst2src = O_sample[local_relation_id]
            t_dst2src = t_sample[local_relation_id]

            # Keep only source/destination rows whose transformed positions stay in the canonical cube.
            pos_dst_in_src = sample_pos @ O_dst2src.T + t_dst2src[None, :]
            pos_src_in_dst = (sample_pos - t_dst2src[None, :]) @ O_dst2src

            mask_dst = torch.all((pos_dst_in_src >= -0.5) & (pos_dst_in_src <= 0.5), dim=1)
            mask_src = torch.all((pos_src_in_dst >= -0.5) & (pos_src_in_dst <= 0.5), dim=1)

            rows_src = sample_rows[mask_src]
            rows_dst = sample_rows[mask_dst]
            grid_src = sample_grid[mask_src]
            grid_dst = sample_grid[mask_dst]

            relation_col_src = torch.full((grid_src.shape[0], 1), relation_id, device=coords.device, dtype=coords.dtype)
            relation_col_dst = torch.full((grid_dst.shape[0], 1), relation_id, device=coords.device, dtype=coords.dtype)

            # Mapper coordinates use relation id in column 0, not the original sample id.
            coords_src_list.append(torch.cat([relation_col_src, grid_src], dim=1))
            coords_dst_list.append(torch.cat([relation_col_dst, grid_dst], dim=1))
            rows_src_list.append(rows_src)
            rows_dst_list.append(rows_dst)

        # Mapper applies transforms directly in grid-index space.
        O_dst2src_list.append(O_sample)
        t_dst2src_list.append(t_abs2grid(t_sample, O_sample, grid_size))
        s_list.append(s_sample)
        relation_offset += O_sample.shape[0]

    coords_src = torch.cat(coords_src_list, dim=0)
    coords_dst = torch.cat(coords_dst_list, dim=0)
    rows_src = torch.cat(rows_src_list, dim=0)
    rows_dst = torch.cat(rows_dst_list, dim=0)
    O_dst2src = torch.cat(O_dst2src_list, dim=0)
    t_dst2src = torch.cat(t_dst2src_list, dim=0)
    s = torch.cat(s_list, dim=0)

    return coords_src, coords_dst, rows_src, rows_dst, O_dst2src, t_dst2src, s
