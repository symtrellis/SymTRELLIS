import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

from .base import BasePairDataset, format_entry_name


class DirectFileLoadDataset(BasePairDataset):
    """Pair dataset backend that reads latent records directly from zip files.

    Each `shape_paths` entry points to one shape archive. At runtime, this
    backend opens the selected archive and loads the source and destination
    `.npz` latent records needed for one ordered pair sample.
    """

    def __init__(
        self,
        shape_paths: List[Union[str, Path]],
        grid_size: int,
        num_scale: int,
        num_rots: int,
        num_perts: int,
    ):
        """Initialize direct archive loading.

        Args:
            shape_paths: List of shape archive paths. Each archive contains
                transformed latent records named by `format_entry_name`.
            grid_size: Cubic latent grid resolution.
            num_scale: Number of scale variants per shape.
            num_rots: Number of rotation samples per scale.
            num_perts: Number of perturbation samples per rotation.
        """
        self.shape_list = [Path(p) for p in shape_paths]
        num_shapes = len(shape_paths)

        super().__init__(
            grid_size=grid_size,
            num_shapes=num_shapes,
            num_scale=num_scale,
            num_rots=num_rots,
            num_perts=num_perts,
        )

    def read_item(
        self,
        shape_idx: int,
        scale_idx: int,
        rot_idx_src: int,
        pert_idx_src: int,
        rot_idx_dst: int,
        pert_idx_dst: int,
    ) -> Tuple[Dict, Dict]:
        """Load source and destination latent records for one pair.

        Args:
            shape_idx: Shape archive index into `shape_list`.
            scale_idx: Shared scale index for both records.
            rot_idx_src: Source rotation index.
            pert_idx_src: Source perturbation index.
            rot_idx_dst: Destination rotation index.
            pert_idx_dst: Destination perturbation index.

        Returns:
            `(data_src, data_dst)`. Each dictionary contains numpy arrays loaded
            from one `.npz` record. Required keys are `transform` and `feats`;
            sparse records may also contain `coords`.

        Algorithm:
            Open the selected shape archive, format the source and destination
            entry names from their indices, and load each `.npz` member without
            pickle support.
        """

        with zipfile.ZipFile(self.shape_list[shape_idx]) as zf:
            # Source and destination are two transformed records from the same shape archive.
            name_src = format_entry_name(scale_idx, rot_idx_src, pert_idx_src)
            name_dst = format_entry_name(scale_idx, rot_idx_dst, pert_idx_dst)

            with zf.open(name_src) as f:
                with np.load(f, allow_pickle=False) as z:
                    data_src: Dict = {k: z[k] for k in z.files}

            with zf.open(name_dst) as f:
                with np.load(f, allow_pickle=False) as z:
                    data_dst: Dict = {k: z[k] for k in z.files}

        return data_src, data_dst
