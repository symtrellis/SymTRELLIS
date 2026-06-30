import re
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from torch.utils.data import Dataset


def build_catalog(
    dataset_dirs: List[Union[str, Path]],
    seed: int = 114514,
    chunk: int = 1,
    chunk_id: int = 0,
) -> List[str]:
    """Build a deterministic list of shape archive paths.

    Each shape archive is a `.zip` file containing multiple latent records for
    the same object under different scale/rotation/perturbation settings. This
    function collects all archives from `dataset_dirs`, shuffles them with a
    fixed numpy RNG seed, and optionally returns one contiguous shard.

    Args:
        dataset_dirs: List of directories. Each directory is scanned for
            top-level `*.zip` files. Elements may be `str` or `Path`.
        seed: Seed for the deterministic archive-level permutation.
        chunk: Number of contiguous shards to split the shuffled catalog into.
            `1` returns the full catalog.
        chunk_id: Shard index in `[0, chunk)`.

    Returns:
        A list of absolute archive paths represented as strings. The order is
        deterministic for a fixed set of input files and `seed`.

    Algorithm:
        1. Resolve each dataset directory and collect sorted `*.zip` files.
        2. Apply one deterministic random permutation to all collected paths.
        3. If requested, split the permuted list into nearly equal contiguous
           chunks and return the selected chunk.
    """
    if chunk <= 0:
        raise ValueError("chunk must be positive")
    if chunk_id < 0 or chunk_id >= chunk:
        raise ValueError("chunk_id must satisfy 0 <= chunk_id < chunk")

    shape_paths = []
    for dataset_dir in dataset_dirs:
        dataset_dir = Path(dataset_dir).expanduser().resolve()
        shape_paths.extend(sorted(str(p) for p in dataset_dir.glob("*.zip")))

    # Shuffle at archive granularity so all records of one shape stay together.
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(shape_paths))
    shape_paths = [shape_paths[i] for i in perm]

    if chunk > 1:
        # Balanced contiguous partition: the first `r` chunks receive one extra item.
        q, r = divmod(len(shape_paths), chunk)
        start = chunk_id * q + min(chunk_id, r)
        end = start + q + (chunk_id < r)
        shape_paths = shape_paths[start:end]

    return shape_paths


ENTRY_FORMAT = "scale_{scale:04d}_rot_{rot:04d}_pert_{pert:04d}.npz"
ENTRY_NAME_RE = re.compile(r"^scale_(\d+)_rot_(\d+)_pert_(\d+)$")


def format_entry_name(scale_idx: int, rot_idx: int, pert_idx: int) -> str:
    """Format one latent record name inside a shape archive.

    A shape archive stores one `.npz` file for each transformed copy of the
    shape. The three integer indices identify the scale bucket, rotation sample,
    and perturbation sample used when the latent record was produced.

    Args:
        scale_idx: Integer scale index.
        rot_idx: Integer rotation index.
        pert_idx: Integer perturbation index.

    Returns:
        The archive member name, e.g. `scale_0000_rot_0003_pert_0001.npz`.

    Algorithm:
        Interpolate the three indices into the canonical fixed-width entry
        format used by both preprocessing writers and runtime readers.
    """
    return ENTRY_FORMAT.format(scale=scale_idx, rot=rot_idx, pert=pert_idx)


def parse_entry_stem(stem: str) -> Tuple[int, int, int]:
    """Parse a latent record stem into scale, rotation, and perturbation ids.

    The input is the file stem without `.npz`, for example
    `scale_0000_rot_0003_pert_0001`. These ids are the inverse of
    `format_entry_name` and define the archive schema.

    Args:
        stem: Archive member stem with shape
            `scale_<int>_rot_<int>_pert_<int>`.

    Returns:
        A tuple `(scale_idx, rot_idx, pert_idx)`, all Python integers.

    Raises:
        ValueError: If `stem` does not match the canonical entry format.

    Algorithm:
        Match the whole stem with one regex, then convert the three captured
        integer fields.
    """
    match = ENTRY_NAME_RE.fullmatch(stem)
    if match is None:
        raise ValueError(f"invalid dataset entry stem: {stem!r}")
    scale_idx, rot_idx, pert_idx = (int(x) for x in match.groups())
    return scale_idx, rot_idx, pert_idx


def build_pair_sample(
    data_src: Dict[str, np.ndarray],
    data_dst: Dict[str, np.ndarray],
    grid_size: int = 64,
) -> Dict[str, np.ndarray]:
    """Build one source/destination pair sample from two latent records.

    `data_src` and `data_dst` are two transformed latent records from the same
    shape and scale. Each record stores latent features in either sparse form
    (`coords`, `feats`) or dense form (`feats` only), plus a 4x4 homogeneous
    transform describing that transformed copy.

    Physical meaning:
        The returned transform maps destination latent positions into the
        source latent coordinate frame:

            pos_src = pos_dst @ O_dst2src.T + t_dst2src

        `s_dst2src` is an orientation token: `1` for orientation preserving and
        `0` for orientation reversing.

    Grid convention:
        Integer coordinate `g` represents the cell center
        `(g + 0.5) / grid_size - 0.5`, matching `symtrellis.geometry.grid2pos`.
        Positions therefore live in the normalized cube `[-0.5, 0.5]^3`.

    Args:
        data_src: Source latent record. Required keys are:
            `transform`: float numpy array with shape `[4, 4]`;
            `feats`: sparse `[Nsrc, C]` or dense `[C, D, H, W]`;
            optional `coords`: integer numpy array with shape `[Nsrc, 3]`.
        data_dst: Destination latent record with the same schema as `data_src`.
        grid_size: Cubic latent grid resolution.

    Returns:
        A dictionary containing:
            `coords_src`: integer numpy array `[Nsrc_visible, 3]`;
            `coords_dst`: integer numpy array `[Ndst_visible, 3]`;
            `feats_src`: float numpy array `[Nsrc_visible, C]`;
            `feats_dst`: float numpy array `[Ndst_visible, C]`;
            `O_dst2src`: float numpy array `[3, 3]`;
            `t_dst2src`: float numpy array `[3]`;
            `s_dst2src`: int numpy array `[1]`;
            `grid_size`: int numpy array `[1]`.

    Algorithm:
        1. Convert dense latent records to sparse `(coords, feats)` when needed.
        2. Remove uniform scale from each stored transform and compute the
           destination-to-source relative orthogonal transform.
        3. Convert sparse grid coordinates to normalized positions.
        4. Keep only source/destination coordinates whose counterpart remains
           inside the canonical cube under the relative transform.
    """

    def latent_to_sparse(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Return sparse coordinates and row-aligned features for one record.

        Args:
            data: One latent record. `feats` is either sparse `[N, C]` when
                `coords` exists, or dense `[C, D, H, W]` when `coords` is
                absent.

        Returns:
            `(coords, feats)`, where `coords` is integer `[N, 3]` and `feats`
            is row-aligned `[N, C]`.

        Algorithm:
            Sparse records are returned directly. Dense records are enumerated
            over the full spatial grid and features are moved from channel-first
            dense layout to row-major sparse layout.
        """
        feats = data["feats"]
        if "coords" in data:
            return data["coords"], feats

        feat_dim = feats.shape[0]
        spatial_shape = feats.shape[1:]
        n_grid = feats[0].size

        coords = np.stack(
            np.unravel_index(np.arange(n_grid), spatial_shape),
            axis=-1,
        )
        feats = feats.transpose(
            *range(1, feats.ndim),
            0,
        ).reshape(-1, feat_dim)

        return coords, feats

    transform_src = data_src["transform"]
    transform_dst = data_dst["transform"]

    # Stored transforms include isotropic scale; remove it before producing the
    # orthogonal destination-to-source relation consumed by the mapper.
    scale_src = np.cbrt(np.abs(np.linalg.det(transform_src[:3, :3])))
    scale_dst = np.cbrt(np.abs(np.linalg.det(transform_dst[:3, :3])))

    O_dst2src = transform_src[:3, :3] @ transform_dst[:3, :3].T / scale_src / scale_dst
    t_dst2src = transform_src[:3, 3] - O_dst2src @ transform_dst[:3, 3]

    # Orientation token follows the mapper convention: 1 means det(O) >= 0,
    # 0 means an orientation-reversing transform.
    s_dst2src = np.array([1 if np.linalg.det(O_dst2src) >= 0 else 0])
    grid_size_arr = np.array([grid_size])

    coords_src, feats_src = latent_to_sparse(data_src)
    coords_dst, feats_dst = latent_to_sparse(data_dst)

    # Convert grid indices to normalized cell-center positions in [-0.5, 0.5]^3.
    pos_src = (coords_src.astype(np.float64) + 0.5) / grid_size - 0.5
    pos_dst = (coords_dst.astype(np.float64) + 0.5) / grid_size - 0.5

    # Visibility masks discard positions whose transformed counterpart would
    # leave the canonical cube. This keeps both branches valid for local mapping.
    pos_dst_in_src = pos_dst @ O_dst2src.T + t_dst2src[None, :]
    pos_src_in_dst = (pos_src - t_dst2src[None, :]) @ O_dst2src

    mask_dst = np.all((pos_dst_in_src >= -0.5) & (pos_dst_in_src <= 0.5), axis=1)
    mask_src = np.all((pos_src_in_dst >= -0.5) & (pos_src_in_dst <= 0.5), axis=1)

    coords_src = coords_src[mask_src]
    feats_src = feats_src[mask_src]
    coords_dst = coords_dst[mask_dst]
    feats_dst = feats_dst[mask_dst]

    return {
        "coords_src": coords_src,
        "coords_dst": coords_dst,
        "feats_src": feats_src,
        "feats_dst": feats_dst,
        "O_dst2src": O_dst2src,
        "t_dst2src": t_dst2src,
        "s_dst2src": s_dst2src,
        "grid_size": grid_size_arr,
    }


class BasePairDataset(Dataset):
    """Base class for pair-indexed SymTRELLIS latent datasets.

    Physical meaning:
        A dataset element is an ordered `(source, destination)` pair sampled
        from transformed latent records of the same shape. The pair provides
        sparse source/destination features and the destination-to-source
        transform needed by the mapper.

    Index layout:
        A linear dataset index is decoded as:
            `shape_idx`, `scale_idx`, `trans_idx_src`, `trans_idx_dst`.
        Each transform index is further decoded into
            `rot_idx_*`, `pert_idx_*`.

    Subclasses implement `read_item`, which loads the two raw latent records
    from a storage backend such as zip files or shared memory.
    """

    def __init__(
        self,
        grid_size: int,
        num_shapes: int,
        num_scale: int,
        num_rots: int,
        num_perts: int,
        **kwargs,
    ):
        """Initialize the pair index space.

        Args:
            grid_size: Cubic latent grid resolution.
            num_shapes: Number of shape archives in this dataset.
            num_scale: Number of scale variants per shape.
            num_rots: Number of rotation samples per scale.
            num_perts: Number of perturbation samples per rotation.
            **kwargs: Reserved for subclass compatibility.

        Attributes:
            num_trans: Number of rotation/perturbation transformed records per
                scale, equal to `num_rots * num_perts`.
            num_pairs: Number of ordered source/destination pairs per shape,
                equal to `num_scale * num_trans * num_trans`.
            length: Total number of linear pair indices.
        """
        self.grid_size = int(grid_size)
        self.num_shapes = int(num_shapes)
        self.num_scale = int(num_scale)
        self.num_rots = int(num_rots)
        self.num_perts = int(num_perts)

        self.num_trans = self.num_rots * self.num_perts
        self.num_pairs = self.num_scale * self.num_trans * self.num_trans
        self.length = self.num_shapes * self.num_pairs

    def __len__(self) -> int:
        """Return the number of ordered pair samples.

        Returns:
            Total dataset length as a Python integer.

        Algorithm:
            The value is precomputed in `__init__` from the full Cartesian
            product of shapes, scales, source transforms, and destination
            transforms.
        """
        return self.length

    def read_item(
        self,
        shape_idx: int,
        scale_idx: int,
        rot_idx_src: int,
        pert_idx_src: int,
        rot_idx_dst: int,
        pert_idx_dst: int,
    ) -> Tuple[Dict, Dict]:
        """Load two raw latent records for one ordered pair.

        Args:
            shape_idx: Shape archive index.
            scale_idx: Shared scale index for both records.
            rot_idx_src: Source rotation index.
            pert_idx_src: Source perturbation index.
            rot_idx_dst: Destination rotation index.
            pert_idx_dst: Destination perturbation index.

        Returns:
            `(data_src, data_dst)`. Each item is a dictionary accepted by
            `build_pair_sample`: it must contain `transform` and `feats`, and
            may contain sparse `coords`.

        Algorithm:
            Subclasses choose the storage backend and implement the lookup.
            This base method only defines the interface.
        """
        raise NotImplementedError

    def __getitem__(self, idx) -> Dict[str, np.ndarray]:
        """Decode one linear index and build the corresponding pair sample.

        Args:
            idx: Linear dataset index in `[0, len(self))`. It may be any integer
                type supported by floor division and modulo.

        Returns:
            The normalized pair sample dictionary produced by
            `build_pair_sample`.

        Algorithm:
            1. Decode `idx` into shape, scale, source transform, and destination
               transform indices.
            2. Decode each transform index into rotation and perturbation ids.
            3. Ask the subclass to load both raw records.
            4. Convert the raw records into the canonical pair sample schema.
        """
        shape_idx = idx // self.num_pairs
        rest = idx % self.num_pairs

        scale_idx = rest // (self.num_trans**2)
        rest = rest % (self.num_trans**2)

        trans_idx_src = rest // self.num_trans
        trans_idx_dst = rest % self.num_trans

        rot_idx_src = trans_idx_src // self.num_perts
        pert_idx_src = trans_idx_src % self.num_perts

        rot_idx_dst = trans_idx_dst // self.num_perts
        pert_idx_dst = trans_idx_dst % self.num_perts

        data_src, data_dst = self.read_item(
            shape_idx,
            scale_idx,
            rot_idx_src,
            pert_idx_src,
            rot_idx_dst,
            pert_idx_dst,
        )

        return build_pair_sample(
            data_src=data_src,
            data_dst=data_dst,
            grid_size=self.grid_size,
        )
