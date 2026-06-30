from typing import Any, Iterator, List, Union

import numpy as np
import numpy.typing as npt
from torch.utils.data import Sampler

_MASK64 = np.uint64((1 << 64) - 1)

_C1 = np.uint64(0x9E3779B97F4A7C15)
_C2 = np.uint64(0xBF58476D1CE4E5B9)
_C3 = np.uint64(0x94D049BB133111EB)


def mix64_np(values: np.ndarray) -> np.ndarray:
    """Mix uint64 values with a splitmix64-style avalanche function.

    Args:
        values: `np.uint64` array of any shape. The operation is elementwise.

    Returns:
        A `np.uint64` array with the same shape as `values`.

    Physical meaning:
        This has no geometric meaning. It is a deterministic pseudo-random hash
        used to derive independent sampler seeds from shape, scale, and pair
        family indices.

    Algorithm:
        Add one Weyl sequence constant, then apply the standard xor-shift and
        multiply avalanche steps in unsigned 64-bit arithmetic.
    """
    with np.errstate(over="ignore"):
        values = (values + _C1) & _MASK64
        values = values ^ (values >> np.uint64(30))
        values = (values * _C2) & _MASK64
        values = values ^ (values >> np.uint64(27))
        values = (values * _C3) & _MASK64
        values = values ^ (values >> np.uint64(31))
    return values & _MASK64


def feistel_prp_np(
    values: np.ndarray,
    seed: np.ndarray,
    m_bits: int,
    rounds: Union[int, np.uint64] = 4,
) -> np.ndarray:
    """Evaluate a deterministic Feistel permutation on uint64 arrays.

    Args:
        values: `np.uint64` array with elements in `[0, 2**m_bits)`.
        seed: `np.uint64` array broadcasted to the same shape as `values`.
        m_bits: Even number of active low bits in the permutation domain.
        rounds: Number of Feistel rounds.

    Returns:
        `np.uint64` array with the same shape as `values`. Each element is the
        permuted value in `[0, 2**m_bits)`.

    Physical meaning:
        This is an index permutation primitive. It does not represent a physical
        transform; it makes per-shape and per-scale visits deterministic without
        materializing large permutations.

    Algorithm:
        Split each `m_bits` value into equal low/high halves. Each Feistel round
        mixes the right half with a round key derived from `seed`, xors that into
        the left half, and swaps halves. The Feistel structure is a permutation
        over the power-of-two domain.
    """
    if m_bits <= 0:
        raise ValueError("m_bits must be >= 1")
    if (m_bits & 1) != 0:
        raise ValueError("m_bits must be even")
    if rounds <= 0:
        raise ValueError("rounds must be >= 1")

    half = m_bits // 2
    half_mask = np.uint64((1 << half) - 1)
    dom_mask = np.uint64((1 << m_bits) - 1)

    left = values & half_mask
    right = (values >> np.uint64(half)) & half_mask

    for round_idx in range(int(rounds)):
        with np.errstate(over="ignore"):
            round_key = seed + _C1 * np.uint64(round_idx)
        round_value = mix64_np(right ^ round_key) & half_mask
        left, right = right, (left ^ round_value) & half_mask

    out = ((right << np.uint64(half)) | left) & dom_mask
    return out


def rand_perm_n_at(
    position: np.ndarray,
    seed: np.ndarray,
    n: int,
    rounds: Union[int, np.uint64] = 4,
) -> np.ndarray:
    """Return values of a deterministic permutation of `[0, n)`.

    Args:
        position: Integer array of any shape. Every element must satisfy
            `0 <= position < n`.
        seed: Integer array broadcastable to `position`. Equal seeds define the
            same permutation; different seeds define independent permutations.
        n: Permutation domain size.
        rounds: Number of Feistel rounds used by the power-of-two permutation.

    Returns:
        Integer array with the same shape and dtype as `position`. Each element
        lies in `[0, n)`.

    Physical meaning:
        In this sampler, the function maps visit positions to shape, scale,
        rotation, or perturbation ids without storing a full permutation table.

    Algorithm:
        Embed `[0, n)` into the next even-bit power-of-two domain, apply a
        Feistel permutation, and use cycle walking until the result falls back
        into `[0, n)`.
    """

    if n <= 0:
        raise ValueError("n must be >= 1")
    if rounds <= 0:
        raise ValueError("rounds must be >= 1")

    positions = np.asarray(position)
    seeds = np.asarray(seed)

    if positions.dtype.kind not in ("i", "u"):
        raise TypeError("position must be an integer array")
    if seeds.dtype.kind not in ("i", "u"):
        raise TypeError("seed must be an integer array")

    positions, seeds = np.broadcast_arrays(positions, seeds)

    if np.any(positions < 0) or np.any(positions >= n):
        raise ValueError("position must satisfy 0 <= position < n elementwise")

    values = positions.astype(np.uint64, copy=False)
    seed_values = seeds.astype(np.uint64, copy=False)

    m = max(1, (n - 1).bit_length())
    if (m & 1) != 0:
        m += 1

    n_u = np.uint64(n)

    permuted = values.copy()
    outside_domain = np.ones(permuted.shape, dtype=bool)

    while True:
        permuted_outside = feistel_prp_np(
            permuted[outside_domain],
            seed_values[outside_domain],
            m_bits=m,
            rounds=rounds,
        )
        permuted[outside_domain] = permuted_outside

        outside_domain = permuted >= n_u
        if not np.any(outside_domain):
            return permuted.astype(positions.dtype, copy=False)


def decode_ordered_neq(
    pair_idx: npt.NDArray[np.uint64],
    n: int,
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]:
    """Decode an ordered non-equal pair from a compact linear id.

    Args:
        pair_idx: `np.uint64` array with elements in `[0, n * (n - 1))`.
        n: Number of possible ids in each side of the pair. Must be at least 2.

    Returns:
        `(idx_src, idx_dst)`, two `np.uint64` arrays with the same shape as
        `pair_idx`. Each element satisfies `idx_src != idx_dst`.

    Physical meaning:
        The returned ordered pair is used as source/destination indices for
        perturbations in the same-rotation family, and rotations in the
        different-rotation family.

    Algorithm:
        View the domain as `n` rows with `n - 1` valid destinations per source.
        Decode the row as source. Decode the column as destination, then skip
        the diagonal by shifting destinations greater than or equal to source.
    """
    if n <= 1:
        raise ValueError("n must be >= 2")

    stride = np.uint64(n - 1)
    idx_src = pair_idx // stride
    idx_dst = pair_idx % stride
    idx_dst = idx_dst + (idx_dst >= idx_src).astype(np.uint64, copy=False)
    return idx_src, idx_dst


_TAG_SCALE = np.uint64(0x13579BDF2468ACE1)
_TAG_MODE = np.uint64(0x9E3779B97F4A7C15)
_TAG_SAME_ROT = np.uint64(0xA24BAED4963EE407)
_TAG_SAME_PERT = np.uint64(0x3C79AC492BA7B653)
_TAG_DIFF_ROT = np.uint64(0x1C69B3F74AC4AE35)
_TAG_DIFF_PERT = np.uint64(0xD1B54A32D192ED03)


def as_uint64(value: Any) -> np.uint64:
    """Convert a scalar value to `np.uint64`.

    Args:
        value: Python or numpy scalar used in sampler index arithmetic.

    Returns:
        The same value represented as `np.uint64`.

    Physical meaning:
        The sampler intentionally uses unsigned 64-bit arithmetic for seed
        mixing and deterministic wrap-around behavior.
    """
    return np.uint64(value)


class MultiScaleMixedPairSampler(Sampler[List[int]]):
    """Family-aware pair sampler with deterministic quota scheduling.

    Pair families per scale:
      1) same_rot_diff_pert:
         choose one rot, then choose an ordered pert pair inside that rot.
         size = num_rots * num_perts * (num_perts - 1)

      2) diff_rot_any_pert:
         choose an ordered rot pair, then choose an ordered pert pair.
         size = num_rots * (num_rots - 1) * num_perts * num_perts

    Notes:
      - shape permutation logic stays position-based and DDP-compatible.
      - scale is also chosen by a per-shape permutation.
      - family choice uses deterministic quota scheduling with an offset mixed
        from `(seed, shape_idx, scale_idx)`, not Bernoulli sampling.
      - output is the linear dataset index consumed by `BasePairDataset`:
            dataset_index = shape_idx * num_pairs + pair_id
        where
            pair_id = scale_id * (num_trans * num_trans) + trans_src * num_trans + trans_dst
      - this sampler never emits diagonal pairs with trans_src == trans_dst.

    Args:
        num_shapes: Number of shape archives in the dataset.
        num_scale: Number of scale variants per shape.
        num_rots: Number of rotation samples per scale.
        num_perts: Number of perturbation samples per rotation.
        batch_size: Global batch size before rank partitioning.
        num_batch_per_epoch: Number of batches yielded by each epoch.
        rank: DDP rank id.
        world_size: Number of DDP ranks.
        seed: Base deterministic seed.
        same_rot_diff_pert_ratio: Quota ratio assigned to same-rotation,
            different-perturbation pairs. Must be in `[0, 1]`.

    Yields:
        Lists of Python `int` dataset indices. Each list is the rank-local
        portion of one global batch.

    Physical meaning:
        Each emitted integer selects one ordered source/destination latent pair
        from one shape and one scale. The source/destination indices are later
        decoded by `BasePairDataset.__getitem__`.

    Algorithm:
        For each global batch slot, deterministically permute shape ids, then
        scale ids per shape. A quota schedule chooses either the same-rotation
        family or the different-rotation family. Family-local counters are
        permuted into rotation/perturbation source/destination ids and encoded
        into the linear dataset index.
    """

    def __init__(
        self,
        num_shapes: int,
        num_scale: int,
        num_rots: int,
        num_perts: int,
        batch_size: int,
        num_batch_per_epoch: int,
        rank: int,
        world_size: int,
        seed: int = 114514,
        same_rot_diff_pert_ratio: float = 0.0,
    ) -> None:
        """Initialize deterministic pair sampling state.

        Args:
            num_shapes: Number of shape archives in the dataset.
            num_scale: Number of scale variants per shape.
            num_rots: Number of rotation samples per scale.
            num_perts: Number of perturbation samples per rotation.
            batch_size: Global batch size before rank partitioning.
            num_batch_per_epoch: Number of batches yielded by each epoch.
            rank: Current DDP rank.
            world_size: Total number of DDP ranks.
            seed: Base seed for every deterministic permutation.
            same_rot_diff_pert_ratio: Fraction of pair visits assigned to the
                same-rotation, different-perturbation family.

        Returns:
            None. The sampler stores integer dimensions, family sizes, rank
            partitioning, and quota weights.
        """
        if not 0.0 <= same_rot_diff_pert_ratio <= 1.0:
            raise ValueError("same_rot_diff_pert_ratio must satisfy 0 <= ratio <= 1")

        self.num_shapes = as_uint64(num_shapes)
        self.num_scale = as_uint64(num_scale)
        self.num_rots = as_uint64(num_rots)
        self.num_perts = as_uint64(num_perts)

        self.num_trans = as_uint64(self.num_rots * self.num_perts)
        self.num_pairs = as_uint64(self.num_scale * self.num_trans * self.num_trans)

        # Family sizes are counted per scale.
        self.same_per_rot = as_uint64(self.num_perts * (self.num_perts - 1))
        self.same_family_size = as_uint64(self.num_rots * self.same_per_rot)

        self.diff_per_rotpair = as_uint64(self.num_perts * self.num_perts)
        self.diff_rotpair_count = as_uint64(self.num_rots * (self.num_rots - 1))
        self.diff_family_size = as_uint64(self.diff_rotpair_count * self.diff_per_rotpair)

        if self.same_family_size == 0 and self.diff_family_size == 0:
            raise ValueError("sampler requires at least one non-diagonal pair family")

        # Disable impossible families, then require positive weight on a valid one.
        weight_scale = 2**32
        req_w_same = as_uint64(round(same_rot_diff_pert_ratio * weight_scale))
        req_w_diff = as_uint64(weight_scale - req_w_same)
        self.weight_same = req_w_same if self.same_family_size > 0 else as_uint64(0)
        self.weight_diff = req_w_diff if self.diff_family_size > 0 else as_uint64(0)
        self.weight_total = self.weight_same + self.weight_diff
        if self.weight_total == 0:
            raise ValueError("same_rot_diff_pert_ratio assigns no weight to any valid pair family")

        self.batch_size = as_uint64(batch_size)
        self.rank = as_uint64(rank)
        self.world_size = as_uint64(world_size)

        q, r = divmod(self.batch_size, self.world_size)
        self.length = q + (self.rank < r)
        self.batch_start = self.rank * q + min(self.rank, r)

        self.num_batch_per_epoch = as_uint64(num_batch_per_epoch)
        self.seed = as_uint64(seed)
        self.iter_count: int = 0

    def set_iter_count(self, iter_count: int = 0):
        """Set the global batch counter used by deterministic iteration.

        Args:
            iter_count: Number of global batches already consumed.

        Returns:
            None. The next `__iter__` call starts from this counter.
        """
        self.iter_count = iter_count

    def __len__(self) -> int:
        """Return the number of rank-local batches per epoch.

        Returns:
            `num_batch_per_epoch` as a Python integer.
        """
        return int(self.num_batch_per_epoch)

    def __iter__(self) -> Iterator[List[int]]:
        """Yield rank-local lists of linear dataset indices.

        Yields:
            A list of Python integers. Each integer encodes
            `(shape_idx, scale_idx, trans_idx_src, trans_idx_dst)` in the
            `BasePairDataset` linear index space.

        Algorithm:
            Compute the global batch slots owned by this rank, generate shape
            and scale ids by deterministic position permutations, choose pair
            family by quota scheduling, decode source/destination transform ids,
            and emit encoded dataset indices.
        """
        for _ in range(int(self.num_batch_per_epoch)):
            start = self.iter_count * self.batch_size + self.batch_start
            end = start + self.length
            instance_pos = np.arange(start, end, dtype=np.uint64)

            # Step 1: sample shape ids, then choose one scale for each shape visit.
            loop_count = instance_pos // self.num_shapes
            loop_pos = instance_pos % self.num_shapes
            loop_seed = loop_count + self.seed
            shape_ids = rand_perm_n_at(
                loop_pos,
                loop_seed,
                n=int(self.num_shapes),
            )

            scale_epoch = loop_count // self.num_scale
            scale_pos = loop_count % self.num_scale
            scale_seed_input = self.seed ^ shape_ids ^ ((scale_epoch * np.uint64(0x100000001B3)) ^ _TAG_SCALE)
            scale_seed = mix64_np(scale_seed_input)
            scale_ids = rand_perm_n_at(
                scale_pos,
                scale_seed,
                n=int(self.num_scale),
            )

            # Step 2: deterministically choose pair family by quota schedule.
            mode_seed_input = self.seed ^ shape_ids ^ (scale_ids * np.uint64(0x9E3779B1)) ^ _TAG_MODE
            mode_offset = mix64_np(mode_seed_input) % self.weight_total
            shifted = scale_epoch + mode_offset

            same_before = (shifted * self.weight_same) // self.weight_total
            same_after = ((shifted + as_uint64(1)) * self.weight_same) // self.weight_total
            use_same = same_after > same_before

            # Family-local visit counters before the current sample.
            diff_before = shifted - same_before

            if self.same_family_size > 0:
                same_epoch = same_before // self.same_family_size
                same_local = same_before % self.same_family_size

                same_rot_block = same_local // self.same_per_rot
                same_pert_block = same_local % self.same_per_rot

                same_rot_seed_input = self.seed ^ shape_ids ^ (scale_ids * np.uint64(0x100000001B3)) ^ (same_epoch * np.uint64(0xD6E8FEB86659FD93)) ^ _TAG_SAME_ROT
                same_rot_seed = mix64_np(same_rot_seed_input)
                same_rot = rand_perm_n_at(
                    same_rot_block,
                    same_rot_seed,
                    n=int(self.num_rots),
                )

                same_pert_seed_input = self.seed ^ shape_ids ^ (scale_ids * np.uint64(0x9E3779B1)) ^ (same_epoch * np.uint64(0x94D049BB133111EB)) ^ (same_rot * np.uint64(0xBF58476D1CE4E5B9)) ^ _TAG_SAME_PERT
                same_pert_seed = mix64_np(same_pert_seed_input)
                same_pert_pair = rand_perm_n_at(
                    same_pert_block,
                    same_pert_seed,
                    n=int(self.same_per_rot),
                )
                same_pert_src, same_pert_dst = decode_ordered_neq(same_pert_pair, int(self.num_perts))

                same_trans_src = same_rot * self.num_perts + same_pert_src
                same_trans_dst = same_rot * self.num_perts + same_pert_dst
            else:
                same_trans_src = np.zeros_like(shape_ids, dtype=np.uint64)
                same_trans_dst = np.zeros_like(shape_ids, dtype=np.uint64)

            if self.diff_family_size > 0:
                diff_epoch = diff_before // self.diff_family_size
                diff_local = diff_before % self.diff_family_size

                diff_rotpair_block = diff_local // self.diff_per_rotpair
                diff_pert_block = diff_local % self.diff_per_rotpair

                diff_rot_seed_input = self.seed ^ shape_ids ^ (scale_ids * np.uint64(0x100000001B3)) ^ (diff_epoch * np.uint64(0xD6E8FEB86659FD93)) ^ _TAG_DIFF_ROT
                diff_rot_seed = mix64_np(diff_rot_seed_input)
                diff_rotpair = rand_perm_n_at(
                    diff_rotpair_block,
                    diff_rot_seed,
                    n=int(self.diff_rotpair_count),
                )
                diff_rot_src, diff_rot_dst = decode_ordered_neq(diff_rotpair, int(self.num_rots))

                diff_pert_seed_input = (
                    self.seed ^ shape_ids ^ (scale_ids * np.uint64(0x9E3779B1)) ^ (diff_epoch * np.uint64(0x94D049BB133111EB)) ^ (diff_rot_src * np.uint64(0xBF58476D1CE4E5B9)) ^ (diff_rot_dst * np.uint64(0x4CF5AD432745937F)) ^ _TAG_DIFF_PERT
                )
                diff_pert_seed = mix64_np(diff_pert_seed_input)
                diff_pert_pair = rand_perm_n_at(
                    diff_pert_block,
                    diff_pert_seed,
                    n=int(self.diff_per_rotpair),
                )
                diff_pert_src = diff_pert_pair // self.num_perts
                diff_pert_dst = diff_pert_pair % self.num_perts

                diff_trans_src = diff_rot_src * self.num_perts + diff_pert_src
                diff_trans_dst = diff_rot_dst * self.num_perts + diff_pert_dst
            else:
                diff_trans_src = np.zeros_like(shape_ids, dtype=np.uint64)
                diff_trans_dst = np.zeros_like(shape_ids, dtype=np.uint64)

            trans_src = np.where(use_same, same_trans_src, diff_trans_src)
            trans_dst = np.where(use_same, same_trans_dst, diff_trans_dst)

            pair_ids = scale_ids * (self.num_trans * self.num_trans) + trans_src * self.num_trans + trans_dst
            dataset_index = shape_ids * np.uint64(self.num_pairs) + pair_ids

            self.iter_count += 1
            yield dataset_index.astype(np.int64, copy=False).tolist()
