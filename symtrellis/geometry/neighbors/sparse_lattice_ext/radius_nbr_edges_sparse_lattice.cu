#include "radius_nbr_edges_sparse_lattice.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace
{
  // Constants and tensor holders.
  constexpr int kBucketSide = 4;
  constexpr int kBucketSlots = 64;
  constexpr int kResidueCount = 64;
  constexpr int kWarpSize = 32;
  constexpr int kMaxProbe = 128;
  constexpr int64_t kInt32Max = 2147483647LL;
  constexpr int64_t kMaxDenseOffsetGroups = 32768LL;

  enum FailureCode : int32_t
  {
    kFailNone = 0,
    kFailNegativeKeyShift = 1,
    kFailNegativeBatchId = 2,
    kFailBucketCoordOverflow = 3,
    kFailDuplicateSlot = 4,
    kFailHashProbeLimit = 5,
    kFailNegativeQueryBatchId = 6,
    kFailOffsetTableRange = 7,
  };

  struct OffsetGroupTable
  {
    torch::Tensor item_start;     // int64 [64, gmax]
    torch::Tensor item_count;     // int64 [64, gmax]
    torch::Tensor item_offset_id; // int64 [64 * L]
    torch::Tensor item_slot;      // int32 [64 * L]
    int64_t gmax = 0;
    int32_t delta_min = 0;
    int32_t delta_dim = 0;
  };
  struct OffsetGroupTableView
  {
    const int64_t *item_start;
    const int64_t *item_count;
    const int64_t *item_offset_id;
    const int32_t *item_slot;
    int64_t gmax;
    int32_t delta_min;
    int32_t delta_dim;
  };
  struct KeyFields
  {
    torch::Tensor bid;
    torch::Tensor bx;
    torch::Tensor by;
    torch::Tensor bz;
    torch::Tensor slot;
    torch::Tensor initial_perm;
    int64_t n = 0;
  };
  struct SortWorkspace
  {
    torch::Tensor perm_tmp;
    torch::Tensor sort_key_a;
    torch::Tensor sort_key_b;
  };
  struct BucketHashTable
  {
    torch::Tensor state;
    torch::Tensor bid;
    torch::Tensor bx;
    torch::Tensor by;
    torch::Tensor bz;
    torch::Tensor value;
    int64_t capacity = 0;
  };
  struct BucketHashTableView
  {
    int32_t *state;
    int32_t *bid;
    int32_t *bx;
    int32_t *by;
    int32_t *bz;
    int64_t *value;
    int64_t capacity;
  };
  struct BucketIndexBuffers
  {
    torch::Tensor bucket_start;
    torch::Tensor bucket_mask;
    torch::Tensor is_new_bucket;
    torch::Tensor bucket_prefix;
    BucketHashTable hash_table;
  };
  struct BucketIndexView
  {
    const int64_t *bucket_start;
    const int64_t *bucket_mask;
    const int64_t *compact_kids;
    BucketHashTableView hash_table;
  };
  struct OffsetGroupConfig
  {
    int64_t gmax = 0;
    int32_t delta_min = 0;
    int32_t delta_dim = 0;
  };
  // Central hard checks.
  int64_t checked_mul_i64(int64_t a, int64_t b, const char *name)
  {
    TORCH_CHECK(a >= 0 && b >= 0, name, " has negative size");
    TORCH_CHECK(a == 0 || b <= std::numeric_limits<int64_t>::max() / a,
                name, " overflow");
    return a * b;
  }
  int cub_count(int64_t n, const char *name)
  {
    TORCH_CHECK(n >= 0 && n <= std::numeric_limits<int>::max(),
                name, " exceeds CUB int count");
    return static_cast<int>(n);
  }
  int launch_blocks_1d(int64_t total, int threads, const char *name)
  {
    TORCH_CHECK(total >= 0 && threads > 0, name, " invalid launch size");
    int64_t blocks = (total + threads - 1) / threads;
    TORCH_CHECK(blocks <= std::numeric_limits<int>::max(),
                name, " launch grid too large");
    return static_cast<int>(blocks);
  }
  __host__ __device__ __forceinline__ int64_t floor_div4(int64_t x)
  {
    return x >= 0 ? x / kBucketSide : -((-x + kBucketSide - 1) / kBucketSide);
  }

  OffsetGroupConfig make_offset_group_config(double radius)
  {
    TORCH_CHECK(std::isfinite(radius) && radius > 0.0, "invalid radius");
    int64_t r = static_cast<int64_t>(std::ceil(radius));
    int64_t delta_min = floor_div4(-r);
    int64_t delta_max = floor_div4(r + 4);
    int64_t delta_dim = delta_max - delta_min + 1;
    TORCH_CHECK(delta_dim > 0 && delta_dim <= kInt32Max, "invalid delta grid");
    TORCH_CHECK(delta_dim <= std::numeric_limits<int64_t>::max() / delta_dim,
                "delta_dim^2 overflow");
    int64_t delta_dim2 = delta_dim * delta_dim;
    TORCH_CHECK(delta_dim <= std::numeric_limits<int64_t>::max() / delta_dim2,
                "delta_dim^3 overflow");
    int64_t gmax = delta_dim2 * delta_dim;
    TORCH_CHECK(gmax <= kMaxDenseOffsetGroups, "dense offset grid too large");
    cub_count(checked_mul_i64(kResidueCount, gmax, "64 * gmax"), "64 * gmax");
    return {gmax, static_cast<int32_t>(delta_min), static_cast<int32_t>(delta_dim)};
  }
  // Device math and bucket hash helpers.
  __device__ __forceinline__ void set_failure(int32_t *fail, FailureCode code)
  {
    atomicCAS(fail, kFailNone, static_cast<int32_t>(code));
  }
  __device__ __forceinline__ void atomic_add_i64(int64_t *ptr, int64_t value)
  {
    atomicAdd(reinterpret_cast<unsigned long long *>(ptr),
              static_cast<unsigned long long>(value));
  }
  __host__ __device__ __forceinline__ void offset_info(
      int64_t residue, const int32_t *nbr_offsets, int64_t oid,
      int32_t *dx, int32_t *dy, int32_t *dz, int32_t *slot)
  {
    int64_t rx = residue & 3;
    int64_t ry = (residue >> 2) & 3;
    int64_t rz = (residue >> 4) & 3;
    int64_t rel_x = rx + static_cast<int64_t>(nbr_offsets[oid * 3 + 0]);
    int64_t rel_y = ry + static_cast<int64_t>(nbr_offsets[oid * 3 + 1]);
    int64_t rel_z = rz + static_cast<int64_t>(nbr_offsets[oid * 3 + 2]);
    int64_t gx = floor_div4(rel_x);
    int64_t gy = floor_div4(rel_y);
    int64_t gz = floor_div4(rel_z);
    int64_t lx = rel_x - kBucketSide * gx;
    int64_t ly = rel_y - kBucketSide * gy;
    int64_t lz = rel_z - kBucketSide * gz;
    *dx = static_cast<int32_t>(gx);
    *dy = static_cast<int32_t>(gy);
    *dz = static_cast<int32_t>(gz);
    *slot = static_cast<int32_t>(lx + kBucketSide * ly + 16 * lz);
  }
  __device__ __forceinline__ int64_t dense_delta_gid_device(
      int32_t dx, int32_t dy, int32_t dz, int32_t delta_min,
      int32_t delta_dim, int64_t gmax, int32_t *fail)
  {
    int64_t ix = static_cast<int64_t>(dx) - delta_min;
    int64_t iy = static_cast<int64_t>(dy) - delta_min;
    int64_t iz = static_cast<int64_t>(dz) - delta_min;
    if (ix < 0 || iy < 0 || iz < 0 || ix >= delta_dim || iy >= delta_dim ||
        iz >= delta_dim)
    {
      set_failure(fail, kFailOffsetTableRange);
      return -1;
    }
    int64_t gid = ix + static_cast<int64_t>(delta_dim) *
                           (iy + static_cast<int64_t>(delta_dim) * iz);
    if (gid >= gmax)
    {
      set_failure(fail, kFailOffsetTableRange);
      return -1;
    }
    return gid;
  }
  __device__ __forceinline__ uint64_t mix64(uint64_t x)
  {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
  }
  __device__ __forceinline__ uint64_t hash_bucket_key(
      int32_t bid, int32_t bx, int32_t by, int32_t bz)
  {
    uint64_t h = 0x9e3779b97f4a7c15ULL;
    h ^= mix64(static_cast<uint32_t>(bid) + 0x100000001b3ULL);
    h = mix64(h ^ static_cast<uint32_t>(bx));
    h = mix64(h ^ static_cast<uint32_t>(by));
    return mix64(h ^ static_cast<uint32_t>(bz));
  }
  __device__ __forceinline__ bool same_bucket_key(
      int32_t bid, int32_t bx, int32_t by, int32_t bz,
      BucketHashTableView table, int64_t slot)
  {
    return bid == table.bid[slot] && bx == table.bx[slot] &&
           by == table.by[slot] && bz == table.bz[slot];
  }
  __device__ int64_t lookup_bucket(
      int32_t bid, int32_t bx, int32_t by, int32_t bz,
      BucketHashTableView table)
  {
    int64_t slot0 =
        static_cast<int64_t>(hash_bucket_key(bid, bx, by, bz) & (table.capacity - 1));
    for (int probe = 0; probe < kMaxProbe; ++probe)
    {
      int64_t slot = (slot0 + probe) & (table.capacity - 1);
      int32_t state = table.state[slot];
      if (state == 0)
      {
        return -1;
      }
      if (state == 2 && same_bucket_key(bid, bx, by, bz, table, slot))
      {
        return table.value[slot];
      }
    }
    return -1;
  }
  __device__ void insert_bucket(
      int32_t bid, int32_t bx, int32_t by, int32_t bz, int64_t record_id,
      BucketHashTableView table, int32_t *fail)
  {
    int64_t slot0 =
        static_cast<int64_t>(hash_bucket_key(bid, bx, by, bz) & (table.capacity - 1));
    for (int probe = 0; probe < kMaxProbe; ++probe)
    {
      int64_t slot = (slot0 + probe) & (table.capacity - 1);
      int32_t old_state = atomicCAS(&table.state[slot], 0, 1);
      if (old_state == 0)
      {
        table.bid[slot] = bid;
        table.bx[slot] = bx;
        table.by[slot] = by;
        table.bz[slot] = bz;
        table.value[slot] = record_id;
        __threadfence();
        atomicExch(&table.state[slot], 2);
        return;
      }
      if (old_state == 2 && same_bucket_key(bid, bx, by, bz, table, slot))
      {
        table.value[slot] = record_id;
        return;
      }
    }
    set_failure(fail, kFailHashProbeLimit);
  }
  int64_t next_power_of_two(int64_t x)
  {
    TORCH_CHECK(x > 0 && x <= (int64_t{1} << 62), "hash capacity overflow");
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return x + 1;
  }
  OffsetGroupTableView view(const OffsetGroupTable &table)
  {
    return {table.item_start.data_ptr<int64_t>(), table.item_count.data_ptr<int64_t>(),
            table.item_offset_id.data_ptr<int64_t>(), table.item_slot.data_ptr<int32_t>(),
            table.gmax, table.delta_min, table.delta_dim};
  }
  BucketHashTableView view(BucketHashTable &table)
  {
    return {table.state.data_ptr<int32_t>(), table.bid.data_ptr<int32_t>(),
            table.bx.data_ptr<int32_t>(), table.by.data_ptr<int32_t>(),
            table.bz.data_ptr<int32_t>(), table.value.data_ptr<int64_t>(),
            table.capacity};
  }
  BucketIndexView view(BucketIndexBuffers &buffers, const torch::Tensor &compact_kids)
  {
    return {buffers.bucket_start.data_ptr<int64_t>(), buffers.bucket_mask.data_ptr<int64_t>(),
            compact_kids.data_ptr<int64_t>(), view(buffers.hash_table)};
  }
  std::string failure_message(int32_t code)
  {
    switch (code)
    {
    case kFailNegativeKeyShift:
      return "key_coords - coord_min must be non-negative";
    case kFailNegativeBatchId:
      return "key_bid must be non-negative";
    case kFailBucketCoordOverflow:
      return "bucket coordinate overflow";
    case kFailDuplicateSlot:
      return "duplicate key coordinate within batch";
    case kFailHashProbeLimit:
      return "hash table probe limit exceeded";
    case kFailNegativeQueryBatchId:
      return "query_bid must be non-negative";
    case kFailOffsetTableRange:
      return "nbr_offsets outside analytic delta grid";
    default:
      return "unknown sparse lattice CUDA failure";
    }
  }
  void check_fail_flag_final(const torch::Tensor &fail, cudaStream_t stream)
  {
    int32_t host = 0;
    C10_CUDA_CHECK(cudaMemcpyAsync(&host, fail.data_ptr<int32_t>(), sizeof(int32_t),
                                   cudaMemcpyDeviceToHost, stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    TORCH_CHECK(host == kFailNone, failure_message(host));
  }

  void validate_inputs_minimal(
      const torch::Tensor &query_pos, const torch::Tensor &query_bid,
      const torch::Tensor &key_coords, const torch::Tensor &key_bid,
      double radius, const torch::Tensor &nbr_offsets,
      const torch::Tensor &coord_min)
  {
    bool same_device = query_bid.device() == query_pos.device() &&
                       key_coords.device() == query_pos.device() &&
                       key_bid.device() == query_pos.device() &&
                       nbr_offsets.device() == query_pos.device() &&
                       coord_min.device() == query_pos.device();
    bool shapes_ok = query_pos.dim() == 2 && query_pos.size(1) == 3 &&
                     query_bid.dim() == 1 && query_bid.size(0) == query_pos.size(0) &&
                     key_coords.dim() == 2 && key_coords.size(1) == 3 &&
                     key_bid.dim() == 1 && key_bid.size(0) == key_coords.size(0) &&
                     nbr_offsets.dim() == 2 && nbr_offsets.size(1) == 3 &&
                     coord_min.dim() == 1 && coord_min.size(0) == 3;
    bool dtypes_ok = (query_pos.scalar_type() == torch::kFloat32 ||
                      query_pos.scalar_type() == torch::kFloat64) &&
                     query_bid.scalar_type() == torch::kInt32 &&
                     key_coords.scalar_type() == torch::kInt32 &&
                     key_bid.scalar_type() == torch::kInt32 &&
                     nbr_offsets.scalar_type() == torch::kInt32 &&
                     coord_min.scalar_type() == torch::kInt32;

    TORCH_CHECK(query_pos.is_cuda() && query_bid.is_cuda() && key_coords.is_cuda() &&
                    key_bid.is_cuda() && nbr_offsets.is_cuda() && coord_min.is_cuda(),
                "expected CUDA tensors");
    TORCH_CHECK(query_pos.is_contiguous() && query_bid.is_contiguous() &&
                    key_coords.is_contiguous() && key_bid.is_contiguous() &&
                    nbr_offsets.is_contiguous() && coord_min.is_contiguous(),
                "expected contiguous tensors");
    TORCH_CHECK(same_device, "expected one CUDA device");
    TORCH_CHECK(shapes_ok, "invalid tensor shape");
    TORCH_CHECK(dtypes_ok, "invalid tensor dtype");
    TORCH_CHECK(key_coords.size(0) <= std::numeric_limits<int>::max(),
                "Nk exceeds CUB int limit");
    checked_mul_i64(query_pos.size(0), nbr_offsets.size(0), "Nq * L");
    checked_mul_i64(2, key_coords.size(0), "2 * Nk");
    make_offset_group_config(radius);
  }

  // GPU-side dense delta-grid offset grouping.

  void cub_exclusive_sum_int64(
      const torch::Tensor &input, const torch::Tensor &output,
      int64_t n, cudaStream_t stream);

  // Count offsets per residue and dense delta cell.
  // The dense cell bounds come only from radius, so invalid offsets hard fail.
  __global__ void count_offset_items_kernel(
      const int32_t *nbr_offsets, int64_t L, int64_t total_items,
      int64_t gmax, int32_t delta_min, int32_t delta_dim,
      int64_t *item_count, int32_t *fail)
  {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total_items)
    {
      return;
    }
    int64_t residue = i / L;
    int64_t oid = i - residue * L;
    int32_t dx, dy, dz, slot;
    offset_info(residue, nbr_offsets, oid, &dx, &dy, &dz, &slot);
    int64_t gid = dense_delta_gid_device(dx, dy, dz, delta_min, delta_dim, gmax, fail);
    if (gid >= 0)
    {
      atomic_add_i64(&item_count[residue * gmax + gid], 1);
    }
  }

  // Fill compact offset items after exclusive scan.
  // item_cursor is temporary and assigns the next row within each dense cell.
  __global__ void fill_offset_items_kernel(
      const int32_t *nbr_offsets, int64_t L, int64_t total_items,
      int64_t gmax, int32_t delta_min, int32_t delta_dim,
      const int64_t *item_start, int64_t *item_cursor,
      int64_t *item_offset_id, int32_t *item_slot,
      int32_t *fail)
  {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total_items)
    {
      return;
    }
    int64_t residue = i / L;
    int64_t oid = i - residue * L;
    int32_t dx, dy, dz, slot;
    offset_info(residue, nbr_offsets, oid, &dx, &dy, &dz, &slot);
    int64_t gid = dense_delta_gid_device(dx, dy, dz, delta_min, delta_dim, gmax, fail);
    if (gid < 0)
    {
      return;
    }
    int64_t flat = residue * gmax + gid;
    auto old = atomicAdd(reinterpret_cast<unsigned long long *>(&item_cursor[flat]), 1ULL);
    int64_t item = item_start[flat] + static_cast<int64_t>(old);
    item_offset_id[item] = oid;
    item_slot[item] = slot;
  }

  OffsetGroupTable build_offset_group_table_cuda(
      const torch::Tensor &nbr_offsets, double radius, int64_t L,
      const torch::TensorOptions &int32_options,
      const torch::TensorOptions &int64_options,
      const torch::Tensor &fail, cudaStream_t stream)
  {
    OffsetGroupConfig config = make_offset_group_config(radius);
    int64_t total_items = checked_mul_i64(kResidueCount, L, "64 * L");
    int64_t total_cells = checked_mul_i64(kResidueCount, config.gmax, "64 * gmax");

    OffsetGroupTable table;
    table.gmax = config.gmax;
    table.delta_min = config.delta_min;
    table.delta_dim = config.delta_dim;
    table.item_start = at::empty({kResidueCount, table.gmax}, int64_options);
    table.item_count = at::zeros({kResidueCount, table.gmax}, int64_options);
    table.item_offset_id = at::empty({total_items}, int64_options);
    table.item_slot = at::empty({total_items}, int32_options);
    auto item_cursor = at::zeros({kResidueCount, table.gmax}, int64_options);

    int threads = 256;
    int blocks = launch_blocks_1d(total_items, threads, "offset item kernels");
    count_offset_items_kernel<<<blocks, threads, 0, stream>>>(
        nbr_offsets.data_ptr<int32_t>(), L, total_items, table.gmax,
        table.delta_min, table.delta_dim, table.item_count.data_ptr<int64_t>(),
        fail.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    cub_exclusive_sum_int64(table.item_count, table.item_start, total_cells, stream);

    fill_offset_items_kernel<<<blocks, threads, 0, stream>>>(
        nbr_offsets.data_ptr<int32_t>(), L, total_items, table.gmax,
        table.delta_min, table.delta_dim, table.item_start.data_ptr<int64_t>(),
        item_cursor.data_ptr<int64_t>(), table.item_offset_id.data_ptr<int64_t>(),
        table.item_slot.data_ptr<int32_t>(), fail.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return table;
  }

  // Key field construction and stable sorting.

  // Converts integer key coordinates into bucket tuple and local slot.
  // key_coords - coord_min must be non-negative for every key.
  __global__ void build_key_fields_kernel(
      const int32_t *key_coords, const int32_t *key_bid,
      const int32_t *coord_min, int64_t Nk, int32_t *bid,
      int32_t *bx, int32_t *by, int32_t *bz, int32_t *slot,
      int64_t *perm, int32_t *fail)
  {
    int64_t kid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (kid >= Nk)
    {
      return;
    }
    int32_t b = key_bid[kid];
    if (b < 0)
    {
      set_failure(fail, kFailNegativeBatchId);
    }

    int64_t sx = static_cast<int64_t>(key_coords[kid * 3 + 0]) - coord_min[0];
    int64_t sy = static_cast<int64_t>(key_coords[kid * 3 + 1]) - coord_min[1];
    int64_t sz = static_cast<int64_t>(key_coords[kid * 3 + 2]) - coord_min[2];
    if (sx < 0 || sy < 0 || sz < 0)
    {
      set_failure(fail, kFailNegativeKeyShift);
      sx = sx < 0 ? 0 : sx;
      sy = sy < 0 ? 0 : sy;
      sz = sz < 0 ? 0 : sz;
    }

    int64_t cbx = sx / kBucketSide;
    int64_t cby = sy / kBucketSide;
    int64_t cbz = sz / kBucketSide;
    if (cbx > kInt32Max || cby > kInt32Max || cbz > kInt32Max)
    {
      set_failure(fail, kFailBucketCoordOverflow);
      cbx = cbx > kInt32Max ? kInt32Max : cbx;
      cby = cby > kInt32Max ? kInt32Max : cby;
      cbz = cbz > kInt32Max ? kInt32Max : cbz;
    }

    int64_t lx = sx - kBucketSide * cbx;
    int64_t ly = sy - kBucketSide * cby;
    int64_t lz = sz - kBucketSide * cbz;
    bid[kid] = b;
    bx[kid] = static_cast<int32_t>(cbx);
    by[kid] = static_cast<int32_t>(cby);
    bz[kid] = static_cast<int32_t>(cbz);
    slot[kid] = static_cast<int32_t>(lx + kBucketSide * ly + 16 * lz);
    perm[kid] = kid;
  }

  KeyFields build_key_fields(
      const torch::Tensor &key_coords, const torch::Tensor &key_bid,
      const torch::Tensor &coord_min, int64_t Nk, const torch::Tensor &fail,
      const torch::TensorOptions &int32_options,
      const torch::TensorOptions &int64_options, cudaStream_t stream)
  {
    KeyFields fields;
    fields.n = Nk;
    fields.bid = at::empty({Nk}, int32_options);
    fields.bx = at::empty({Nk}, int32_options);
    fields.by = at::empty({Nk}, int32_options);
    fields.bz = at::empty({Nk}, int32_options);
    fields.slot = at::empty({Nk}, int32_options);
    fields.initial_perm = at::empty({Nk}, int64_options);

    int blocks = launch_blocks_1d(Nk, 256, "build key fields");
    build_key_fields_kernel<<<blocks, 256, 0, stream>>>(
        key_coords.data_ptr<int32_t>(), key_bid.data_ptr<int32_t>(),
        coord_min.data_ptr<int32_t>(), Nk, fields.bid.data_ptr<int32_t>(),
        fields.bx.data_ptr<int32_t>(), fields.by.data_ptr<int32_t>(),
        fields.bz.data_ptr<int32_t>(), fields.slot.data_ptr<int32_t>(),
        fields.initial_perm.data_ptr<int64_t>(), fail.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return fields;
  }

  SortWorkspace make_sort_workspace(
      int64_t n, const torch::TensorOptions &int32_options,
      const torch::TensorOptions &int64_options)
  {
    return {at::empty({n}, int64_options),
            at::empty({n}, int32_options),
            at::empty({n}, int32_options)};
  }

  void cub_sort_pairs_int32_int64(
      const torch::Tensor &keys_in, const torch::Tensor &keys_out,
      const torch::Tensor &values_in, const torch::Tensor &values_out,
      int64_t n, cudaStream_t stream)
  {
    int count = cub_count(n, "sort pairs");
    size_t bytes = 0;
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, bytes, keys_in.data_ptr<int32_t>(), keys_out.data_ptr<int32_t>(),
        values_in.data_ptr<int64_t>(), values_out.data_ptr<int64_t>(), count, 0,
        8 * static_cast<int>(sizeof(int32_t)), stream));
    auto tmp = at::empty({static_cast<int64_t>(bytes)},
                         keys_in.options().dtype(torch::kUInt8));
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        tmp.data_ptr<uint8_t>(), bytes, keys_in.data_ptr<int32_t>(),
        keys_out.data_ptr<int32_t>(), values_in.data_ptr<int64_t>(),
        values_out.data_ptr<int64_t>(), count, 0,
        8 * static_cast<int>(sizeof(int32_t)), stream));
  }

  void cub_inclusive_sum_int32(
      const torch::Tensor &input, const torch::Tensor &output,
      int64_t n, cudaStream_t stream)
  {
    int count = cub_count(n, "inclusive sum");
    size_t bytes = 0;
    C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, bytes, input.data_ptr<int32_t>(), output.data_ptr<int32_t>(),
        count, stream));
    auto tmp = at::empty({static_cast<int64_t>(bytes)},
                         input.options().dtype(torch::kUInt8));
    C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        tmp.data_ptr<uint8_t>(), bytes, input.data_ptr<int32_t>(),
        output.data_ptr<int32_t>(), count, stream));
  }

  void cub_exclusive_sum_int64(
      const torch::Tensor &input, const torch::Tensor &output,
      int64_t n, cudaStream_t stream)
  {
    int count = cub_count(n, "exclusive sum");
    size_t bytes = 0;
    C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, bytes, input.data_ptr<int64_t>(), output.data_ptr<int64_t>(),
        count, stream));
    auto tmp = at::empty({static_cast<int64_t>(bytes)},
                         input.options().dtype(torch::kUInt8));
    C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        tmp.data_ptr<uint8_t>(), bytes, input.data_ptr<int64_t>(),
        output.data_ptr<int64_t>(), count, stream));
  }

  __global__ void gather_sort_key_kernel(
      const int32_t *field, const int64_t *perm, int64_t n, int32_t *sort_key)
  {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
    {
      sort_key[i] = field[perm[i]];
    }
  }

  void sort_next_field(
      const torch::Tensor &field, torch::Tensor &current, torch::Tensor &other,
      SortWorkspace &workspace, int64_t n, cudaStream_t stream)
  {
    int blocks = launch_blocks_1d(n, 256, "gather sort key");
    gather_sort_key_kernel<<<blocks, 256, 0, stream>>>(
        field.data_ptr<int32_t>(), current.data_ptr<int64_t>(), n,
        workspace.sort_key_a.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    cub_sort_pairs_int32_int64(
        workspace.sort_key_a, workspace.sort_key_b, current, other, n, stream);
    std::swap(current, other);
  }

  torch::Tensor stable_lexsort_bucket_slot(
      const KeyFields &fields, SortWorkspace &workspace, cudaStream_t stream)
  {
    torch::Tensor current = fields.initial_perm;
    torch::Tensor other = workspace.perm_tmp;
    cub_sort_pairs_int32_int64(
        fields.slot, workspace.sort_key_b, current, other, fields.n, stream);
    std::swap(current, other);
    sort_next_field(fields.bz, current, other, workspace, fields.n, stream);
    sort_next_field(fields.by, current, other, workspace, fields.n, stream);
    sort_next_field(fields.bx, current, other, workspace, fields.n, stream);
    sort_next_field(fields.bid, current, other, workspace, fields.n, stream);
    return current;
  }

  // Bucket record construction and hash table publishing.

  __global__ void mark_bucket_boundaries_kernel(
      const int32_t *bid, const int32_t *bx, const int32_t *by,
      const int32_t *bz, const int64_t *perm, int64_t Nk,
      int32_t *is_new_bucket)
  {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= Nk)
    {
      return;
    }
    if (i == 0)
    {
      is_new_bucket[i] = 1;
      return;
    }
    int64_t kid = perm[i];
    int64_t prev = perm[i - 1];
    is_new_bucket[i] =
        bid[kid] != bid[prev] || bx[kid] != bx[prev] || by[kid] != by[prev] ||
                bz[kid] != bz[prev]
            ? 1
            : 0;
  }

  // One thread owns one sorted bucket.
  // It builds a uint64 slot mask and publishes the hash record after tuple writes.
  __global__ void build_bucket_records_kernel(
      const int32_t *bid, const int32_t *bx, const int32_t *by,
      const int32_t *bz, const int32_t *slot, const int64_t *compact_kids,
      const int32_t *is_new_bucket, const int32_t *bucket_prefix,
      int64_t Nk, int64_t *bucket_start, int64_t *bucket_mask,
      BucketHashTableView hash_table, int32_t *fail)
  {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= Nk || is_new_bucket[i] == 0)
    {
      return;
    }

    int64_t record_id = static_cast<int64_t>(bucket_prefix[i] - 1);
    int64_t first = compact_kids[i];
    uint64_t mask = 0;
    int count = 0;
    for (int64_t p = i; p < Nk && (p == i || is_new_bucket[p] == 0); ++p)
    {
      int64_t kid = compact_kids[p];
      uint64_t bit = 1ULL << slot[kid];
      if ((mask & bit) != 0)
      {
        set_failure(fail, kFailDuplicateSlot);
      }
      mask |= bit;
      ++count;
    }
    if (count > kBucketSlots)
    {
      set_failure(fail, kFailDuplicateSlot);
    }

    bucket_start[record_id] = i;
    bucket_mask[record_id] = static_cast<int64_t>(mask);
    insert_bucket(
        bid[first], bx[first], by[first], bz[first], record_id, hash_table, fail);
  }

  BucketHashTable make_bucket_hash_table(
      int64_t Nk, const torch::TensorOptions &int32_options,
      const torch::TensorOptions &int64_options)
  {
    BucketHashTable table;
    table.capacity = next_power_of_two(checked_mul_i64(2, Nk, "2 * Nk"));
    table.state = at::zeros({table.capacity}, int32_options);
    table.bid = at::empty({table.capacity}, int32_options);
    table.bx = at::empty({table.capacity}, int32_options);
    table.by = at::empty({table.capacity}, int32_options);
    table.bz = at::empty({table.capacity}, int32_options);
    table.value = at::empty({table.capacity}, int64_options);
    return table;
  }

  BucketIndexBuffers build_bucket_index_and_hash(
      const KeyFields &fields, const torch::Tensor &compact_kids,
      const torch::Tensor &fail,
      const torch::TensorOptions &int32_options,
      const torch::TensorOptions &int64_options, cudaStream_t stream)
  {
    BucketIndexBuffers buffers;
    buffers.is_new_bucket = at::empty({fields.n}, int32_options);
    buffers.bucket_prefix = at::empty({fields.n}, int32_options);
    buffers.bucket_start = at::empty({fields.n}, int64_options);
    buffers.bucket_mask = at::empty({fields.n}, int64_options);
    buffers.hash_table = make_bucket_hash_table(fields.n, int32_options, int64_options);

    int blocks = launch_blocks_1d(fields.n, 256, "bucket index kernels");
    mark_bucket_boundaries_kernel<<<blocks, 256, 0, stream>>>(
        fields.bid.data_ptr<int32_t>(), fields.bx.data_ptr<int32_t>(),
        fields.by.data_ptr<int32_t>(), fields.bz.data_ptr<int32_t>(),
        compact_kids.data_ptr<int64_t>(), fields.n,
        buffers.is_new_bucket.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    cub_inclusive_sum_int32(
        buffers.is_new_bucket, buffers.bucket_prefix, fields.n, stream);

    auto hash_view = view(buffers.hash_table);
    build_bucket_records_kernel<<<blocks, 256, 0, stream>>>(
        fields.bid.data_ptr<int32_t>(), fields.bx.data_ptr<int32_t>(),
        fields.by.data_ptr<int32_t>(), fields.bz.data_ptr<int32_t>(),
        fields.slot.data_ptr<int32_t>(), compact_kids.data_ptr<int64_t>(),
        buffers.is_new_bucket.data_ptr<int32_t>(),
        buffers.bucket_prefix.data_ptr<int32_t>(), fields.n,
        buffers.bucket_start.data_ptr<int64_t>(),
        buffers.bucket_mask.data_ptr<int64_t>(), hash_view,
        fail.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return buffers;
  }

  // Query kernel.

  // One warp owns one query.
  // It walks non-empty dense delta cells, checks the bucket bit, then exact radius.
  template <typename scalar_t>
  __global__ void query_kids_by_offset_kernel(
      const scalar_t *query_pos, const int32_t *query_bid,
      const int32_t *coord_min, const int32_t *nbr_offsets,
      int64_t Nq, int64_t L, double radius2, OffsetGroupTableView offsets,
      BucketIndexView buckets, int64_t *kids_by_offset, int32_t *fail)
  {
    int64_t global_thread =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t qid = global_thread / kWarpSize;
    int lane = threadIdx.x & (kWarpSize - 1);
    if (qid >= Nq)
    {
      return;
    }

    // initialize kids_by_offset row to -1
    int64_t row = qid * L;
    for (int64_t oid = lane; oid < L; oid += kWarpSize)
    {
      kids_by_offset[row + oid] = -1;
    }
    __syncwarp();

    int32_t qbid = query_bid[qid];
    if (qbid < 0)
    {
      set_failure(fail, kFailNegativeQueryBatchId);
      return;
    }

    // compute query base cell, residual, bucket, residue
    double qx = static_cast<double>(query_pos[qid * 3 + 0]);
    double qy = static_cast<double>(query_pos[qid * 3 + 1]);
    double qz = static_cast<double>(query_pos[qid * 3 + 2]);
    int64_t base_x = static_cast<int64_t>(floor(qx));
    int64_t base_y = static_cast<int64_t>(floor(qy));
    int64_t base_z = static_cast<int64_t>(floor(qz));
    double rx = qx - static_cast<double>(base_x);
    double ry = qy - static_cast<double>(base_y);
    double rz = qz - static_cast<double>(base_z);
    int64_t sx = base_x - coord_min[0];
    int64_t sy = base_y - coord_min[1];
    int64_t sz = base_z - coord_min[2];
    int64_t qbx = floor_div4(sx);
    int64_t qby = floor_div4(sy);
    int64_t qbz = floor_div4(sz);
    int64_t res_x = sx - kBucketSide * qbx;
    int64_t res_y = sy - kBucketSide * qby;
    int64_t res_z = sz - kBucketSide * qbz;
    int64_t residue = res_x + kBucketSide * res_y + 16 * res_z;

    // iterate non-empty dense delta groups
    for (int64_t g = 0; g < offsets.gmax; ++g)
    {
      int64_t flat = residue * offsets.gmax + g;
      int64_t count = offsets.item_count[flat];
      if (count == 0)
      {
        continue;
      }

      int64_t ix = g % offsets.delta_dim;
      int64_t iy = (g / offsets.delta_dim) % offsets.delta_dim;
      int64_t iz = g / (static_cast<int64_t>(offsets.delta_dim) * offsets.delta_dim);
      int64_t abs_bx = qbx + offsets.delta_min + ix;
      int64_t abs_by = qby + offsets.delta_min + iy;
      int64_t abs_bz = qbz + offsets.delta_min + iz;
      if (abs_bx < 0 || abs_by < 0 || abs_bz < 0 || abs_bx > kInt32Max ||
          abs_by > kInt32Max || abs_bz > kInt32Max)
      {
        continue;
      }

      // lookup target coarse bucket
      int64_t record = -1;
      if (lane == 0)
      {
        record = lookup_bucket(
            qbid, static_cast<int32_t>(abs_bx), static_cast<int32_t>(abs_by),
            static_cast<int32_t>(abs_bz), buckets.hash_table);
      }
      long long shfl_record = static_cast<long long>(record);
      shfl_record = __shfl_sync(0xffffffff, shfl_record, 0);
      record = static_cast<int64_t>(shfl_record);
      if (record < 0)
      {
        continue;
      }

      // test local slot and exact radius
      uint64_t mask = static_cast<uint64_t>(buckets.bucket_mask[record]);
      int64_t start = buckets.bucket_start[record];
      int64_t begin = offsets.item_start[flat];
      for (int64_t j = lane; j < count; j += kWarpSize)
      {
        int64_t item = begin + j;
        int64_t oid = offsets.item_offset_id[item];
        int32_t s = offsets.item_slot[item];
        uint64_t bit = 1ULL << s;
        if ((mask & bit) == 0)
        {
          continue;
        }
        int rank = __popcll(mask & (bit - 1));
        int64_t kid = buckets.compact_kids[start + rank];
        double dx = static_cast<double>(nbr_offsets[oid * 3 + 0]) - rx;
        double dy = static_cast<double>(nbr_offsets[oid * 3 + 1]) - ry;
        double dz = static_cast<double>(nbr_offsets[oid * 3 + 2]) - rz;
        if (dx * dx + dy * dy + dz * dz <= radius2)
        {
          kids_by_offset[row + oid] = kid;
        }
      }
    }
  }

  torch::Tensor query_kids_by_offset(
      const torch::Tensor &query_pos, const torch::Tensor &query_bid,
      const torch::Tensor &coord_min, const torch::Tensor &nbr_offsets,
      int64_t Nq, int64_t L, double radius, const OffsetGroupTable &offset_table,
      BucketIndexBuffers &bucket_index,
      const torch::Tensor &compact_kids,
      const torch::Tensor &fail,
      const torch::TensorOptions &int64_options, cudaStream_t stream)
  {
    auto kids_by_offset = at::empty({Nq, L}, int64_options);
    int64_t query_threads = checked_mul_i64(Nq, kWarpSize, "query threads");
    int blocks = launch_blocks_1d(query_threads, 128, "query kernel");
    auto offset_view = view(offset_table);
    auto bucket_view = view(bucket_index, compact_kids);
    double radius2 = radius * radius;

    AT_DISPATCH_FLOATING_TYPES(query_pos.scalar_type(), "query_kids_by_offset_kernel", [&]
                               { query_kids_by_offset_kernel<scalar_t><<<blocks, 128, 0, stream>>>(
                                     query_pos.data_ptr<scalar_t>(), query_bid.data_ptr<int32_t>(),
                                     coord_min.data_ptr<int32_t>(), nbr_offsets.data_ptr<int32_t>(),
                                     Nq, L, radius2, offset_view, bucket_view,
                                     kids_by_offset.data_ptr<int64_t>(), fail.data_ptr<int32_t>()); });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return kids_by_offset;
  }

} // namespace

// Public dispatch pipeline.

torch::Tensor radius_nbr_kids_by_offset_sparse_lattice_cuda(
    torch::Tensor query_pos,
    torch::Tensor query_bid,
    torch::Tensor key_coords,
    torch::Tensor key_bid,
    double radius,
    torch::Tensor nbr_offsets,
    torch::Tensor coord_min)
{
  validate_inputs_minimal(
      query_pos, query_bid, key_coords, key_bid, radius, nbr_offsets, coord_min);

  c10::cuda::CUDAGuard device_guard(query_pos.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  int64_t Nq = query_pos.size(0);
  int64_t Nk = key_coords.size(0);
  int64_t L = nbr_offsets.size(0);
  auto int32_options = key_coords.options().dtype(torch::kInt32);
  auto int64_options = key_coords.options().dtype(torch::kInt64);

  auto empty_kids = at::empty({Nq, L}, int64_options);
  if (Nq == 0 || L == 0)
  {
    return empty_kids;
  }
  if (Nk == 0)
  {
    empty_kids.fill_(-1);
    return empty_kids;
  }

  auto fail = at::zeros({1}, int32_options);

  // Pipeline:
  // 1. Build residue-dependent dense delta-grid offset groups on GPU.
  // 2. Convert key coordinates into coarse bucket fields and local slots.
  // 3. Stable-sort keys by (bid, bx, by, bz, slot).
  // 4. Build compact bucket records and publish the bucket hash table.
  // 5. Query one warp per query and write kids_by_offset[qid, offset_id].
  auto offset_table = build_offset_group_table_cuda(
      nbr_offsets, radius, L, int32_options, int64_options, fail, stream);
  auto key_fields = build_key_fields(
      key_coords, key_bid, coord_min, Nk, fail, int32_options, int64_options, stream);
  auto sort_workspace = make_sort_workspace(Nk, int32_options, int64_options);
  auto compact_kids = stable_lexsort_bucket_slot(key_fields, sort_workspace, stream);
  auto bucket_index = build_bucket_index_and_hash(
      key_fields, compact_kids, fail, int32_options, int64_options, stream);
  auto kids_by_offset = query_kids_by_offset(
      query_pos, query_bid, coord_min, nbr_offsets, Nq, L, radius, offset_table,
      bucket_index, compact_kids, fail, int64_options, stream);

  check_fail_flag_final(fail, stream);
  return kids_by_offset;
}
