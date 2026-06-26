#include "radius_nbr_edges_sparse_lattice.h"

#include <ATen/ATen.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace
{

  // Constants and CPU-side holders.

  constexpr int kBucketSide = 4;
  constexpr int kBucketSlots = 64;
  constexpr int kResidueCount = 64;
  constexpr int64_t kInt32Max = 2147483647LL;
  constexpr int64_t kMaxDenseOffsetGroups = 32768LL;

  struct OffsetGroupConfig
  {
    int64_t gmax = 0;
    int32_t delta_min = 0;
    int32_t delta_dim = 0;
  };

  struct OffsetGroupTable
  {
    std::vector<int64_t> item_start;     // [64 * gmax]
    std::vector<int64_t> item_count;     // [64 * gmax]
    std::vector<int64_t> item_offset_id; // [64 * L]
    std::vector<int32_t> item_slot;      // [64 * L]
    int64_t gmax = 0;
    int32_t delta_min = 0;
    int32_t delta_dim = 0;
  };

  struct KeyRecord
  {
    int32_t bid = 0;
    int32_t bx = 0;
    int32_t by = 0;
    int32_t bz = 0;
    int32_t slot = 0;
    int64_t kid = 0;
  };

  struct BucketKey
  {
    int32_t bid = 0;
    int32_t bx = 0;
    int32_t by = 0;
    int32_t bz = 0;
  };

  struct BucketKeyEq
  {
    bool operator()(const BucketKey &a, const BucketKey &b) const
    {
      return a.bid == b.bid && a.bx == b.bx && a.by == b.by && a.bz == b.bz;
    }
  };

  uint64_t mix64(uint64_t x)
  {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
  }

  struct BucketKeyHash
  {
    size_t operator()(const BucketKey &k) const
    {
      uint64_t h = 0x9e3779b97f4a7c15ULL;
      h ^= mix64(static_cast<uint32_t>(k.bid) + 0x100000001b3ULL);
      h = mix64(h ^ static_cast<uint32_t>(k.bx));
      h = mix64(h ^ static_cast<uint32_t>(k.by));
      return static_cast<size_t>(mix64(h ^ static_cast<uint32_t>(k.bz)));
    }
  };

  struct BucketIndex
  {
    std::vector<int64_t> compact_kids;
    std::vector<int64_t> bucket_start;
    std::vector<uint64_t> bucket_mask;
    std::unordered_map<BucketKey, int64_t, BucketKeyHash, BucketKeyEq> bucket_map;
  };

  // Central hard checks.

  int64_t checked_mul_i64(int64_t a, int64_t b, const char *name)
  {
    TORCH_CHECK(a >= 0 && b >= 0, name, " has negative size");
    TORCH_CHECK(a == 0 || b <= std::numeric_limits<int64_t>::max() / a,
                name, " overflow");
    return a * b;
  }

  int64_t floor_div4(int64_t x)
  {
    return x >= 0 ? x / kBucketSide : -((-x + kBucketSide - 1) / kBucketSide);
  }

  // Radius gives the only accepted dense delta-grid size.
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
    checked_mul_i64(kResidueCount, gmax, "64 * gmax");
    return {gmax, static_cast<int32_t>(delta_min), static_cast<int32_t>(delta_dim)};
  }

  // Scalar math and bucket-key hash helpers.

  void offset_info(
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

  int64_t dense_delta_gid(
      int32_t dx, int32_t dy, int32_t dz,
      int32_t delta_min, int32_t delta_dim, int64_t gmax)
  {
    int64_t ix = static_cast<int64_t>(dx) - delta_min;
    int64_t iy = static_cast<int64_t>(dy) - delta_min;
    int64_t iz = static_cast<int64_t>(dz) - delta_min;
    TORCH_CHECK(ix >= 0 && iy >= 0 && iz >= 0 &&
                    ix < delta_dim && iy < delta_dim && iz < delta_dim,
                "nbr_offsets outside analytic delta grid");
    int64_t gid = ix + static_cast<int64_t>(delta_dim) *
                           (iy + static_cast<int64_t>(delta_dim) * iz);
    TORCH_CHECK(gid >= 0 && gid < gmax, "nbr_offsets outside analytic delta grid");
    return gid;
  }

  // CPU backend input validation.

  void validate_inputs_cpu(
      const torch::Tensor &query_pos, const torch::Tensor &query_bid,
      const torch::Tensor &key_coords, const torch::Tensor &key_bid,
      double radius, const torch::Tensor &nbr_offsets,
      const torch::Tensor &coord_min)
  {
    bool shapes_ok = query_pos.dim() == 2 && query_pos.size(1) == 3 &&
                     query_bid.dim() == 1 && query_bid.size(0) == query_pos.size(0) &&
                     key_coords.dim() == 2 && key_coords.size(1) == 3 &&
                     key_bid.dim() == 1 && key_bid.size(0) == key_coords.size(0) &&
                     nbr_offsets.dim() == 2 && nbr_offsets.size(1) == 3 &&
                     coord_min.dim() == 1 && coord_min.size(0) == 3;
    bool dtypes_ok = (query_pos.scalar_type() == at::kFloat ||
                      query_pos.scalar_type() == at::kDouble) &&
                     query_bid.scalar_type() == at::kInt && key_coords.scalar_type() == at::kInt &&
                     key_bid.scalar_type() == at::kInt && nbr_offsets.scalar_type() == at::kInt &&
                     coord_min.scalar_type() == at::kInt;

    TORCH_CHECK(query_pos.is_cpu() && query_bid.is_cpu() && key_coords.is_cpu() &&
                    key_bid.is_cpu() && nbr_offsets.is_cpu() && coord_min.is_cpu(),
                "expected CPU tensors");
    TORCH_CHECK(query_pos.is_contiguous() && query_bid.is_contiguous() &&
                    key_coords.is_contiguous() && key_bid.is_contiguous() &&
                    nbr_offsets.is_contiguous() && coord_min.is_contiguous(),
                "expected contiguous tensors");
    TORCH_CHECK(shapes_ok, "invalid tensor shape");
    TORCH_CHECK(dtypes_ok, "invalid tensor dtype");
    checked_mul_i64(query_pos.size(0), nbr_offsets.size(0), "Nq * L");
    make_offset_group_config(radius);
  }

  // CPU-side dense delta-grid offset grouping.

  // Build residue-dependent dense delta-grid offset groups.
  // This mirrors count -> prefix sum -> fill, but uses plain loops.
  OffsetGroupTable build_offset_group_table_cpu(
      const int32_t *nbr_offsets, int64_t L, double radius)
  {
    OffsetGroupConfig config = make_offset_group_config(radius);
    int64_t total_items = checked_mul_i64(kResidueCount, L, "64 * L");
    int64_t total_cells = checked_mul_i64(kResidueCount, config.gmax, "64 * gmax");

    OffsetGroupTable table;
    table.gmax = config.gmax;
    table.delta_min = config.delta_min;
    table.delta_dim = config.delta_dim;
    table.item_start.assign(total_cells, 0);
    table.item_count.assign(total_cells, 0);
    table.item_offset_id.assign(total_items, 0);
    table.item_slot.assign(total_items, 0);

    // count offsets per residue and dense delta cell
    for (int64_t residue = 0; residue < kResidueCount; ++residue)
    {
      for (int64_t oid = 0; oid < L; ++oid)
      {
        int32_t dx, dy, dz, slot;
        offset_info(residue, nbr_offsets, oid, &dx, &dy, &dz, &slot);
        TORCH_CHECK(slot >= 0 && slot < kBucketSlots, "invalid local offset slot");
        int64_t gid = dense_delta_gid(
            dx, dy, dz, table.delta_min, table.delta_dim, table.gmax);
        table.item_count[residue * table.gmax + gid] += 1;
      }
    }

    // exclusive prefix sum over dense cells
    int64_t running = 0;
    for (int64_t i = 0; i < total_cells; ++i)
    {
      table.item_start[i] = running;
      running += table.item_count[i];
    }
    TORCH_CHECK(running == total_items, "offset group table size mismatch");

    // fill compact offset items after prefix sum
    std::vector<int64_t> cursor = table.item_start;
    for (int64_t residue = 0; residue < kResidueCount; ++residue)
    {
      for (int64_t oid = 0; oid < L; ++oid)
      {
        int32_t dx, dy, dz, slot;
        offset_info(residue, nbr_offsets, oid, &dx, &dy, &dz, &slot);
        int64_t gid = dense_delta_gid(
            dx, dy, dz, table.delta_min, table.delta_dim, table.gmax);
        int64_t flat = residue * table.gmax + gid;
        int64_t item = cursor[flat]++;
        table.item_offset_id[item] = oid;
        table.item_slot[item] = slot;
      }
    }
    return table;
  }

  // Key record construction and tuple sorting.

  // Convert keys into bucket tuple, local slot, and deterministic sorted records.
  std::vector<KeyRecord> build_key_records_cpu(
      const int32_t *key_coords,
      const int32_t *key_bid,
      const int32_t *coord_min,
      int64_t Nk)
  {
    std::vector<KeyRecord> records;
    records.reserve(static_cast<size_t>(Nk));

    for (int64_t kid = 0; kid < Nk; ++kid)
    {
      int32_t b = key_bid[kid];
      TORCH_CHECK(b >= 0, "key_bid must be non-negative");

      int64_t sx = static_cast<int64_t>(key_coords[kid * 3 + 0]) - coord_min[0];
      int64_t sy = static_cast<int64_t>(key_coords[kid * 3 + 1]) - coord_min[1];
      int64_t sz = static_cast<int64_t>(key_coords[kid * 3 + 2]) - coord_min[2];
      TORCH_CHECK(sx >= 0 && sy >= 0 && sz >= 0,
                  "key_coords - coord_min must be non-negative");

      int64_t bx = sx / kBucketSide;
      int64_t by = sy / kBucketSide;
      int64_t bz = sz / kBucketSide;
      TORCH_CHECK(bx <= kInt32Max && by <= kInt32Max && bz <= kInt32Max,
                  "bucket coordinate overflow");

      int64_t lx = sx - kBucketSide * bx;
      int64_t ly = sy - kBucketSide * by;
      int64_t lz = sz - kBucketSide * bz;
      records.push_back({b,
                         static_cast<int32_t>(bx),
                         static_cast<int32_t>(by),
                         static_cast<int32_t>(bz),
                         static_cast<int32_t>(lx + kBucketSide * ly + 16 * lz),
                         kid});
    }

    std::sort(records.begin(), records.end(), [](const KeyRecord &a, const KeyRecord &b)
              { return std::tie(a.bid, a.bx, a.by, a.bz, a.slot, a.kid) <
                       std::tie(b.bid, b.bx, b.by, b.bz, b.slot, b.kid); });
    return records;
  }

  bool same_bucket(const KeyRecord &a, const KeyRecord &b)
  {
    return a.bid == b.bid && a.bx == b.bx && a.by == b.by && a.bz == b.bz;
  }

  // Bucket record construction and CPU hash-map indexing.

  // Build one compact bucket record and one hash-map entry per non-empty bucket.
  BucketIndex build_bucket_index_cpu(const std::vector<KeyRecord> &records)
  {
    BucketIndex index;
    index.compact_kids.reserve(records.size());
    index.bucket_start.reserve(records.size());
    index.bucket_mask.reserve(records.size());
    index.bucket_map.reserve(records.size());

    for (size_t i = 0; i < records.size();)
    {
      const KeyRecord &first = records[i];
      int64_t record_id = static_cast<int64_t>(index.bucket_start.size());
      BucketKey key{first.bid, first.bx, first.by, first.bz};
      index.bucket_map.emplace(key, record_id);
      index.bucket_start.push_back(static_cast<int64_t>(index.compact_kids.size()));

      uint64_t mask = 0;
      int count = 0;
      size_t p = i;
      for (; p < records.size() && same_bucket(records[p], first); ++p)
      {
        int32_t slot = records[p].slot;
        TORCH_CHECK(slot >= 0 && slot < kBucketSlots, "invalid key local slot");
        uint64_t bit = uint64_t{1} << slot;
        TORCH_CHECK((mask & bit) == 0, "duplicate key coordinate within batch");
        mask |= bit;
        index.compact_kids.push_back(records[p].kid);
        ++count;
      }
      TORCH_CHECK(count <= kBucketSlots, "duplicate key coordinate within batch");
      index.bucket_mask.push_back(mask);
      i = p;
    }
    return index;
  }

  int popcount_u64(uint64_t x)
  {
    return __builtin_popcountll(static_cast<unsigned long long>(x));
  }

  // Query loop.

  // One ordinary CPU loop owns one query at a time.
  // It visits non-empty dense delta cells, checks the bucket bit, then exact radius.
  template <typename scalar_t>
  void query_cpu_impl(
      const scalar_t *query_pos,
      const int32_t *query_bid,
      const int32_t *coord_min,
      const int32_t *nbr_offsets,
      int64_t Nq,
      int64_t L,
      double radius2,
      const OffsetGroupTable &offsets,
      const BucketIndex &buckets,
      int64_t *kids_by_offset)
  {
    int64_t delta_dim = offsets.delta_dim;
    int64_t delta_dim2 = delta_dim * delta_dim;

    for (int64_t qid = 0; qid < Nq; ++qid)
    {
      int32_t qbid = query_bid[qid];
      TORCH_CHECK(qbid >= 0, "query_bid must be non-negative");

      // compute query base cell, residual, bucket, residue
      double qx = static_cast<double>(query_pos[qid * 3 + 0]);
      double qy = static_cast<double>(query_pos[qid * 3 + 1]);
      double qz = static_cast<double>(query_pos[qid * 3 + 2]);
      int64_t base_x = static_cast<int64_t>(std::floor(qx));
      int64_t base_y = static_cast<int64_t>(std::floor(qy));
      int64_t base_z = static_cast<int64_t>(std::floor(qz));
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
      int64_t row = qid * L;

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
        int64_t iz = g / delta_dim2;
        int64_t abs_bx = qbx + offsets.delta_min + ix;
        int64_t abs_by = qby + offsets.delta_min + iy;
        int64_t abs_bz = qbz + offsets.delta_min + iz;
        if (abs_bx < 0 || abs_by < 0 || abs_bz < 0 || abs_bx > kInt32Max ||
            abs_by > kInt32Max || abs_bz > kInt32Max)
        {
          continue;
        }

        // lookup target coarse bucket
        BucketKey key{
            qbid,
            static_cast<int32_t>(abs_bx),
            static_cast<int32_t>(abs_by),
            static_cast<int32_t>(abs_bz)};
        auto found = buckets.bucket_map.find(key);
        if (found == buckets.bucket_map.end())
        {
          continue;
        }

        int64_t record_id = found->second;
        uint64_t mask = buckets.bucket_mask[record_id];
        int64_t start = buckets.bucket_start[record_id];
        int64_t begin = offsets.item_start[flat];

        // test local slot and exact radius
        for (int64_t j = 0; j < count; ++j)
        {
          int64_t item = begin + j;
          int64_t oid = offsets.item_offset_id[item];
          int32_t slot = offsets.item_slot[item];
          uint64_t bit = uint64_t{1} << slot;
          if ((mask & bit) == 0)
          {
            continue;
          }

          int rank = popcount_u64(mask & (bit - 1));
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
  }

  void query_cpu(
      const torch::Tensor &query_pos,
      const torch::Tensor &query_bid,
      const torch::Tensor &coord_min,
      const torch::Tensor &nbr_offsets,
      double radius,
      const OffsetGroupTable &offset_table,
      const BucketIndex &bucket_index,
      torch::Tensor &kids_by_offset)
  {
    int64_t Nq = query_pos.size(0);
    int64_t L = nbr_offsets.size(0);
    double radius2 = radius * radius;

    AT_DISPATCH_FLOATING_TYPES(query_pos.scalar_type(), "query_cpu_impl", [&]
                               { query_cpu_impl<scalar_t>(
                                     query_pos.data_ptr<scalar_t>(), query_bid.data_ptr<int32_t>(),
                                     coord_min.data_ptr<int32_t>(), nbr_offsets.data_ptr<int32_t>(),
                                     Nq, L, radius2, offset_table, bucket_index,
                                     kids_by_offset.data_ptr<int64_t>()); });
  }

} // namespace

// Public dispatch pipeline.

torch::Tensor radius_nbr_kids_by_offset_sparse_lattice_cpu(
    torch::Tensor query_pos,
    torch::Tensor query_bid,
    torch::Tensor key_coords,
    torch::Tensor key_bid,
    double radius,
    torch::Tensor nbr_offsets,
    torch::Tensor coord_min)
{
  validate_inputs_cpu(
      query_pos, query_bid, key_coords, key_bid, radius, nbr_offsets, coord_min);

  int64_t Nq = query_pos.size(0);
  int64_t Nk = key_coords.size(0);
  int64_t L = nbr_offsets.size(0);
  auto int64_options = key_coords.options().dtype(torch::kInt64);
  if (Nq == 0 || L == 0)
  {
    return at::empty({Nq, L}, int64_options);
  }

  // initialize kids_by_offset to -1
  auto kids_by_offset = at::full({Nq, L}, -1, int64_options);
  if (Nk == 0)
  {
    return kids_by_offset;
  }

  // Pipeline:
  // 1. Build dense delta-grid offset groups.
  // 2. Sort key records by bucket tuple and slot.
  // 3. Build compact bucket records and a CPU tuple hash map.
  // 4. Query every row and write kids_by_offset[qid, offset_id].
  auto offset_table = build_offset_group_table_cpu(
      nbr_offsets.data_ptr<int32_t>(), L, radius);
  auto records = build_key_records_cpu(
      key_coords.data_ptr<int32_t>(), key_bid.data_ptr<int32_t>(),
      coord_min.data_ptr<int32_t>(), Nk);
  auto bucket_index = build_bucket_index_cpu(records);
  query_cpu(
      query_pos, query_bid, coord_min, nbr_offsets, radius,
      offset_table, bucket_index, kids_by_offset);
  return kids_by_offset;
}
