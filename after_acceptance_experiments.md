# After-acceptance evaluation

## Initial metadata check

```bash
# Full metadata check
FULL_CHECK=1 bash slurm/00_sync_metadata.sbatch
```

## Wave 1

```bash
# Generate accepted-table GT shapes from old GT sparse structures
# Default GPU memory
sbatch --export=ALL,MAPPER_PATH=trellis2_shape_neighbor_graph_pretrain,SPARSE_STRUCTURE_FOLDER=experiments/sparse_old_gt,SYMMETRY_PREDICTION_FOLDER=experiments/old_gt slurm/03_generate_shape.sbatch
# Expanded GPU memory
sbatch --gpus=h100_3g.40gb:1 --export=ALL,MAPPER_PATH=trellis2_shape_neighbor_graph_pretrain,SPARSE_STRUCTURE_FOLDER=experiments/sparse_old_gt,SYMMETRY_PREDICTION_FOLDER=experiments/old_gt slurm/03_generate_shape.sbatch
# Sync metadata
FOLDER=experiments/shape_s114514_ns0.2_gs0.4_gd0.3_m_trellis2_shape_neighbor_graph_pretrain_ss_sparse_old_gt_sym_old_gt bash slurm/00_sync_metadata.sbatch

# Generate accepted-table predicted-symmetry shapes from old predicted sparse structures
# Default GPU memory
sbatch --export=ALL,MAPPER_PATH=trellis2_shape_neighbor_graph_pretrain,SPARSE_STRUCTURE_FOLDER=experiments/sparse_old_pred,SYMMETRY_PREDICTION_FOLDER=experiments/old_pred slurm/03_generate_shape.sbatch
# Expanded GPU memory
sbatch --gpus=h100_3g.40gb:1 --export=ALL,MAPPER_PATH=trellis2_shape_neighbor_graph_pretrain,SPARSE_STRUCTURE_FOLDER=experiments/sparse_old_pred,SYMMETRY_PREDICTION_FOLDER=experiments/old_pred slurm/03_generate_shape.sbatch
# Sync metadata
FOLDER=experiments/shape_s114514_ns0.2_gs0.4_gd0.3_m_trellis2_shape_neighbor_graph_pretrain_ss_sparse_old_pred_sym_old_pred bash slurm/00_sync_metadata.sbatch

# Generate new GT sparse structures
# Default GPU memory
sbatch --export=ALL,MAPPER_PATH=trellis2_sparse_structure_neighbor_graph_finetune slurm/02_generate_sparse_structure.sbatch
# Expanded GPU memory
sbatch --gpus=h100_3g.40gb:1 --export=ALL,MAPPER_PATH=trellis2_sparse_structure_neighbor_graph_finetune slurm/02_generate_sparse_structure.sbatch
# Sync metadata
FOLDER=experiments/ss_s114514_ns0.2_gs0.9_gd0.3_m_trellis2_sparse_structure_neighbor_graph_finetune_sym_gt_vc0 bash slurm/00_sync_metadata.sbatch

# Generate new predicted-symmetry sparse structures
# Default GPU memory
sbatch --export=ALL,MAPPER_PATH=trellis2_sparse_structure_neighbor_graph_finetune,SYMMETRY_PREDICTION_FOLDER=experiments/old_pred slurm/02_generate_sparse_structure.sbatch
# Expanded GPU memory
sbatch --gpus=h100_3g.40gb:1 --export=ALL,MAPPER_PATH=trellis2_sparse_structure_neighbor_graph_finetune,SYMMETRY_PREDICTION_FOLDER=experiments/old_pred slurm/02_generate_sparse_structure.sbatch
# Sync metadata
FOLDER=experiments/ss_s114514_ns0.2_gs0.9_gd0.3_m_trellis2_sparse_structure_neighbor_graph_finetune_sym_old_pred_vc0 bash slurm/00_sync_metadata.sbatch

# Postprocess vanilla TRELLIS.2 with GT symmetry
# Default GPU memory
sbatch --export=ALL,SHAPE_FOLDER=experiments/vanilla_trellis2 slurm/07_symmetry_postprocess.sbatch
# Expanded GPU memory
sbatch --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 --export=ALL,SHAPE_FOLDER=experiments/vanilla_trellis2 slurm/07_symmetry_postprocess.sbatch
# Sync metadata
FOLDER=experiments/postprocess_vanilla_trellis2_symmetry_gt/sector_replication bash slurm/00_sync_metadata.sbatch && FOLDER=experiments/postprocess_vanilla_trellis2_symmetry_gt/voxel_majority bash slurm/00_sync_metadata.sbatch && FOLDER=experiments/postprocess_vanilla_trellis2_symmetry_gt/closest_point_average bash slurm/00_sync_metadata.sbatch

# Postprocess vanilla TRELLIS.2 with predicted symmetry
# Default GPU memory
sbatch --export=ALL,SHAPE_FOLDER=experiments/vanilla_trellis2,SYMMETRY_PREDICTION_FOLDER=experiments/old_pred slurm/07_symmetry_postprocess.sbatch
# Expanded GPU memory
sbatch --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 --export=ALL,SHAPE_FOLDER=experiments/vanilla_trellis2,SYMMETRY_PREDICTION_FOLDER=experiments/old_pred slurm/07_symmetry_postprocess.sbatch
# Sync metadata
FOLDER=experiments/postprocess_vanilla_trellis2_symmetry_old_pred/sector_replication bash slurm/00_sync_metadata.sbatch && FOLDER=experiments/postprocess_vanilla_trellis2_symmetry_old_pred/voxel_majority bash slurm/00_sync_metadata.sbatch && FOLDER=experiments/postprocess_vanilla_trellis2_symmetry_old_pred/closest_point_average bash slurm/00_sync_metadata.sbatch
```

## Wave 2

```bash
# Score accepted-table GT shapes
# Default GPU memory
sbatch --export=ALL,SHAPE_FOLDER=experiments/shape_s114514_ns0.2_gs0.4_gd0.3_m_trellis2_shape_neighbor_graph_pretrain_ss_sparse_old_gt_sym_old_gt slurm/04_calculate_score.sbatch
# Expanded GPU memory
sbatch --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 --export=ALL,SHAPE_FOLDER=experiments/shape_s114514_ns0.2_gs0.4_gd0.3_m_trellis2_shape_neighbor_graph_pretrain_ss_sparse_old_gt_sym_old_gt slurm/04_calculate_score.sbatch
# Sync metadata
FOLDER=experiments/score_shape_s114514_ns0.2_gs0.4_gd0.3_m_trellis2_shape_neighbor_graph_pretrain_ss_sparse_old_gt_sym_old_gt bash slurm/00_sync_metadata.sbatch

# Score accepted-table predicted-symmetry shapes
# Default GPU memory
sbatch --export=ALL,SHAPE_FOLDER=experiments/shape_s114514_ns0.2_gs0.4_gd0.3_m_trellis2_shape_neighbor_graph_pretrain_ss_sparse_old_pred_sym_old_pred slurm/04_calculate_score.sbatch
# Expanded GPU memory
sbatch --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 --export=ALL,SHAPE_FOLDER=experiments/shape_s114514_ns0.2_gs0.4_gd0.3_m_trellis2_shape_neighbor_graph_pretrain_ss_sparse_old_pred_sym_old_pred slurm/04_calculate_score.sbatch
# Sync metadata
FOLDER=experiments/score_shape_s114514_ns0.2_gs0.4_gd0.3_m_trellis2_shape_neighbor_graph_pretrain_ss_sparse_old_pred_sym_old_pred bash slurm/00_sync_metadata.sbatch

# Generate fully new GT shapes
# Default GPU memory
sbatch --export=ALL,MAPPER_PATH=trellis2_shape_neighbor_graph_pretrain,SPARSE_STRUCTURE_FOLDER=experiments/ss_s114514_ns0.2_gs0.9_gd0.3_m_trellis2_sparse_structure_neighbor_graph_finetune_sym_gt_vc0 slurm/03_generate_shape.sbatch
# Expanded GPU memory
sbatch --gpus=h100_3g.40gb:1 --export=ALL,MAPPER_PATH=trellis2_shape_neighbor_graph_pretrain,SPARSE_STRUCTURE_FOLDER=experiments/ss_s114514_ns0.2_gs0.9_gd0.3_m_trellis2_sparse_structure_neighbor_graph_finetune_sym_gt_vc0 slurm/03_generate_shape.sbatch
# Sync metadata
FOLDER=experiments/shape_s114514_ns0.2_gs0.4_gd0.3_m_trellis2_shape_neighbor_graph_pretrain_ss_ss_s114514_ns0.2_gs0.9_gd0.3_m_trellis2_sparse_structure_neighbor_graph_finetune_sym_gt_vc0_sym_gt bash slurm/00_sync_metadata.sbatch

# Generate fully new predicted-symmetry shapes
# Default GPU memory
sbatch --export=ALL,MAPPER_PATH=trellis2_shape_neighbor_graph_pretrain,SPARSE_STRUCTURE_FOLDER=experiments/ss_s114514_ns0.2_gs0.9_gd0.3_m_trellis2_sparse_structure_neighbor_graph_finetune_sym_old_pred_vc0,SYMMETRY_PREDICTION_FOLDER=experiments/old_pred slurm/03_generate_shape.sbatch
# Expanded GPU memory
sbatch --gpus=h100_3g.40gb:1 --export=ALL,MAPPER_PATH=trellis2_shape_neighbor_graph_pretrain,SPARSE_STRUCTURE_FOLDER=experiments/ss_s114514_ns0.2_gs0.9_gd0.3_m_trellis2_sparse_structure_neighbor_graph_finetune_sym_old_pred_vc0,SYMMETRY_PREDICTION_FOLDER=experiments/old_pred slurm/03_generate_shape.sbatch
# Sync metadata
FOLDER=experiments/shape_s114514_ns0.2_gs0.4_gd0.3_m_trellis2_shape_neighbor_graph_pretrain_ss_ss_s114514_ns0.2_gs0.9_gd0.3_m_trellis2_sparse_structure_neighbor_graph_finetune_sym_old_pred_vc0_sym_old_pred bash slurm/00_sync_metadata.sbatch

# Generate SS-only GT shapes
# Default GPU memory
sbatch --export=ALL,MAPPER_PATH=trellis2_shape_neighbor_graph_pretrain,SPARSE_STRUCTURE_FOLDER=experiments/ss_s114514_ns0.2_gs0.9_gd0.3_m_trellis2_sparse_structure_neighbor_graph_finetune_sym_gt_vc0,NOISE_STRENGTH=0,GUIDANCE_STRENGTH=0,GUIDANCE_DURATION=0 slurm/03_generate_shape.sbatch
# Expanded GPU memory
sbatch --gpus=h100_3g.40gb:1 --export=ALL,MAPPER_PATH=trellis2_shape_neighbor_graph_pretrain,SPARSE_STRUCTURE_FOLDER=experiments/ss_s114514_ns0.2_gs0.9_gd0.3_m_trellis2_sparse_structure_neighbor_graph_finetune_sym_gt_vc0,NOISE_STRENGTH=0,GUIDANCE_STRENGTH=0,GUIDANCE_DURATION=0 slurm/03_generate_shape.sbatch
# Sync metadata
FOLDER=experiments/shape_s114514_ns0.0_gs0.0_gd0.0_m_trellis2_shape_neighbor_graph_pretrain_ss_ss_s114514_ns0.2_gs0.9_gd0.3_m_trellis2_sparse_structure_neighbor_graph_finetune_sym_gt_vc0_sym_gt bash slurm/00_sync_metadata.sbatch

# Generate SS-only predicted-symmetry shapes
# Default GPU memory
sbatch --export=ALL,MAPPER_PATH=trellis2_shape_neighbor_graph_pretrain,SPARSE_STRUCTURE_FOLDER=experiments/ss_s114514_ns0.2_gs0.9_gd0.3_m_trellis2_sparse_structure_neighbor_graph_finetune_sym_old_pred_vc0,NOISE_STRENGTH=0,GUIDANCE_STRENGTH=0,GUIDANCE_DURATION=0 slurm/03_generate_shape.sbatch
# Expanded GPU memory
sbatch --gpus=h100_3g.40gb:1 --export=ALL,MAPPER_PATH=trellis2_shape_neighbor_graph_pretrain,SPARSE_STRUCTURE_FOLDER=experiments/ss_s114514_ns0.2_gs0.9_gd0.3_m_trellis2_sparse_structure_neighbor_graph_finetune_sym_old_pred_vc0,NOISE_STRENGTH=0,GUIDANCE_STRENGTH=0,GUIDANCE_DURATION=0 slurm/03_generate_shape.sbatch
# Sync metadata
FOLDER=experiments/shape_s114514_ns0.0_gs0.0_gd0.0_m_trellis2_shape_neighbor_graph_pretrain_ss_ss_s114514_ns0.2_gs0.9_gd0.3_m_trellis2_sparse_structure_neighbor_graph_finetune_sym_old_pred_vc0_sym_gt bash slurm/00_sync_metadata.sbatch

# Score GT sector replication
# Default GPU memory
sbatch --export=ALL,SHAPE_FOLDER=experiments/postprocess_vanilla_trellis2_symmetry_gt/sector_replication slurm/04_calculate_score.sbatch
# Expanded GPU memory
sbatch --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 --export=ALL,SHAPE_FOLDER=experiments/postprocess_vanilla_trellis2_symmetry_gt/sector_replication slurm/04_calculate_score.sbatch
# Sync metadata
FOLDER=experiments/score_postprocess_vanilla_trellis2_symmetry_gt_sector_replication bash slurm/00_sync_metadata.sbatch

# Score GT voxel majority
# Default GPU memory
sbatch --export=ALL,SHAPE_FOLDER=experiments/postprocess_vanilla_trellis2_symmetry_gt/voxel_majority slurm/04_calculate_score.sbatch
# Expanded GPU memory
sbatch --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 --export=ALL,SHAPE_FOLDER=experiments/postprocess_vanilla_trellis2_symmetry_gt/voxel_majority slurm/04_calculate_score.sbatch
# Sync metadata
FOLDER=experiments/score_postprocess_vanilla_trellis2_symmetry_gt_voxel_majority bash slurm/00_sync_metadata.sbatch

# Score GT closest-point average
# Default GPU memory
sbatch --export=ALL,SHAPE_FOLDER=experiments/postprocess_vanilla_trellis2_symmetry_gt/closest_point_average slurm/04_calculate_score.sbatch
# Expanded GPU memory
sbatch --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 --export=ALL,SHAPE_FOLDER=experiments/postprocess_vanilla_trellis2_symmetry_gt/closest_point_average slurm/04_calculate_score.sbatch
# Sync metadata
FOLDER=experiments/score_postprocess_vanilla_trellis2_symmetry_gt_closest_point_average bash slurm/00_sync_metadata.sbatch

# Score predicted-symmetry sector replication
# Default GPU memory
sbatch --export=ALL,SHAPE_FOLDER=experiments/postprocess_vanilla_trellis2_symmetry_old_pred/sector_replication slurm/04_calculate_score.sbatch
# Expanded GPU memory
sbatch --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 --export=ALL,SHAPE_FOLDER=experiments/postprocess_vanilla_trellis2_symmetry_old_pred/sector_replication slurm/04_calculate_score.sbatch
# Sync metadata
FOLDER=experiments/score_postprocess_vanilla_trellis2_symmetry_old_pred_sector_replication bash slurm/00_sync_metadata.sbatch

# Score predicted-symmetry voxel majority
# Default GPU memory
sbatch --export=ALL,SHAPE_FOLDER=experiments/postprocess_vanilla_trellis2_symmetry_old_pred/voxel_majority slurm/04_calculate_score.sbatch
# Expanded GPU memory
sbatch --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 --export=ALL,SHAPE_FOLDER=experiments/postprocess_vanilla_trellis2_symmetry_old_pred/voxel_majority slurm/04_calculate_score.sbatch
# Sync metadata
FOLDER=experiments/score_postprocess_vanilla_trellis2_symmetry_old_pred_voxel_majority bash slurm/00_sync_metadata.sbatch

# Score predicted-symmetry closest-point average
# Default GPU memory
sbatch --export=ALL,SHAPE_FOLDER=experiments/postprocess_vanilla_trellis2_symmetry_old_pred/closest_point_average slurm/04_calculate_score.sbatch
# Expanded GPU memory
sbatch --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 --export=ALL,SHAPE_FOLDER=experiments/postprocess_vanilla_trellis2_symmetry_old_pred/closest_point_average slurm/04_calculate_score.sbatch
# Sync metadata
FOLDER=experiments/score_postprocess_vanilla_trellis2_symmetry_old_pred_closest_point_average bash slurm/00_sync_metadata.sbatch
```

## Wave 3

```bash
# Score fully new GT shapes
# Default GPU memory
sbatch --export=ALL,SHAPE_FOLDER=experiments/shape_s114514_ns0.2_gs0.4_gd0.3_m_trellis2_shape_neighbor_graph_pretrain_ss_ss_s114514_ns0.2_gs0.9_gd0.3_m_trellis2_sparse_structure_neighbor_graph_finetune_sym_gt_vc0_sym_gt slurm/04_calculate_score.sbatch
# Expanded GPU memory
sbatch --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 --export=ALL,SHAPE_FOLDER=experiments/shape_s114514_ns0.2_gs0.4_gd0.3_m_trellis2_shape_neighbor_graph_pretrain_ss_ss_s114514_ns0.2_gs0.9_gd0.3_m_trellis2_sparse_structure_neighbor_graph_finetune_sym_gt_vc0_sym_gt slurm/04_calculate_score.sbatch
# Sync metadata
FOLDER=experiments/score_shape_s114514_ns0.2_gs0.4_gd0.3_m_trellis2_shape_neighbor_graph_pretrain_ss_ss_s114514_ns0.2_gs0.9_gd0.3_m_trellis2_sparse_structure_neighbor_graph_finetune_sym_gt_vc0_sym_gt bash slurm/00_sync_metadata.sbatch

# Score fully new predicted-symmetry shapes
# Default GPU memory
sbatch --export=ALL,SHAPE_FOLDER=experiments/shape_s114514_ns0.2_gs0.4_gd0.3_m_trellis2_shape_neighbor_graph_pretrain_ss_ss_s114514_ns0.2_gs0.9_gd0.3_m_trellis2_sparse_structure_neighbor_graph_finetune_sym_old_pred_vc0_sym_old_pred slurm/04_calculate_score.sbatch
# Expanded GPU memory
sbatch --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 --export=ALL,SHAPE_FOLDER=experiments/shape_s114514_ns0.2_gs0.4_gd0.3_m_trellis2_shape_neighbor_graph_pretrain_ss_ss_s114514_ns0.2_gs0.9_gd0.3_m_trellis2_sparse_structure_neighbor_graph_finetune_sym_old_pred_vc0_sym_old_pred slurm/04_calculate_score.sbatch
# Sync metadata
FOLDER=experiments/score_shape_s114514_ns0.2_gs0.4_gd0.3_m_trellis2_shape_neighbor_graph_pretrain_ss_ss_s114514_ns0.2_gs0.9_gd0.3_m_trellis2_sparse_structure_neighbor_graph_finetune_sym_old_pred_vc0_sym_old_pred bash slurm/00_sync_metadata.sbatch

# Score SS-only GT shapes
# Default GPU memory
sbatch --export=ALL,SHAPE_FOLDER=experiments/shape_s114514_ns0.0_gs0.0_gd0.0_m_trellis2_shape_neighbor_graph_pretrain_ss_ss_s114514_ns0.2_gs0.9_gd0.3_m_trellis2_sparse_structure_neighbor_graph_finetune_sym_gt_vc0_sym_gt slurm/04_calculate_score.sbatch
# Expanded GPU memory
sbatch --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 --export=ALL,SHAPE_FOLDER=experiments/shape_s114514_ns0.0_gs0.0_gd0.0_m_trellis2_shape_neighbor_graph_pretrain_ss_ss_s114514_ns0.2_gs0.9_gd0.3_m_trellis2_sparse_structure_neighbor_graph_finetune_sym_gt_vc0_sym_gt slurm/04_calculate_score.sbatch
# Sync metadata
FOLDER=experiments/score_shape_s114514_ns0.0_gs0.0_gd0.0_m_trellis2_shape_neighbor_graph_pretrain_ss_ss_s114514_ns0.2_gs0.9_gd0.3_m_trellis2_sparse_structure_neighbor_graph_finetune_sym_gt_vc0_sym_gt bash slurm/00_sync_metadata.sbatch

# Score SS-only predicted-symmetry shapes
# Default GPU memory
sbatch --export=ALL,SHAPE_FOLDER=experiments/shape_s114514_ns0.0_gs0.0_gd0.0_m_trellis2_shape_neighbor_graph_pretrain_ss_ss_s114514_ns0.2_gs0.9_gd0.3_m_trellis2_sparse_structure_neighbor_graph_finetune_sym_old_pred_vc0_sym_gt slurm/04_calculate_score.sbatch
# Expanded GPU memory
sbatch --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 --export=ALL,SHAPE_FOLDER=experiments/shape_s114514_ns0.0_gs0.0_gd0.0_m_trellis2_shape_neighbor_graph_pretrain_ss_ss_s114514_ns0.2_gs0.9_gd0.3_m_trellis2_sparse_structure_neighbor_graph_finetune_sym_old_pred_vc0_sym_gt slurm/04_calculate_score.sbatch
# Sync metadata
FOLDER=experiments/score_shape_s114514_ns0.0_gs0.0_gd0.0_m_trellis2_shape_neighbor_graph_pretrain_ss_ss_s114514_ns0.2_gs0.9_gd0.3_m_trellis2_sparse_structure_neighbor_graph_finetune_sym_old_pred_vc0_sym_gt bash slurm/00_sync_metadata.sbatch
```
