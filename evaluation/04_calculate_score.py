import argparse
import json
import math
from datetime import datetime

import cumesh
import numpy as np
import pandas as pd
import torch
import trimesh
from pytorch3d.loss.point_mesh_distance import point_face_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes

from preprocess.utils import Pipeline, Stage
from symtrellis.detection import detect_mesh_symmetry, sample_mesh_farthest_points
from symtrellis.detection.icp import iterative_closest_point
from symtrellis.detection.sampling import sample_random_rotations
from symtrellis.symmetry import get_3d_point_group

from .base import Files, Workspace

AXIS_VECTORS = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}

AXIS_ALIGN_THRESHOLD = math.sqrt(1 - 0.07**2)
DISTANCE_THRESHOLDS = (0.003, 0.01, 0.03, 0.05, 0.10)
FAIL_DISTANCE = math.sqrt(3.0)
MESH_MAX_FACES = 100000
SCORE_NUM_SAMPLES = 16384 * 32
SCORE_BATCH_SIZE = 16384


def load_mesh(mesh_path, device):
    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices).copy()
    faces = np.asarray(mesh.faces).copy()

    if len(faces) > MESH_MAX_FACES:
        cuda_mesh = cumesh.CuMesh()
        cuda_mesh.init(
            torch.from_numpy(vertices).to(device=device, dtype=torch.float32).contiguous(),
            torch.from_numpy(faces).to(device=device, dtype=torch.int32).contiguous(),
        )
        cuda_mesh.simplify(MESH_MAX_FACES, verbose=False)
        vertices, faces = cuda_mesh.read()
        vertices = vertices.cpu().numpy().copy()
        faces = faces.cpu().numpy().copy()

    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 1] *= -1

    vertices = torch.from_numpy(vertices).to(device=device, dtype=torch.float32)
    faces = torch.from_numpy(faces).to(device=device, dtype=torch.int64)
    return vertices, faces


def align_meshes(
    query_vertices,
    query_faces,
    target_vertices,
    target_faces,
    rotations=None,
    translations=None,
    scales=None,
    num_samples=4096,
    num_icp_iter=128,
    num_icp_init=512,
    estimate_scale=False,
):
    device = query_vertices.device
    query_mesh = Meshes(verts=[query_vertices], faces=[query_faces])
    target_mesh = Meshes(verts=[target_vertices], faces=[target_faces])
    query_samples = sample_mesh_farthest_points(query_mesh, num_points=num_samples)
    target_samples = sample_mesh_farthest_points(target_mesh, num_points=num_samples)

    if rotations is None:
        rotations = sample_random_rotations(num_icp_init, device=device)
    if translations is None:
        translations = torch.zeros(rotations.shape[0], 3, device=device, dtype=query_vertices.dtype)
    if scales is None:
        scales = torch.ones(rotations.shape[0], device=device, dtype=query_vertices.dtype)

    batch_size = rotations.shape[0]
    translations = translations.expand(batch_size, -1)
    scales = scales.expand(batch_size)
    query_samples = query_samples[None].expand(batch_size, -1, -1)
    target_samples = target_samples[None].expand(batch_size, -1, -1)

    rotations, translations, scales, _, rmse = iterative_closest_point(
        X=query_samples,
        Y=target_samples,
        init_transform=(rotations, translations, scales),
        max_iterations=num_icp_iter,
        relative_rmse_thr=1e-6,
        estimate_scale=estimate_scale,
        allow_improper=False,
    )
    return rotations, translations, scales, rmse


def point_to_mesh_squared_distance(points, target_mesh):
    vertices = target_mesh.verts_packed()
    faces = target_mesh.faces_packed()
    triangles = vertices[faces].contiguous()
    points_first_index = torch.tensor([0], dtype=torch.long, device=points.device)
    triangles_first_index = torch.tensor([0], dtype=torch.long, device=points.device)

    return point_face_distance(
        points,
        points_first_index,
        triangles,
        triangles_first_index,
        points.shape[0],
        1e-12,
    )


def mesh_surface_chamfer_distances(gt_vertices, gt_faces, prediction_vertices, prediction_faces):
    gt_mesh = Meshes(verts=[gt_vertices], faces=[gt_faces])
    prediction_mesh = Meshes(verts=[prediction_vertices], faces=[prediction_faces])
    gt_to_prediction_chunks = []
    prediction_to_gt_chunks = []

    for start in range(0, SCORE_NUM_SAMPLES, SCORE_BATCH_SIZE):
        batch_size = min(SCORE_BATCH_SIZE, SCORE_NUM_SAMPLES - start)
        gt_samples = sample_points_from_meshes(gt_mesh, num_samples=batch_size)[0]
        prediction_samples = sample_points_from_meshes(prediction_mesh, num_samples=batch_size)[0]

        gt_to_prediction_chunks.append(point_to_mesh_squared_distance(gt_samples, prediction_mesh).sqrt())
        prediction_to_gt_chunks.append(point_to_mesh_squared_distance(prediction_samples, gt_mesh).sqrt())

    return torch.cat(gt_to_prediction_chunks), torch.cat(prediction_to_gt_chunks)


def mesh_surface_symmetry_distances(vertices, faces, transforms, translations):
    mesh = Meshes(verts=[vertices], faces=[faces])
    distance_chunks = []

    for start in range(0, SCORE_NUM_SAMPLES, SCORE_BATCH_SIZE):
        batch_size = min(SCORE_BATCH_SIZE, SCORE_NUM_SAMPLES - start)
        samples = sample_points_from_meshes(mesh, num_samples=batch_size)[0]
        transform_distances = []

        for transform, translation in zip(transforms, translations):
            transformed_samples = samples @ transform.T + translation[None]
            transform_distances.append(point_to_mesh_squared_distance(transformed_samples, mesh).sqrt())

        distance_chunks.append(torch.stack(transform_distances, dim=1))

    return torch.cat(distance_chunks)


@torch.no_grad()
def calculate_score(
    idx,
    shape_id,
    view_id,
    symmetry_group,
    symmetry_fold,
    symmetry_axis,
    major_axis,
    num_samples,
    intrinsic_dim,
    num_icp_iter,
    num_icp_init,
    max_fold,
    num_icp_refine_samples,
    num_icp_refine_iter,
    prediction_files,
    ground_truth_files,
    output_files,
):
    fallback_fold = 1 if symmetry_group == "S1" else max_fold
    result = {
        "eval_success": False,
        "fail_reason": "",
        "pred_fold": fallback_fold,
        "pred_major_axis": [0.0, 0.0, 1.0],
        "pred_minor_axis": [1.0, 0.0, 0.0],
        "pred_center": [0.0, 0.0, 0.0],
        "rot_axes": [],
        "refl_planes": [],
        "self_symm_result": {
            "mean_sd": FAIL_DISTANCE,
            "max_sd": FAIL_DISTANCE,
            "mean_bad_rate": {str(threshold): 1.0 for threshold in DISTANCE_THRESHOLDS},
            "max_bad_rate": {str(threshold): 1.0 for threshold in DISTANCE_THRESHOLDS},
        },
        "reconstruction_result": {
            "cd_recon": FAIL_DISTANCE,
        },
    }

    prediction_path = prediction_files.path(".glb", shape_id=shape_id, view_id=view_id)
    output_path = output_files.path(".json", shape_id=shape_id, view_id=view_id)
    if not prediction_path.is_file():
        result["fail_reason"] = "no prediction mesh"
        output_path.write_text(json.dumps(result, indent=4), encoding="utf-8")
        return {"idx": idx, output_files.rel_path: True}

    device = torch.device("cuda:0")
    prediction_vertices, prediction_faces = load_mesh(prediction_path, device)
    detection = detect_mesh_symmetry(
        verts=prediction_vertices,
        faces=prediction_faces,
        num_samples=num_samples,
        intrinsic_dim=intrinsic_dim,
        num_icp_iter=num_icp_iter,
        num_icp_init=num_icp_init,
        max_fold=max_fold,
    )
    rotation_candidates = detection["rotational_symmetry"]
    reflection_candidates = detection["reflectional_symmetry"]
    result["rot_axes"] = rotation_candidates
    result["refl_planes"] = reflection_candidates

    if (symmetry_group == "S1" and not reflection_candidates) or (symmetry_group != "S1" and not rotation_candidates):
        result["fail_reason"] = "no symmery detected"
        output_path.write_text(json.dumps(result, indent=4), encoding="utf-8")
        return {"idx": idx, output_files.rel_path: True}

    if symmetry_group != "S1":
        aligned_candidate = max(rotation_candidates, key=lambda candidate: abs(np.dot(candidate["axis"], major_axis)))
        alignment = abs(np.dot(aligned_candidate["axis"], major_axis))
        selected_candidate = aligned_candidate if alignment > AXIS_ALIGN_THRESHOLD else min(rotation_candidates, key=lambda candidate: candidate["rmse"])
        pred_fold = symmetry_fold if selected_candidate["fold_e"] % symmetry_fold == 0 or selected_candidate["fold_i"] % symmetry_fold == 0 else selected_candidate["fold_i"]
        pred_major_axis = torch.tensor(selected_candidate["axis"], device=device, dtype=torch.float32)
        pred_center = torch.tensor(selected_candidate["q"], device=device, dtype=torch.float32)
        symmetry_label = f"C{pred_fold}"
    else:
        aligned_candidate = max(reflection_candidates, key=lambda candidate: abs(np.dot(candidate["n"], major_axis)))
        alignment = abs(np.dot(aligned_candidate["n"], major_axis))
        selected_candidate = aligned_candidate if alignment > AXIS_ALIGN_THRESHOLD else min(reflection_candidates, key=lambda candidate: candidate["rmse"])
        pred_fold = 1
        pred_major_axis = torch.tensor(selected_candidate["n"], device=device, dtype=torch.float32)
        pred_center = selected_candidate["c"] * pred_major_axis
        symmetry_label = "S1"

    major_direction = pred_major_axis / pred_major_axis.norm()
    minor_index = (pred_major_axis.abs().argmax().item() + 1) % 3
    pred_minor_axis = torch.zeros_like(pred_major_axis)
    pred_minor_axis[minor_index] = 1.0
    pred_minor_axis = pred_minor_axis - (pred_minor_axis * major_direction).sum() * major_direction
    pred_minor_axis = pred_minor_axis / pred_minor_axis.norm()

    transforms, translations, _ = get_3d_point_group(
        label=symmetry_label,
        center=pred_center,
        major_axis=pred_major_axis,
        minor_axis=None,
        include_identity=False,
    )
    transforms, translations, _, _ = align_meshes(
        query_vertices=prediction_vertices,
        query_faces=prediction_faces,
        target_vertices=prediction_vertices,
        target_faces=prediction_faces,
        rotations=transforms,
        translations=translations,
        scales=torch.ones(transforms.shape[0], device=device, dtype=prediction_vertices.dtype),
        num_samples=num_icp_refine_samples,
        num_icp_iter=num_icp_refine_iter,
        num_icp_init=num_icp_init,
        estimate_scale=False,
    )

    symmetry_distances = mesh_surface_symmetry_distances(
        vertices=prediction_vertices,
        faces=prediction_faces,
        transforms=transforms,
        translations=translations,
    )
    max_symmetry_distances = symmetry_distances.amax(dim=1)
    self_symmetry_result = {
        "mean_sd": symmetry_distances.mean().item(),
        "max_sd": max_symmetry_distances.mean().item(),
        "mean_bad_rate": {str(threshold): (symmetry_distances > threshold).float().mean().item() for threshold in DISTANCE_THRESHOLDS},
        "max_bad_rate": {str(threshold): (max_symmetry_distances > threshold).float().mean().item() for threshold in DISTANCE_THRESHOLDS},
    }

    ground_truth_path = ground_truth_files.path(
        ".glb",
        shape_id=shape_id,
        symmetry_group=symmetry_group.lower(),
        symmetry_axis=symmetry_axis,
    )
    gt_vertices, gt_faces = load_mesh(ground_truth_path, device)
    prediction_center = prediction_vertices.mean(dim=0)
    gt_center = gt_vertices.mean(dim=0)
    prediction_radius = (prediction_vertices - prediction_center).norm(dim=1).mean().clamp_min(1e-6)
    gt_radius = (gt_vertices - gt_center).norm(dim=1).mean().clamp_min(1e-6)
    initial_scale = gt_radius / prediction_radius

    rotations, translations, scales, rmse = align_meshes(
        query_vertices=prediction_vertices,
        query_faces=prediction_faces,
        target_vertices=gt_vertices,
        target_faces=gt_faces,
        scales=initial_scale[None],
        num_samples=num_samples,
        num_icp_iter=num_icp_iter,
        num_icp_init=num_icp_init,
        estimate_scale=False,
    )
    best = rmse.argmin().item()
    rotations, translations, scales, _ = align_meshes(
        query_vertices=prediction_vertices,
        query_faces=prediction_faces,
        target_vertices=gt_vertices,
        target_faces=gt_faces,
        rotations=rotations[best : best + 1],
        translations=translations[best : best + 1],
        scales=scales[best : best + 1],
        num_samples=num_icp_refine_samples,
        num_icp_iter=num_icp_refine_iter,
        num_icp_init=num_icp_init,
        estimate_scale=False,
    )
    aligned_prediction_vertices = scales[0] * prediction_vertices @ rotations[0].T + translations[0:1]
    gt_to_prediction, prediction_to_gt = mesh_surface_chamfer_distances(
        gt_vertices=gt_vertices,
        gt_faces=gt_faces,
        prediction_vertices=aligned_prediction_vertices,
        prediction_faces=prediction_faces,
    )

    result.update(
        {
            "eval_success": True,
            "pred_fold": pred_fold,
            "pred_major_axis": pred_major_axis.tolist(),
            "pred_minor_axis": pred_minor_axis.tolist(),
            "pred_center": pred_center.tolist(),
            "self_symm_result": self_symmetry_result,
            "reconstruction_result": {"cd_recon": (0.5 * (gt_to_prediction.mean() + prediction_to_gt.mean())).item()},
        }
    )
    output_path.write_text(json.dumps(result, indent=4), encoding="utf-8")
    return {"idx": idx, output_files.rel_path: True}


def main():
    parser = argparse.ArgumentParser(description="Calculate symmetry and reconstruction scores")
    parser.add_argument("--workspace-dir", required=True)
    parser.add_argument("--shape-folder", required=True)
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--intrinsic-dim", type=int, default=64)
    parser.add_argument("--num-icp-iter", type=int, default=128)
    parser.add_argument("--num-icp-init", type=int, default=512)
    parser.add_argument("--max-fold", type=int, default=30)
    parser.add_argument("--num-icp-refine-samples", type=int, default=16384)
    parser.add_argument("--num-icp-refine-iter", type=int, default=32)
    parser.add_argument("--recompute-finished", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()

    workspace = Workspace(args.workspace_dir)
    shape_tag = args.shape_folder.strip("/").replace("/", "_")
    experiment_name = f"eval_score_{shape_tag}"

    prediction_files = workspace.files(args.shape_folder)
    ground_truth_files = workspace.files(
        "shapes",
        format="{shape_id:06d}_{symmetry_group}_{symmetry_axis}",
    )
    output_files = workspace.files(f"experiments/{experiment_name}")
    output_files.mkdir()

    metadata = workspace.read_metadata().sort_values("idx")
    metadata = metadata.loc[metadata[args.shape_folder]]
    if not args.recompute_finished:
        metadata = metadata.loc[[not output_files.exists(shape_id=row["shape_id"], view_id=row["view_id"]) for _, row in metadata.iterrows()]]

    start = len(metadata) * args.rank // args.world_size
    end = len(metadata) * (args.rank + 1) // args.world_size
    selected = metadata.iloc[start:end]
    if selected.empty:
        return

    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    inputs = []

    for _, row in selected.iterrows():
        shape_label = workspace.shape_labels[row["shape_id"]]
        inputs.append(
            {
                "idx": row["idx"],
                "shape_id": row["shape_id"],
                "view_id": row["view_id"],
                "symmetry_group": shape_label["symmetry_group"],
                "symmetry_fold": shape_label["symmetry_fold"],
                "symmetry_axis": shape_label["symmetry_axis"],
                "major_axis": AXIS_VECTORS[shape_label["major_axis"]],
            }
        )

    stages = [
        Stage(
            "calculate scores",
            calculate_score,
            params={
                "num_samples": args.num_samples,
                "intrinsic_dim": args.intrinsic_dim,
                "num_icp_iter": args.num_icp_iter,
                "num_icp_init": args.num_icp_init,
                "max_fold": args.max_fold,
                "num_icp_refine_samples": args.num_icp_refine_samples,
                "num_icp_refine_iter": args.num_icp_refine_iter,
            },
            resources={
                "prediction_files": prediction_files,
                "ground_truth_files": ground_truth_files,
                "output_files": output_files,
            },
        )
    ]
    results = Pipeline(stages, total=len(selected)).run(inputs)

    records = pd.DataFrame(results, columns=["idx", output_files.rel_path])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    record_path = workspace.path(f"unmerged_records/04_calculate_score_{timestamp}_rank{args.rank}.csv")
    records.to_csv(record_path, index=False)

    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
    print(f"peak allocated: {peak_allocated:.1f} MiB")
    print(f"peak reserved:  {peak_reserved:.1f} MiB")


if __name__ == "__main__":
    main()
