import argparse
import json
import math
from datetime import datetime

import cumesh
import numpy as np
import pandas as pd
import torch
import trimesh
from skimage import measure

from preprocess.utils import Pipeline, Stage
from symtrellis.symmetry import get_3d_point_group

from .base import Workspace

AXIS_VECTORS = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}


def sector_replication(
    vertices,
    faces,
    center,
    major_axis,
    transforms,
    translations,
    signs,
    phase_samples,
):
    axis = major_axis / major_axis.norm().clamp_min(1e-12)
    triangles = vertices[faces]
    face_area = 0.5 * torch.linalg.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]).norm(dim=-1)

    if signs.sum().item() < transforms.shape[0]:
        vertex_sector = (((vertices - center) * axis).sum(dim=-1) >= 0).long()
        face_sector_vertices = vertex_sector[faces]
        face_sector = face_sector_vertices[:, 0]
        non_boundary = (face_sector_vertices[:, 0] == face_sector_vertices[:, 1]) & (face_sector_vertices[:, 1] == face_sector_vertices[:, 2])
        sector_count = 2
    else:
        local_x_axis = torch.tensor([1.0, 0.0, 0.0], device=vertices.device)
        if torch.abs((local_x_axis * axis).sum()) > 0.9:
            local_x_axis = torch.tensor([0.0, 1.0, 0.0], device=vertices.device)
        local_x_axis = local_x_axis - (local_x_axis * axis).sum() * axis
        local_x_axis = local_x_axis / local_x_axis.norm().clamp_min(1e-12)
        local_y_axis = torch.linalg.cross(axis, local_x_axis)
        local_y_axis = local_y_axis / local_y_axis.norm().clamp_min(1e-12)

        centered_vertices = vertices - center
        local_x = (centered_vertices * local_x_axis).sum(dim=-1)
        local_y = (centered_vertices * local_y_axis).sum(dim=-1)
        theta = torch.remainder(torch.atan2(local_y, local_x), 2.0 * math.pi)
        sector_count = transforms.shape[0]
        sector_angle = 2.0 * math.pi / sector_count
        best_cost = None

        for phase_index in range(phase_samples):
            phase = phase_index * sector_angle / phase_samples
            shifted_theta = torch.remainder(theta - phase, 2.0 * math.pi)
            vertex_sector = torch.floor(shifted_theta / sector_angle).long().clamp(max=sector_count - 1)
            face_sector_vertices = vertex_sector[faces]
            same_sector = (face_sector_vertices[:, 0] == face_sector_vertices[:, 1]) & (face_sector_vertices[:, 1] == face_sector_vertices[:, 2])
            boundary_cost = face_area[~same_sector].sum().item()

            if best_cost is None or boundary_cost < best_cost:
                best_cost = boundary_cost
                face_sector = face_sector_vertices[:, 0].clone()
                non_boundary = same_sector.clone()

    faces_numpy = faces.cpu().numpy()
    face_area_numpy = face_area.cpu().numpy()
    best_component_area = -1.0
    best_component_face_ids = None

    for sector_id in range(sector_count):
        sector_face_ids = torch.nonzero(non_boundary & (face_sector == sector_id), as_tuple=False).squeeze(1).cpu().numpy()
        if sector_face_ids.size == 0:
            continue

        sector_faces = faces_numpy[sector_face_ids]
        adjacency = trimesh.graph.face_adjacency(faces=sector_faces)
        components = trimesh.graph.connected_components(
            adjacency,
            nodes=np.arange(len(sector_face_ids)),
            engine="scipy",
        )

        for component in components:
            component_face_ids = sector_face_ids[np.asarray(component, dtype=np.int64)]
            component_area = face_area_numpy[component_face_ids].sum()
            if component_area > best_component_area:
                best_component_area = component_area
                best_component_face_ids = component_face_ids

    selected_faces = faces_numpy[best_component_face_ids]
    selected_vertex_ids, selected_faces = np.unique(
        selected_faces.reshape(-1),
        return_inverse=True,
    )
    selected_faces = selected_faces.reshape(-1, 3)
    selected_vertices = vertices[torch.from_numpy(selected_vertex_ids).to(device=vertices.device)]
    selected_faces = torch.from_numpy(selected_faces).to(device=vertices.device)

    vertex_copies = []
    face_copies = []
    for group_index, (transform, translation, sign) in enumerate(zip(transforms, translations, signs)):
        vertex_copies.append(selected_vertices @ transform.T + translation)
        copy_faces = selected_faces if sign.item() else selected_faces[:, [0, 2, 1]]
        face_copies.append(copy_faces + group_index * selected_vertices.shape[0])

    output_vertices = torch.cat(vertex_copies).cpu().numpy()
    output_faces = torch.cat(face_copies).cpu().numpy()
    return output_vertices, output_faces


def voxel_occupancy_majority(
    vertices,
    faces,
    transforms,
    translations,
    resolution,
    bbox_padding,
    slab_depth,
):
    bbox_min = vertices.amin(dim=0)
    bbox_max = vertices.amax(dim=0)
    grid_center = 0.5 * (bbox_min + bbox_max)
    grid_side = (bbox_max - bbox_min).max() * (1.0 + 2.0 * bbox_padding)
    grid_min = grid_center - 0.5 * grid_side
    voxel_size = grid_side / resolution

    bvh = cumesh.cuBVH(vertices.contiguous(), faces.int().contiguous())
    occupancy = torch.empty(resolution, resolution, resolution, device=vertices.device, dtype=torch.bool)
    grid_coordinates = torch.arange(resolution, device=vertices.device, dtype=vertices.dtype) + 0.5
    coordinates_x = grid_min[0] + grid_coordinates * voxel_size
    coordinates_y = grid_min[1] + grid_coordinates * voxel_size
    coordinates_z = grid_min[2] + grid_coordinates * voxel_size

    for slab_start in range(0, resolution, slab_depth):
        slab_end = min(slab_start + slab_depth, resolution)
        grid_x, grid_y, grid_z = torch.meshgrid(
            coordinates_x,
            coordinates_y,
            coordinates_z[slab_start:slab_end],
            indexing="ij",
        )
        query_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3).contiguous()
        signed_distance = bvh.signed_distance(
            query_points,
            mode="watertight",
        )[0]
        occupancy[:, :, slab_start:slab_end] = signed_distance.reshape(resolution, resolution, slab_end - slab_start) < 0.0

    symmetric_occupancy = torch.empty_like(occupancy)
    majority_threshold = math.ceil(transforms.shape[0] / 2)

    for slab_start in range(0, resolution, slab_depth):
        slab_end = min(slab_start + slab_depth, resolution)
        grid_x, grid_y, grid_z = torch.meshgrid(coordinates_x, coordinates_y, coordinates_z[slab_start:slab_end], indexing="ij")
        output_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3).contiguous()
        votes = torch.zeros(output_points.shape[0], device=vertices.device, dtype=torch.int16)

        for transform, translation in zip(transforms, translations):
            input_points = (output_points - translation) @ transform
            grid_coordinates = torch.round((input_points - grid_min) / voxel_size - 0.5).long()
            valid = (grid_coordinates[:, 0] >= 0) & (grid_coordinates[:, 0] < resolution) & (grid_coordinates[:, 1] >= 0) & (grid_coordinates[:, 1] < resolution) & (grid_coordinates[:, 2] >= 0) & (grid_coordinates[:, 2] < resolution)
            indices = grid_coordinates[valid]
            votes[valid] += occupancy[indices[:, 0], indices[:, 1], indices[:, 2]].to(torch.int16)

        symmetric_occupancy[:, :, slab_start:slab_end] = votes.reshape(resolution, resolution, slab_end - slab_start) >= majority_threshold

    volume = symmetric_occupancy.cpu().numpy().astype(np.float32)
    spacing = float(voxel_size.cpu())
    output_vertices, output_faces, _, _ = measure.marching_cubes(volume, level=0.5, spacing=(spacing, spacing, spacing), gradient_direction="ascent")
    output_vertices += (grid_min + 0.5 * voxel_size).cpu().numpy()[None]
    return output_vertices.astype(np.float32), output_faces.astype(np.int64)


def closest_point_orbit_average(
    vertices,
    faces,
    transforms,
    translations,
    iterations,
):
    averaged_vertices = vertices.clone()

    for _ in range(iterations):
        bvh = cumesh.cuBVH(
            averaged_vertices.contiguous(),
            faces.int().contiguous(),
        )
        vertex_sum = torch.zeros_like(averaged_vertices)

        for transform, translation in zip(transforms, translations):
            transformed_vertices = (averaged_vertices @ transform.T + translation).contiguous()
            _, face_ids, barycentric_coordinates = bvh.unsigned_distance(transformed_vertices, return_uvw=True)
            triangles = averaged_vertices[faces[face_ids.long()]]
            closest_points = (triangles * barycentric_coordinates.unsqueeze(-1)).sum(dim=1)
            vertex_sum += (closest_points - translation) @ transform

        averaged_vertices = vertex_sum / transforms.shape[0]

    return averaged_vertices.cpu().numpy(), faces.cpu().numpy()


@torch.no_grad()
def postprocess_mesh(
    idx,
    shape_id,
    view_id,
    symmetry_label,
    symmetry_fold,
    symmetry_center,
    major_axis,
    minor_axis,
    sector_phase_samples,
    voxel_resolution,
    bbox_padding,
    voxel_slab_depth,
    closest_point_iterations,
    input_files,
    sector_files,
    voxel_files,
    closest_files,
):
    mesh = trimesh.load(
        input_files.path(".glb", shape_id=shape_id, view_id=view_id),
        force="mesh",
        process=False,
    )
    vertices = np.asarray(mesh.vertices, dtype=np.float32).copy()
    faces = np.asarray(mesh.faces, dtype=np.int64).copy()
    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 1] *= -1

    device = torch.device("cuda:0")
    vertices = torch.from_numpy(vertices).to(device)
    faces = torch.from_numpy(faces).to(device)
    center = torch.tensor(symmetry_center, device=device, dtype=torch.float32)
    major_axis = torch.tensor(major_axis, device=device, dtype=torch.float32)
    minor_axis = torch.tensor(minor_axis, device=device, dtype=torch.float32)

    transforms, translations, _ = get_3d_point_group(
        label=symmetry_label,
        center=center,
        major_axis=major_axis,
        minor_axis=minor_axis,
        include_identity=True,
    )
    sector_label = "S1" if symmetry_label == "S1" else f"C{symmetry_fold}"
    sector_transforms, sector_translations, sector_signs = get_3d_point_group(
        label=sector_label,
        center=center,
        major_axis=major_axis,
        minor_axis=minor_axis,
        include_identity=True,
    )

    sector_mesh = sector_replication(
        vertices=vertices,
        faces=faces,
        center=center,
        major_axis=major_axis,
        transforms=sector_transforms,
        translations=sector_translations,
        signs=sector_signs,
        phase_samples=sector_phase_samples,
    )
    voxel_mesh = voxel_occupancy_majority(
        vertices=vertices,
        faces=faces,
        transforms=transforms,
        translations=translations,
        resolution=voxel_resolution,
        bbox_padding=bbox_padding,
        slab_depth=voxel_slab_depth,
    )
    closest_mesh = closest_point_orbit_average(
        vertices=vertices,
        faces=faces,
        transforms=transforms,
        translations=translations,
        iterations=closest_point_iterations,
    )

    outputs = [
        (
            sector_mesh,
            sector_files.path(".glb", shape_id=shape_id, view_id=view_id),
        ),
        (
            voxel_mesh,
            voxel_files.path(".glb", shape_id=shape_id, view_id=view_id),
        ),
        (
            closest_mesh,
            closest_files.path(".glb", shape_id=shape_id, view_id=view_id),
        ),
    ]
    for (output_vertices, output_faces), output_path in outputs:
        glb_vertices = output_vertices[:, [0, 2, 1]].copy()
        glb_vertices[:, 2] *= -1
        trimesh.Trimesh(vertices=glb_vertices, faces=output_faces, process=False).export(output_path)

    return {
        "idx": idx,
        sector_files.rel_path: True,
        voxel_files.rel_path: True,
        closest_files.rel_path: True,
    }


def main():
    parser = argparse.ArgumentParser(description="Apply direct mesh-space symmetry postprocessing baselines")
    parser.add_argument("--workspace-dir", required=True)
    parser.add_argument("--shape-folder", required=True)
    parser.add_argument("--symmetry-prediction-folder")
    parser.add_argument("--sector-phase-samples", type=int, default=64)
    parser.add_argument("--voxel-resolution", type=int, default=192)
    parser.add_argument("--bbox-padding", type=float, default=0.03)
    parser.add_argument("--voxel-slab-depth", type=int, default=4)
    parser.add_argument("--closest-point-iterations", type=int, default=10)
    parser.add_argument("--recompute-finished", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()

    workspace = Workspace(args.workspace_dir)
    shape_tag = args.shape_folder.strip("/").replace("/", "_")
    symmetry_tag = args.symmetry_prediction_folder.strip("/").replace("/", "_") if args.symmetry_prediction_folder else "gt"
    experiment_name = f"postprocess_{shape_tag}_symmetry_{symmetry_tag}"

    input_files = workspace.files(args.shape_folder)
    sector_files = workspace.files(f"experiments/{experiment_name}/sector_replication")
    voxel_files = workspace.files(f"experiments/{experiment_name}/voxel_majority")
    closest_files = workspace.files(f"experiments/{experiment_name}/closest_point_average")
    sector_files.mkdir()
    voxel_files.mkdir()
    closest_files.mkdir()
    output_files = (sector_files, voxel_files, closest_files)

    prediction_files = None
    if args.symmetry_prediction_folder:
        prediction_files = workspace.files(args.symmetry_prediction_folder)

    metadata = workspace.read_metadata().sort_values("idx")
    metadata = metadata.loc[[input_files.path(".glb", shape_id=row["shape_id"], view_id=row["view_id"]).is_file() for _, row in metadata.iterrows()]]
    if not args.recompute_finished:
        metadata = metadata.loc[[not all(files.path(".glb", shape_id=row["shape_id"], view_id=row["view_id"]).is_file() for files in output_files) for _, row in metadata.iterrows()]]

    start = len(metadata) * args.rank // args.world_size
    end = len(metadata) * (args.rank + 1) // args.world_size
    selected = metadata.iloc[start:end]
    if selected.empty:
        return

    inputs = []
    for _, row in selected.iterrows():
        shape_id = row["shape_id"]
        view_id = row["view_id"]
        shape_label = workspace.shape_labels[shape_id]

        if prediction_files is not None:
            with prediction_files.path(".json", shape_id=shape_id, view_id=view_id).open(encoding="utf-8") as file:
                prediction = json.load(file)
            symmetry_label = "S1" if shape_label["symmetry_group"] == "S1" else f"C{prediction['pred_fold']}"
            symmetry_fold = prediction["pred_fold"]
            symmetry_center = prediction["pred_center"]
            major_axis = prediction["pred_major_axis"]
            minor_axis = prediction["pred_minor_axis"]
        else:
            symmetry_label = shape_label["symmetry_group"]
            symmetry_fold = shape_label["symmetry_fold"]
            symmetry_center = [0.0, 0.0, 0.0]
            major_axis = AXIS_VECTORS[shape_label["major_axis"]]
            minor_axis = AXIS_VECTORS[shape_label["minor_axis"]]

        inputs.append(
            {
                "idx": row["idx"],
                "shape_id": shape_id,
                "view_id": view_id,
                "symmetry_label": symmetry_label,
                "symmetry_fold": symmetry_fold,
                "symmetry_center": symmetry_center,
                "major_axis": major_axis,
                "minor_axis": minor_axis,
            }
        )

    stages = [
        Stage(
            "postprocess meshes",
            postprocess_mesh,
            params={
                "sector_phase_samples": args.sector_phase_samples,
                "voxel_resolution": args.voxel_resolution,
                "bbox_padding": args.bbox_padding,
                "voxel_slab_depth": args.voxel_slab_depth,
                "closest_point_iterations": args.closest_point_iterations,
            },
            resources={
                "input_files": input_files,
                "sector_files": sector_files,
                "voxel_files": voxel_files,
                "closest_files": closest_files,
            },
        )
    ]
    results = Pipeline(stages, total=len(selected)).run(inputs)

    records = pd.DataFrame(
        results,
        columns=[
            "idx",
            sector_files.rel_path,
            voxel_files.rel_path,
            closest_files.rel_path,
        ],
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    record_path = workspace.path(f"unmerged_records/07_symmetry_postprocess_{timestamp}_rank{args.rank}.csv")
    records.to_csv(record_path, index=False)


if __name__ == "__main__":
    main()
