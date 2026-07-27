import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import trellis2.models as trellis2_model_registry
from trellis2.models.sc_vaes.fdg_vae import FlexiDualGridVaeDecoder
from trellis2.models.structured_latent_flow import ElasticSLatFlowModel
from trellis2.modules.sparse import SparseTensor

from inference.trellis2 import (
    TRELLIS2_SHAPE_LATENT_CFG_INTERVAL,
    TRELLIS2_SHAPE_LATENT_CFG_RESCALE,
    TRELLIS2_SHAPE_LATENT_CFG_STRENGTH,
    TRELLIS2_SHAPE_LATENT_RESCALE_T,
    TRELLIS2FlowPredictor,
    TRELLIS2ShapeLatentNoiseSampler,
    TRELLIS2ShapeLatentView,
    trelli2_mesh_to_glb,
    trellis2_shape_latent_to_sparse_view,
)
from preprocess.utils import Pipeline, Stage
from symtrellis.flow import (
    BaseFlowPredictor,
    BaseInitialNoiseSampler,
    ClassifierFreeGuidanceWrapper,
    EulerSolver,
    SymmetryProjectionGuidanceWrapper,
    SymmetryProjectionNoiseSampler,
)
from symtrellis.mapper import (
    BaseSpatialTransformLatentMapper,
    SymmetryProjector,
    concat_coeff,
    from_pretrained,
)
from symtrellis.symmetry import build_symmetry_relation_inputs, get_3d_point_group

from .base import Files, Workspace

SHAPE_FLOW_MODEL = "microsoft/TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16"
SHAPE_DECODER = "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16"

MAPPER_RELATION_CHUNK_SIZE = 4
MESH_RESOLUTION = 512
MESH_DECIMATION_TARGET = 100000
MESH_REMESH_PROJECT = 0.9

AXIS_VECTORS = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}


@torch.no_grad()
def generate_shape(
    idx: str,
    shape_id: int,
    view_id: int,
    symmetry_label: str,
    symmetry_center: list[float],
    major_axis: list[float],
    minor_axis: list[float],
    seed: int,
    steps: int,
    projection_required: bool,
    guidance_projection_required: bool,
    condition_files: Files,
    sparse_structure_files: Files,
    output_files: Files,
    flow_model: ElasticSLatFlowModel,
    decoder: FlexiDualGridVaeDecoder,
    mapper: BaseSpatialTransformLatentMapper,
    noise_sampler: BaseInitialNoiseSampler,
    predictor: BaseFlowPredictor,
    solver: EulerSolver,
) -> dict[str, object]:
    output_path = output_files.path(".glb", shape_id=shape_id, view_id=view_id)
    fail_path = output_files.path(".fail", shape_id=shape_id, view_id=view_id)
    sparse_structure_fail_path = sparse_structure_files.path(".fail", shape_id=shape_id, view_id=view_id)

    if sparse_structure_fail_path.is_file():
        if output_path.exists():
            output_path.unlink()
        fail_path.write_text(
            sparse_structure_fail_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        return {"idx": idx, output_files.rel_path: True}

    with np.load(condition_files.path(".npz", shape_id=shape_id, view_id=view_id)) as data:
        condition = torch.from_numpy(data["cond"])
    with np.load(sparse_structure_files.path(".npz", shape_id=shape_id, view_id=view_id)) as data:
        coords = torch.from_numpy(data["coords"])

    if coords.numel() == 0:
        fail_path.write_text("Empty coordinates!\n", encoding="utf-8")
        if output_path.exists():
            output_path.unlink()
        return {"idx": idx, output_files.rel_path: True}

    device = torch.device("cuda:0")
    condition = condition.to(device)
    coords = coords.to(device=device, dtype=torch.int32)
    negative_condition = torch.zeros_like(condition)
    noise_projection_arguments = {}
    guidance_projection_arguments = {}

    if projection_required:
        center = torch.tensor(symmetry_center, device=device, dtype=torch.float32)
        major = torch.tensor(major_axis, device=device, dtype=torch.float32)
        minor = torch.tensor(minor_axis, device=device, dtype=torch.float32)
        transforms, translations, signs = get_3d_point_group(
            label=symmetry_label,
            center=center,
            major_axis=major,
            minor_axis=minor,
            include_identity=False,
        )

        coefficient_chunks = []
        source_row_chunks = []
        destination_row_chunks = []
        mapper.to(device)

        for relation_start in range(0, transforms.shape[0], MAPPER_RELATION_CHUNK_SIZE):
            relation_end = relation_start + MAPPER_RELATION_CHUNK_SIZE
            relations = [
                (
                    transforms[relation_start:relation_end],
                    translations[relation_start:relation_end],
                    signs[relation_start:relation_end],
                )
                for _ in range(condition.shape[0])
            ]
            (
                source_coords,
                destination_coords,
                source_rows,
                destination_rows,
                relation_transforms,
                relation_translations,
                relation_signs,
            ) = build_symmetry_relation_inputs(
                coords=coords,
                relations=relations,
                grid_size=flow_model.resolution,
            )

            try:
                coefficients = mapper(
                    coords_src=source_coords,
                    coords_dst=destination_coords,
                    O_dst2src=relation_transforms,
                    t_dst2src=relation_translations,
                    s_dst2src=relation_signs,
                )
            except Exception as error:
                mapper.cpu()
                fail_path.write_text(f"Shape mapper failed: {error}\n", encoding="utf-8")
                if output_path.exists():
                    output_path.unlink()
                return {"idx": idx, output_files.rel_path: True}

            coefficient_chunks.append(coefficients.cpu())
            source_row_chunks.append(source_rows.cpu())
            destination_row_chunks.append(destination_rows.cpu())

        mapper.cpu()
        projector = SymmetryProjector(
            num_rows=coords.shape[0],
            rows_src=torch.cat(source_row_chunks).to(device),
            rows_dst=torch.cat(destination_row_chunks).to(device),
            coeff=concat_coeff(coefficient_chunks).to(device),
        )
        latent_view = TRELLIS2ShapeLatentView(coords=coords, sp_class=SparseTensor)
        noise_projection_arguments = {
            "projector": projector,
            "to_sparse_view": latent_view.to_sparse_view,
            "to_original_view": latent_view.to_original_view,
            "self_include": False,
        }
        guidance_projection_arguments = {
            "projector": projector,
            "to_sparse_view": latent_view.to_sparse_view,
            "to_original_view": latent_view.to_original_view,
            "self_include": True,
        }

    noise = noise_sampler.sample(
        sp_class=SparseTensor,
        coords=coords,
        feat_dim=flow_model.in_channels,
        grid_size=flow_model.resolution,
        seed=seed,
        device="cuda:0",
        **noise_projection_arguments,
    )
    predictor_arguments = {
        "cond": condition,
        "neg_cond": negative_condition,
    }
    if guidance_projection_required:
        predictor_arguments.update(guidance_projection_arguments)

    flow_model.to(device)
    shape_latent = solver.sample(
        noise=noise,
        predictor=predictor,
        steps=steps,
        predictor_args=predictor_arguments,
        rescale_t=TRELLIS2_SHAPE_LATENT_RESCALE_T,
        verbose=False,
    )[-1].x_t
    flow_model.cpu()

    shape_latent = shape_latent.replace(trellis2_shape_latent_to_sparse_view(shape_latent))

    try:
        decoder.to(device)
        shape_meshes, _ = decoder(shape_latent, return_subs=True)
        decoder.cpu()
        glb = trelli2_mesh_to_glb(
            shape_mesh=shape_meshes[0],
            res=MESH_RESOLUTION,
            device=device,
            remesh=True,
            decimation_target=MESH_DECIMATION_TARGET,
            remesh_project=MESH_REMESH_PROJECT,
        )
        glb.export(output_path)
    except Exception as error:
        decoder.cpu()
        fail_path.write_text(f"Shape decoder failed: {error}\n", encoding="utf-8")
        if output_path.exists():
            output_path.unlink()
        return {"idx": idx, output_files.rel_path: True}

    if fail_path.exists():
        fail_path.unlink()
    return {"idx": idx, output_files.rel_path: True}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TRELLIS.2 shapes")
    parser.add_argument("--workspace-dir", required=True)
    parser.add_argument("--condition-folder", default="conditions")
    parser.add_argument("--sparse-structure-folder", required=True)
    parser.add_argument("--mapper-path", required=True)
    parser.add_argument("--symmetry-prediction-folder")
    parser.add_argument("--noise-strength", type=float, default=0.5)
    parser.add_argument("--guidance-strength", type=float, default=1.0)
    parser.add_argument("--guidance-duration", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--recompute-finished", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()

    workspace = Workspace(args.workspace_dir)
    mapper_tag = args.mapper_path.strip("/").replace("/", "_")
    sparse_structure_tag = args.sparse_structure_folder.strip("/").replace("/", "_")
    symmetry_tag = args.symmetry_prediction_folder.strip("/").replace("/", "_") if args.symmetry_prediction_folder else "gt"
    experiment_name = "_".join(
        [
            f"shape_flow_seed_{args.seed}",
            f"steps_{args.steps}",
            f"noise_strength_{args.noise_strength}",
            f"guidance_strength_{args.guidance_strength}",
            f"guidance_duration_{args.guidance_duration}",
            f"mapper_{mapper_tag}",
            f"sparse_structure_{sparse_structure_tag}",
            f"symmetry_{symmetry_tag}",
        ]
    )

    condition_files = workspace.files(args.condition_folder)
    sparse_structure_files = workspace.files(args.sparse_structure_folder)
    output_files = workspace.files(f"experiments/{experiment_name}")
    output_files.mkdir()

    prediction_files = None
    if args.symmetry_prediction_folder:
        prediction_files = workspace.files(args.symmetry_prediction_folder)

    metadata = workspace.read_metadata().sort_values("idx")
    if output_files.rel_path not in metadata.columns:
        metadata[output_files.rel_path] = False

    if not args.recompute_finished:
        metadata = metadata.loc[~metadata[output_files.rel_path].eq(True)]

    start = len(metadata) * args.rank // args.world_size
    end = len(metadata) * (args.rank + 1) // args.world_size
    selected = metadata.iloc[start:end]
    if selected.empty:
        return

    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)

    flow_model = trellis2_model_registry.from_pretrained(SHAPE_FLOW_MODEL).eval()
    decoder = trellis2_model_registry.from_pretrained(SHAPE_DECODER).eval()
    decoder.set_resolution(MESH_RESOLUTION)
    mapper = from_pretrained(args.mapper_path, device="cpu").eval()

    guidance_projection_required = args.guidance_strength > 0.0 and args.guidance_duration > 0.0
    projection_required = args.noise_strength > 0.0 or guidance_projection_required

    noise_sampler = TRELLIS2ShapeLatentNoiseSampler()
    if args.noise_strength > 0.0:
        noise_sampler = SymmetryProjectionNoiseSampler(
            sampler=noise_sampler,
            symmetry_strength=args.noise_strength,
        )

    predictor = ClassifierFreeGuidanceWrapper(
        predictor=TRELLIS2FlowPredictor(model=flow_model),
        strength=TRELLIS2_SHAPE_LATENT_CFG_STRENGTH,
        interval=TRELLIS2_SHAPE_LATENT_CFG_INTERVAL,
        rescale=TRELLIS2_SHAPE_LATENT_CFG_RESCALE,
    )
    if guidance_projection_required:
        predictor = SymmetryProjectionGuidanceWrapper(
            predictor=predictor,
            strength=args.guidance_strength,
            interval=(0.0, args.guidance_duration),
            symmetrize_target="x_start",
            rescale=0.0,
        )

    solver = EulerSolver()
    inputs = []

    for _, row in selected.iterrows():
        shape_id = row["shape_id"]
        view_id = row["view_id"]
        shape_label = workspace.shape_labels[shape_id]

        if prediction_files is not None:
            with prediction_files.path(".json", shape_id=shape_id, view_id=view_id).open(encoding="utf-8") as file:
                prediction = json.load(file)
            symmetry_label = "S1" if shape_label["symmetry_group"] == "S1" else f"C{prediction['pred_fold']}"
            symmetry_center = prediction["pred_center"]
            major_axis = prediction["pred_major_axis"]
            minor_axis = prediction["pred_minor_axis"]
        else:
            symmetry_label = shape_label["symmetry_group"]
            symmetry_center = [0.0, 0.0, 0.0]
            major_axis = AXIS_VECTORS[shape_label["major_axis"]]
            minor_axis = AXIS_VECTORS[shape_label["minor_axis"]]

        inputs.append(
            {
                "idx": row["idx"],
                "shape_id": shape_id,
                "view_id": view_id,
                "symmetry_label": symmetry_label,
                "symmetry_center": symmetry_center,
                "major_axis": major_axis,
                "minor_axis": minor_axis,
            }
        )

    stages = [
        Stage(
            "generate shapes",
            generate_shape,
            params={
                "seed": args.seed,
                "steps": args.steps,
                "projection_required": projection_required,
                "guidance_projection_required": guidance_projection_required,
            },
            resources={
                "condition_files": condition_files,
                "sparse_structure_files": sparse_structure_files,
                "output_files": output_files,
                "flow_model": flow_model,
                "decoder": decoder,
                "mapper": mapper,
                "noise_sampler": noise_sampler,
                "predictor": predictor,
                "solver": solver,
            },
        )
    ]
    results = Pipeline(stages, total=len(selected)).run(inputs)

    records = pd.DataFrame(results, columns=["idx", output_files.rel_path])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    record_path = workspace.path(f"unmerged_records/03_generate_shape_{timestamp}_rank{args.rank}.csv")
    records.to_csv(record_path, index=False)

    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
    print(f"peak allocated: {peak_allocated:.1f} MiB")
    print(f"peak reserved:  {peak_reserved:.1f} MiB")


if __name__ == "__main__":
    main()
