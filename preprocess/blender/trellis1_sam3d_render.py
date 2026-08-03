"""Render normalized meshes for TRELLIS.1 and SAM3D preprocessing."""

import argparse
import json
import math
import os
import sys
from typing import Callable, Dict, Tuple

import bpy
import numpy as np
from mathutils import Vector

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": (bpy.ops.import_scene.obj if bpy.app.version[0] < 4 else bpy.ops.wm.obj_import),
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl if bpy.app.version[0] < 4 else bpy.ops.wm.stl_import,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply if bpy.app.version[0] < 4 else bpy.ops.wm.ply_import,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}


def init_render(resolution: int = 1472) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.render.use_persistent_data = True

    scene.cycles.samples = 128
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 1
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 3
    scene.cycles.transmission_bounces = 3
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.denoising_quality = "HIGH"

    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1

    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "CUDA"
    preferences.get_devices()
    cuda_devices = [cycles_device for cycles_device in preferences.devices if cycles_device.type == "CUDA"]
    if not cuda_devices:
        scene.cycles.device = "CPU"
        scene.cycles.denoising_use_gpu = False
        return
    for cycles_device in preferences.devices:
        cycles_device.use = cycles_device.type == "CUDA"
    scene.cycles.device = "GPU"
    scene.cycles.denoising_use_gpu = True


def init_scene() -> None:
    """Reset the scene to a clean state."""
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def init_camera():
    cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.sensor_height = cam.data.sensor_width = 32

    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"

    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    bpy.context.scene.collection.objects.link(cam_empty)
    cam_constraint.target = cam_empty
    return cam


def init_lighting() -> Dict[str, bpy.types.Object]:
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    world = bpy.data.worlds.new("RenderWorld")
    world.use_nodes = True
    world_nodes = world.node_tree.nodes
    world_nodes.clear()
    background = world_nodes.new("ShaderNodeBackground")
    background.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    background.inputs["Strength"].default_value = 0.5
    world_output = world_nodes.new("ShaderNodeOutputWorld")
    world.node_tree.links.new(background.outputs["Background"], world_output.inputs["Surface"])
    bpy.context.scene.world = world

    light_positions = (
        ("PositiveX", (4, 0, 0)),
        ("NegativeX", (-4, 0, 0)),
        ("PositiveY", (0, 4, 0)),
        ("NegativeY", (0, -4, 0)),
        ("PositiveZ", (0, 0, 4)),
        ("NegativeZ", (0, 0, -4)),
    )
    lights = {}
    for name, position in light_positions:
        light_data = bpy.data.lights.new(name, type="AREA")
        light_data.color = (1.0, 1.0, 1.0)
        light_data.energy = 60
        light_data.shape = "DISK"
        light_data.size = 5

        light = bpy.data.objects.new(name, light_data)
        bpy.context.collection.objects.link(light)
        light.location = position
        light.rotation_euler = (-Vector(position)).to_track_quat("-Z", "Y").to_euler()
        lights[name] = light
    return lights


def load_object(object_path: str) -> None:
    """Load a supported model into the scene."""
    file_extension = object_path.rsplit(".", 1)[-1].lower()

    if file_extension == "usdz":
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        bpy.ops.preferences.addon_enable(module="io_scene_usdz")

        from io_scene_usdz.import_usdz import import_usdz

        import_usdz(bpy.context, filepath=object_path, materials=True, animations=True)
        return

    if file_extension not in IMPORT_FUNCTIONS:
        raise ValueError("unsupported file type: {}".format(object_path))

    import_function = IMPORT_FUNCTIONS[file_extension]
    print("Loading object from {}".format(object_path))
    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True, import_shading="NORMALS")
    else:
        import_function(filepath=object_path)


def delete_invisible_objects() -> None:
    """Delete objects and collections hidden in the loaded scene."""
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            obj.select_set(True)
    bpy.ops.object.delete()

    invisible_collections = [collection for collection in bpy.data.collections if collection.hide_viewport]
    for collection in invisible_collections:
        bpy.data.collections.remove(collection)


def unhide_all_objects() -> None:
    for obj in bpy.context.scene.objects:
        obj.hide_set(False)


def convert_to_meshes() -> None:
    bpy.ops.object.select_all(action="DESELECT")
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    bpy.context.view_layer.objects.active = meshes[0]
    for obj in bpy.context.scene.objects:
        obj.select_set(True)
    bpy.ops.object.convert(target="MESH")


def triangulate_meshes() -> None:
    bpy.ops.object.select_all(action="DESELECT")
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    bpy.context.view_layer.objects.active = meshes[0]
    for obj in meshes:
        obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.reveal()
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")


def scene_bbox() -> Tuple[Vector, Vector]:
    """Return the world-space bounding box of all scene meshes."""
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    scene_meshes = [obj for obj in bpy.context.scene.objects.values() if isinstance(obj.data, bpy.types.Mesh)]
    for obj in scene_meshes:
        found = True
        for coord in obj.bound_box:
            coord = obj.matrix_world @ Vector(coord)
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def normalize_scene() -> Tuple[float, Vector]:
    """Scale and translate the scene into a centered unit cube."""
    scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
    if len(scene_root_objects) > 1:
        scene = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(scene)
        for obj in scene_root_objects:
            obj.parent = scene
    else:
        scene = scene_root_objects[0]

    bbox_min, bbox_max = scene_bbox()
    center = (bbox_min + bbox_max) / 2
    extent = bbox_max - bbox_min
    max_extent = max(extent.x, extent.y, extent.z)
    if max_extent <= 0:
        raise RuntimeError("degenerate bounding box: max extent is not positive")
    scale = 1 / max_extent

    pivot = bpy.data.objects.new("NormalizePivot", None)
    bpy.context.scene.collection.objects.link(pivot)
    pivot.matrix_world.translation = center
    for obj in scene_root_objects:
        matrix_world = obj.matrix_world.copy()
        obj.parent = pivot
        obj.matrix_world = matrix_world
    bpy.context.view_layer.update()

    pivot.scale *= scale
    bpy.context.view_layer.update()

    offset = -center
    pivot.matrix_world.translation += offset
    bpy.context.view_layer.update()

    bpy.ops.object.select_all(action="DESELECT")
    return scale, offset


def get_transform_matrix(obj: bpy.types.Object) -> list:
    position, rotation, _ = obj.matrix_world.decompose()
    rotation = rotation.to_matrix()
    matrix = []
    for row_index in range(3):
        row = []
        for column_index in range(3):
            row.append(rotation[row_index][column_index])
        row.append(position[row_index])
        matrix.append(row)
    matrix.append([0, 0, 0, 1])
    return matrix


def main(arg) -> None:
    os.makedirs(arg.output_folder, exist_ok=True)

    init_render(resolution=arg.resolution)
    if arg.object.endswith(".blend"):
        delete_invisible_objects()
    else:
        init_scene()
        load_object(arg.object)
    print("[INFO] Scene initialized.")

    scale, offset = normalize_scene()
    print("[INFO] Scene normalized.")

    camera = init_camera()
    init_lighting()
    print("[INFO] Camera and lighting initialized.")

    transforms = {
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "scale": scale,
        "offset": [offset.x, offset.y, offset.z],
        "frames": [],
    }

    views = json.loads(arg.views)
    for index, view in enumerate(views):
        camera.location = (
            view["radius"] * np.cos(view["yaw"]) * np.cos(view["pitch"]),
            view["radius"] * np.sin(view["yaw"]) * np.cos(view["pitch"]),
            view["radius"] * np.sin(view["pitch"]),
        )
        camera.data.lens = 16 / np.tan(view["fov"] / 2)

        file_name = "{:03d}.png".format(index)
        bpy.context.scene.render.filepath = os.path.join(arg.output_folder, file_name)
        bpy.ops.render.render(write_still=True)
        bpy.context.view_layer.update()

        transforms["frames"].append(
            {
                "file_path": file_name,
                "camera_angle_x": view["fov"],
                "transform_matrix": get_transform_matrix(camera),
            }
        )

    with open(os.path.join(arg.output_folder, "transforms.json"), "w") as file:
        json.dump(transforms, file, indent=4)

    unhide_all_objects()
    convert_to_meshes()
    triangulate_meshes()
    print("[INFO] Meshes triangulated.")

    mesh_path = os.path.join(arg.output_folder, "mesh.ply")
    if bpy.app.version < (4, 0, 0):
        bpy.ops.export_mesh.ply(filepath=mesh_path)
    elif bpy.app.version < (4, 1, 0):
        bpy.ops.wm.ply_export(filepath=mesh_path)
    else:
        bpy.ops.wm.ply_export(filepath=mesh_path, export_attributes=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a normalized object for TRELLIS.1 and SAM3D preprocessing.")
    parser.add_argument(
        "--views",
        type=str,
        required=True,
        help="JSON list of yaw, pitch, radius, and fov values.",
    )
    parser.add_argument("--object", type=str, required=True, help="Path to the 3D model.")
    parser.add_argument(
        "--output-folder",
        type=str,
        default="/tmp",
        help="Directory for images, transforms, and mesh.",
    )
    parser.add_argument("--resolution", type=int, default=1472, help="Square render resolution.")
    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)
    main(args)
