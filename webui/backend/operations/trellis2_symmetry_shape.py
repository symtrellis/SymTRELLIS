from ..loaders.trellis2 import TRELLIS2Loader
from . import Operation
from .trellis2_vanilla_shape import SHAPE_LATENT, SHAPE_RAW_MESH, SHAPE_VISUALIZATION_MESH


class Trellis2SymmetryShape(Operation):
    operation_id = "trellis2.shape.symmetry"
    execution_kind = "node_run"
    queue_kind = "gpu"
    output_roles = (SHAPE_LATENT, SHAPE_RAW_MESH, SHAPE_VISUALIZATION_MESH)

    def __init__(self, loader: TRELLIS2Loader):
        self.loader = loader
