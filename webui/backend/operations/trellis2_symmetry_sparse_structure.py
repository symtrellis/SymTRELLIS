from ..loaders.trellis2 import TRELLIS2Loader
from . import Operation
from .trellis2_vanilla_sparse_structure import OCC_COORDINATES, OCC_VISUALIZATION_MESH, SPARSE_STRUCTURE_LATENT


class Trellis2SymmetrySparseStructure(Operation):
    operation_id = "trellis2.sparse_structure.symmetry"
    execution_kind = "node_run"
    queue_kind = "gpu"
    output_roles = (SPARSE_STRUCTURE_LATENT, OCC_VISUALIZATION_MESH, OCC_COORDINATES)

    def __init__(self, loader: TRELLIS2Loader):
        self.loader = loader
