import type { ModelSpec } from './types';
import type {
  CommonGenerationParams,
  DurationRange,
  GenerationAction,
  GenerationState,
  SymmetryProjectionParams,
} from '../state/generation';

export type Trellis2ShapeMode = '512' | 'cascade';

export type Trellis2ShapeParams = {
  maxTokens: number;
  mode: Trellis2ShapeMode;
};

export type Trellis2VanillaSparseStructureParams = CommonGenerationParams;

export type Trellis2SymmetrySparseStructureParams =
  CommonGenerationParams & SymmetryProjectionParams;

export type Trellis2VanillaShapeParams = CommonGenerationParams & Trellis2ShapeParams;

export type Trellis2SymmetryShapeParams =
  CommonGenerationParams & SymmetryProjectionParams & Trellis2ShapeParams;

export type Trellis2TextureParams = CommonGenerationParams;

export type Trellis2ExportParams = {
  faceDecimationTarget: number;
  remesh: boolean;
  remeshBand: number;
  remeshProject: number;
  textureSize: number;
};

export type Trellis2SparseMetadata = {
  voxelCount: number;
};

export type Trellis2ShapeMetadata = {
  oVoxelGridSize: number;
  shapeLatentGridSize: number;
  voxelCount: number;
};

export type Trellis2TextureMetadata = Record<string, never>;

export type Trellis2VanillaSparseStructureState = GenerationState<
  Trellis2VanillaSparseStructureParams,
  Trellis2SparseMetadata
>;

export type Trellis2SymmetrySparseStructureState = GenerationState<
  Trellis2SymmetrySparseStructureParams,
  Trellis2SparseMetadata
>;

export type Trellis2VanillaShapeState = GenerationState<
  Trellis2VanillaShapeParams,
  Trellis2ShapeMetadata
>;

export type Trellis2SymmetryShapeState = GenerationState<
  Trellis2SymmetryShapeParams,
  Trellis2ShapeMetadata
>;

export type Trellis2TextureState = GenerationState<Trellis2TextureParams, Trellis2TextureMetadata>;

export type Trellis2VanillaSparseStructureAction = GenerationAction<
  Trellis2VanillaSparseStructureParams,
  Trellis2SparseMetadata
>;

export type Trellis2SymmetrySparseStructureAction = GenerationAction<
  Trellis2SymmetrySparseStructureParams,
  Trellis2SparseMetadata
>;

export type Trellis2VanillaShapeAction = GenerationAction<
  Trellis2VanillaShapeParams,
  Trellis2ShapeMetadata
>;

export type Trellis2SymmetryShapeAction = GenerationAction<
  Trellis2SymmetryShapeParams,
  Trellis2ShapeMetadata
>;

export type Trellis2TextureAction = GenerationAction<Trellis2TextureParams, Trellis2TextureMetadata>;

export const trellis2OutputRoleCandidates = {
  exportBundle: ['export_bundle', 'latent_bundle', 'latents_zip', 'zip'],
  exportGlb: ['glb', 'export', 'download'],
  shapeMesh: ['mesh', 'shape', 'glb', 'preview'],
  sparseStructureMesh: ['occ_visualization_mesh', 'occ', 'occupancy', 'glb', 'preview'],
  texturedMesh: ['texture', 'textured_mesh', 'mesh', 'glb'],
};

const cfgDuration: DurationRange = [0, 0.4];
const symmetryProjectionDuration: DurationRange = [0, 0.3];

const commonGenerationDefaults: CommonGenerationParams = {
  cfgDuration,
  cfgRescale: 0.7,
  cfgStrength: 7.5,
  seed: 42,
  steps: 12,
  timeStepRescale: 0.5,
};

const symmetryProjectionDefaults: SymmetryProjectionParams = {
  noiseSymmetryProjectionStrength: 0.2,
  symmetryProjectionDuration,
  symmetryProjectionStrength: 0.9,
};

const shapeDefaults: Trellis2ShapeParams = {
  maxTokens: 32768,
  mode: '512',
};

export const trellis2ExportDefaults: Trellis2ExportParams = {
  faceDecimationTarget: 1000000,
  remesh: false,
  remeshBand: 1,
  remeshProject: 0,
  textureSize: 4096,
};

export const trellis2GenerationDefaults = {
  symmetryShape: {
    ...commonGenerationDefaults,
    ...symmetryProjectionDefaults,
    ...shapeDefaults,
    cfgRescale: 0.5,
    noiseSymmetryProjectionStrength: 0.2,
    timeStepRescale: 3,
  } satisfies Trellis2SymmetryShapeParams,
  symmetrySparseStructure: {
    ...commonGenerationDefaults,
    ...symmetryProjectionDefaults,
    noiseSymmetryProjectionStrength: 0.4,
  } satisfies Trellis2SymmetrySparseStructureParams,
  texture: {
    ...commonGenerationDefaults,
    cfgDuration: [0.1, 0.4],
    cfgRescale: 0,
    cfgStrength: 1,
    timeStepRescale: 3,
  } satisfies Trellis2TextureParams,
  vanillaShape: {
    ...commonGenerationDefaults,
    ...shapeDefaults,
    cfgRescale: 0.5,
    timeStepRescale: 3,
  } satisfies Trellis2VanillaShapeParams,
  vanillaSparseStructure: {
    ...commonGenerationDefaults,
  } satisfies Trellis2VanillaSparseStructureParams,
};

export const trellis2InitialSparseMetadata: Trellis2SparseMetadata = {
  voxelCount: 0,
};

export const trellis2InitialShapeMetadata: Trellis2ShapeMetadata = {
  oVoxelGridSize: 0,
  shapeLatentGridSize: 0,
  voxelCount: 0,
};

export const trellis2InitialTextureMetadata: Trellis2TextureMetadata = {};

export function estimateTrellis2Bf16FlowPeakGb(maxTokens: number): number {
  return 2.691 + 0.00004342 * maxTokens;
}

export function trellis2SparseMetadata(metadata: Record<string, unknown>): Trellis2SparseMetadata {
  return {
    voxelCount: Number(metadata.voxelCount ?? 0),
  };
}

export function trellis2ShapeMetadata(metadata: Record<string, unknown>): Trellis2ShapeMetadata {
  return {
    oVoxelGridSize: Number(metadata.oVoxelGridSize ?? 0),
    shapeLatentGridSize: Number(metadata.shapeLatentGridSize ?? 0),
    voxelCount: Number(metadata.voxelCount ?? 0),
  };
}

export function trellis2TextureMetadata(): Trellis2TextureMetadata {
  return {};
}

export const trellis2ModelSpec: ModelSpec = {
  disabled: false,
  id: 'trellis2',
  label: 'TRELLIS.2',
  dag: {
    entryNodeId: 'image_condition',
    nodes: [
      {
        id: 'image_condition',
        kind: 'trellis2_image_condition',
        label: 'Image condition',
        operation: 'trellis2.image_condition',
        shortLabel: 'IMG COND',
      },
      {
        id: 'vanilla_sparse_structure',
        kind: 'trellis2_vanilla_sparse_structure',
        label: 'Vanilla sparse structure generation',
        operation: 'trellis2.sparse_structure.vanilla',
        shortLabel: 'VANILLA SS',
      },
      {
        id: 'vanilla_shape',
        kind: 'trellis2_vanilla_shape',
        label: 'Generate shape with vanilla model',
        operation: 'trellis2.shape.vanilla',
        shortLabel: 'VANILLA SHAPE',
      },
      {
        id: 'detect_adjust_symmetry',
        kind: 'detect_adjust_symmetry',
        label: 'Detect and adjust symmetry',
        operation: 'symmetry.confirm_detected_tuple',
        shortLabel: 'DETECT SYM',
      },
      {
        id: 'manual_symmetry',
        kind: 'manual_symmetry',
        label: 'Manually specify symmetry',
        operation: 'symmetry.confirm_manual_tuple',
        shortLabel: 'MANUAL SYM',
      },
      {
        id: 'symmetry_sparse_structure',
        kind: 'trellis2_symmetry_sparse_structure',
        label: 'Symmetry enforced sparse structure generation',
        operation: 'trellis2.sparse_structure.symmetry',
        shortLabel: 'SYM SS',
      },
      {
        id: 'symmetry_shape',
        kind: 'trellis2_symmetry_shape',
        label: 'Symmetry enforced shape generation',
        operation: 'trellis2.shape.symmetry',
        shortLabel: 'SYM SHAPE',
      },
      {
        id: 'texture',
        kind: 'trellis2_texture',
        label: 'Generate texture',
        operation: 'trellis2.texture.generate',
        shortLabel: 'TEXTURE',
      },
    ],
    edges: [
      {
        id: 'image_condition-manual_symmetry',
        routeLabel: 'Manually specify symmetry',
        source: 'image_condition',
        target: 'manual_symmetry',
      },
      {
        id: 'image_condition-vanilla_sparse_structure',
        routeLabel: 'Generate then detect',
        source: 'image_condition',
        target: 'vanilla_sparse_structure',
      },
      {
        id: 'vanilla_sparse_structure-vanilla_shape',
        source: 'vanilla_sparse_structure',
        target: 'vanilla_shape',
      },
      {
        id: 'vanilla_shape-detect_adjust_symmetry',
        source: 'vanilla_shape',
        target: 'detect_adjust_symmetry',
      },
      { id: 'vanilla_shape-texture', source: 'vanilla_shape', target: 'texture' },
      {
        id: 'detect_adjust_symmetry-symmetry_sparse_structure',
        source: 'detect_adjust_symmetry',
        target: 'symmetry_sparse_structure',
      },
      {
        id: 'manual_symmetry-symmetry_sparse_structure',
        source: 'manual_symmetry',
        target: 'symmetry_sparse_structure',
      },
      {
        id: 'symmetry_sparse_structure-symmetry_shape',
        routeLabel: 'Generate symmetry enforced shape',
        source: 'symmetry_sparse_structure',
        target: 'symmetry_shape',
      },
      { id: 'symmetry_shape-texture', source: 'symmetry_shape', target: 'texture' },
    ],
    layout: {
      nodes: {
        image_condition: { lane: 'main', rank: 0 },
        vanilla_sparse_structure: { lane: 'main', rank: 1 },
        vanilla_shape: { lane: 'main', rank: 2 },
        manual_symmetry: { lane: 'left', rank: 3 },
        detect_adjust_symmetry: { lane: 'main', rank: 3 },
        symmetry_sparse_structure: { lane: 'main', rank: 4 },
        symmetry_shape: { lane: 'main', rank: 5 },
        texture: { lane: 'main', rank: 6 },
      },
      edges: {
        'image_condition-manual_symmetry': { route: 'side_branch' },
        'image_condition-vanilla_sparse_structure': { route: 'straight' },
        'vanilla_sparse_structure-vanilla_shape': { route: 'straight' },
        'vanilla_shape-detect_adjust_symmetry': { route: 'straight' },
        'vanilla_shape-texture': { route: 'right_bypass' },
        'detect_adjust_symmetry-symmetry_sparse_structure': { route: 'straight' },
        'manual_symmetry-symmetry_sparse_structure': { route: 'side_merge' },
        'symmetry_sparse_structure-symmetry_shape': { route: 'straight' },
        'symmetry_shape-texture': { route: 'straight' },
      },
    },
  },
  viewer: {
    detect_adjust_symmetry: {
      outputCandidates: [
        {
          material: 'neutral',
          nodeId: 'vanilla_shape',
          roles: trellis2OutputRoleCandidates.shapeMesh,
        },
      ],
    },
    manual_symmetry: {
      outputCandidates: [],
    },
    symmetry_shape: {
      outputCandidates: [
        {
          material: 'neutral',
          nodeId: 'symmetry_shape',
          roles: trellis2OutputRoleCandidates.shapeMesh,
        },
        {
          material: 'neutral',
          nodeId: 'symmetry_sparse_structure',
          roles: trellis2OutputRoleCandidates.sparseStructureMesh,
        },
      ],
    },
    symmetry_sparse_structure: {
      outputCandidates: [
        {
          material: 'neutral',
          nodeId: 'symmetry_sparse_structure',
          roles: trellis2OutputRoleCandidates.sparseStructureMesh,
        },
      ],
      showConfirmedSymmetryPreview: true,
    },
    texture: {
      outputCandidates: [
        {
          material: 'source',
          nodeId: 'texture',
          roles: trellis2OutputRoleCandidates.texturedMesh,
        },
        {
          material: 'neutral',
          nodeId: 'symmetry_shape',
          roles: trellis2OutputRoleCandidates.shapeMesh,
        },
        {
          material: 'neutral',
          nodeId: 'vanilla_shape',
          roles: trellis2OutputRoleCandidates.shapeMesh,
        },
      ],
    },
    vanilla_shape: {
      outputCandidates: [
        {
          material: 'neutral',
          nodeId: 'vanilla_shape',
          roles: trellis2OutputRoleCandidates.shapeMesh,
        },
        {
          material: 'neutral',
          nodeId: 'vanilla_sparse_structure',
          roles: trellis2OutputRoleCandidates.sparseStructureMesh,
        },
      ],
    },
    vanilla_sparse_structure: {
      outputCandidates: [
        {
          material: 'neutral',
          nodeId: 'vanilla_sparse_structure',
          roles: trellis2OutputRoleCandidates.sparseStructureMesh,
        },
      ],
    },
  },
};
