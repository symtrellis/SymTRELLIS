import type {
  DagStatus,
  FinerSymmetryResult,
  NodeId,
  ReflectionPlaneCandidate,
  RotationAxisCandidate,
  SymmetryFamily,
  SymmetryOverlay,
  SymmetryTuple,
  ThemeMode,
  Vector3,
} from './types';

export const themeStorageKey = 'symtrellis.theme';
const secondaryAxisParallelThreshold = 0.98;

const inactiveDagStatus: Record<NodeId, DagStatus> = {
  img_cond: 'inactive',
  manual_sym: 'inactive',
  nat_ss: 'inactive',
  nat_shape: 'inactive',
  detect_sym: 'inactive',
  sym_ss: 'inactive',
  sym_shape: 'inactive',
  texture: 'inactive',
};

export function readStoredTheme(): ThemeMode {
  const storedTheme = window.localStorage.getItem(themeStorageKey);

  if (storedTheme === 'dark' || storedTheme === 'light') {
    return storedTheme;
  }

  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export function writeStoredTheme(theme: ThemeMode) {
  window.localStorage.setItem(themeStorageKey, theme);
}

export type DetectionStatus = 'idle' | 'running' | 'ready' | 'empty';

export type ImageConditionStatus = 'idle' | 'ready';

export type ProposedSymmetry = SymmetryTuple;

export type DurationRange = [number, number];

export type SymShapeMode = '512' | 'cascade';

export type ImageConditionState = {
  conditionStatus: ImageConditionStatus;
  uploadedImageFile: Blob | File | null;
  uploadedImageName: string;
  uploadedImageUrl: string;
};

export type ManualSymmetryState = {
  center: Vector3;
  family: SymmetryFamily;
  fold: number;
  labels: string[];
  majorAxis: Vector3;
  minorAxis: Vector3;
  proposedSymmetry: ProposedSymmetry | null;
  selectedLabel: string;
  symmetryPreview: SymmetryTuple | null;
};

export type SymSparseStructureState = {
  cfgDuration: DurationRange;
  cfgRescale: number;
  cfgStrength: number;
  generatedOccUrl: string;
  noiseSymmetryProjectionStrength: number;
  progress: number;
  seed: number;
  status: 'idle' | 'running' | 'ready';
  steps: number;
  symmetryProjectionDuration: DurationRange;
  symmetryProjectionStrength: number;
  timeStepRescale: number;
  voxelCount: number;
};

export type VanillaSparseStructureState = {
  cfgDuration: DurationRange;
  cfgRescale: number;
  cfgStrength: number;
  generatedOccUrl: string;
  progress: number;
  seed: number;
  status: 'idle' | 'running' | 'ready';
  steps: number;
  timeStepRescale: number;
  voxelCount: number;
};

export type SymShapeState = {
  cfgDuration: DurationRange;
  cfgRescale: number;
  cfgStrength: number;
  generatedShapeUrl: string;
  inputOccUrl: string;
  maxTokens: number;
  mode: SymShapeMode;
  noiseSymmetryProjectionStrength: number;
  oVoxelGridSize: number;
  progress: number;
  seed: number;
  shapeLatentGridSize: number;
  status: 'idle' | 'running' | 'ready';
  steps: number;
  symmetryProjectionDuration: DurationRange;
  symmetryProjectionStrength: number;
  timeStepRescale: number;
  voxelCount: number;
};

export type VanillaShapeState = {
  cfgDuration: DurationRange;
  cfgRescale: number;
  cfgStrength: number;
  generatedShapeUrl: string;
  inputOccUrl: string;
  maxTokens: number;
  mode: SymShapeMode;
  oVoxelGridSize: number;
  progress: number;
  seed: number;
  shapeLatentGridSize: number;
  status: 'idle' | 'running' | 'ready';
  steps: number;
  timeStepRescale: number;
  voxelCount: number;
};

export type TextureState = {
  cfgDuration: DurationRange;
  cfgRescale: number;
  cfgStrength: number;
  generatedTextureUrl: string;
  inputShapeUrl: string;
  progress: number;
  seed: number;
  status: 'idle' | 'running' | 'ready';
  steps: number;
  timeStepRescale: number;
};

export type DetectionState = {
  c2Axes: FinerSymmetryResult['c2Axes'];
  center: Vector3;
  family: SymmetryFamily | null;
  finerStatus: DetectionStatus;
  fold: number;
  labels: string[];
  majorAxis: Vector3;
  majorStatus: DetectionStatus;
  minorAxis: Vector3;
  overlays: SymmetryOverlay[];
  proposedSymmetry: ProposedSymmetry | null;
  reflectionPlanesContainingAxis: ReflectionPlaneCandidate[];
  reflectionPlanesPerpendicularToAxis: ReflectionPlaneCandidate[];
  rotationAxes: RotationAxisCandidate[];
  selectedLabel: string;
  selectedMajorCandidateId: string | null;
  selectedMinorItemId: string | null;
  selectedOverlayId: string | null;
  selectableOverlayIds: string[];
  symmetryPreview: SymmetryTuple | null;
};

export type DetectionAction =
  | { type: 'majorDetectionStarted' }
  | { candidates: RotationAxisCandidate[]; type: 'rotationAxesLoaded' }
  | { candidateId: string; type: 'majorCandidatePicked' }
  | { axis: Vector3; type: 'majorAxisChanged' }
  | { center: Vector3; type: 'centerChanged' }
  | { fold: number; type: 'foldChanged' }
  | { type: 'majorAxisNormalized' }
  | { type: 'centerNormalized' }
  | { family: SymmetryFamily; type: 'familyPicked' }
  | { type: 'finerDetectionStarted' }
  | { result: FinerSymmetryResult; type: 'finerResultLoaded' }
  | { label: string; type: 'labelPicked' }
  | { axis: Vector3; type: 'minorAxisChanged' }
  | { itemId: string; type: 'minorItemPicked' }
  | { overlayId: string; type: 'overlayPicked' }
  | { type: 'minorAxisNormalized' }
  | { type: 'proposeSymmetry' }
  | { type: 'confirmSymmetry' };

export type ImageConditionAction =
  | { file: Blob | File; name: string; type: 'imageUploaded'; url: string }
  | { type: 'conditionGenerated' };

export type ManualSymmetryAction =
  | { axis: Vector3; type: 'majorAxisChanged' }
  | { axis: Vector3; type: 'minorAxisChanged' }
  | { center: Vector3; type: 'centerChanged' }
  | { axis: Vector3; type: 'majorAxisShortcutPicked' }
  | { axis: Vector3; type: 'minorAxisShortcutPicked' }
  | { family: SymmetryFamily; type: 'familyPicked' }
  | { fold: number; type: 'foldChanged' }
  | { label: string; type: 'labelPicked' }
  | { type: 'proposeSymmetry' }
  | { type: 'confirmSymmetry' };

export type SymSparseStructureAction =
  | {
      params: Partial<
        Omit<SymSparseStructureState, 'generatedOccUrl' | 'progress' | 'status' | 'voxelCount'>
      >;
      type: 'paramsChanged';
    }
  | { type: 'seedRandomized' }
  | { type: 'generationStarted' }
  | { progress: number; type: 'generationProgressed' }
  | { generatedOccUrl: string; voxelCount: number; type: 'generationFinished' };

export type VanillaSparseStructureAction =
  | {
      params: Partial<
        Omit<
          VanillaSparseStructureState,
          'generatedOccUrl' | 'progress' | 'status' | 'voxelCount'
        >
      >;
      type: 'paramsChanged';
    }
  | { type: 'seedRandomized' }
  | { type: 'generationStarted' }
  | { progress: number; type: 'generationProgressed' }
  | { generatedOccUrl: string; voxelCount: number; type: 'generationFinished' };

export type SymShapeAction =
  | {
      params: Partial<
        Omit<
          SymShapeState,
          | 'generatedShapeUrl'
          | 'inputOccUrl'
          | 'oVoxelGridSize'
          | 'progress'
          | 'shapeLatentGridSize'
          | 'status'
          | 'voxelCount'
        >
      >;
      type: 'paramsChanged';
    }
  | { type: 'seedRandomized' }
  | { inputOccUrl: string; type: 'generationStarted' }
  | { progress: number; type: 'generationProgressed' }
  | {
      generatedShapeUrl: string;
      oVoxelGridSize: number;
      shapeLatentGridSize: number;
      type: 'generationFinished';
      voxelCount: number;
    };

export type VanillaShapeAction =
  | {
      params: Partial<
        Omit<
          VanillaShapeState,
          | 'generatedShapeUrl'
          | 'inputOccUrl'
          | 'oVoxelGridSize'
          | 'progress'
          | 'shapeLatentGridSize'
          | 'status'
          | 'voxelCount'
        >
      >;
      type: 'paramsChanged';
    }
  | { type: 'seedRandomized' }
  | { inputOccUrl: string; type: 'generationStarted' }
  | { progress: number; type: 'generationProgressed' }
  | {
      generatedShapeUrl: string;
      oVoxelGridSize: number;
      shapeLatentGridSize: number;
      type: 'generationFinished';
      voxelCount: number;
    };

export type TextureAction =
  | {
      params: Partial<
        Omit<TextureState, 'generatedTextureUrl' | 'inputShapeUrl' | 'progress' | 'status'>
      >;
      type: 'paramsChanged';
    }
  | { type: 'seedRandomized' }
  | { inputShapeUrl: string; type: 'generationStarted' }
  | { progress: number; type: 'generationProgressed' }
  | { generatedTextureUrl: string; type: 'generationFinished' };

export const initialImageConditionState: ImageConditionState = {
  conditionStatus: 'idle',
  uploadedImageFile: null,
  uploadedImageName: '',
  uploadedImageUrl: '',
};

export const initialManualSymmetryState: ManualSymmetryState = {
  center: [0, 0, 0],
  family: 'axial',
  fold: 2,
  labels: labelsForFamily('axial', 2),
  majorAxis: [0, 0, 1],
  minorAxis: [1, 0, 0],
  proposedSymmetry: null,
  selectedLabel: 'C2',
  symmetryPreview: null,
};

export const initialSymSparseStructureState: SymSparseStructureState = {
  cfgDuration: [0, 0.4],
  cfgRescale: 0.7,
  cfgStrength: 7.5,
  generatedOccUrl: '',
  noiseSymmetryProjectionStrength: 0.4,
  progress: 0,
  seed: 42,
  status: 'idle',
  steps: 12,
  symmetryProjectionDuration: [0, 0.3],
  symmetryProjectionStrength: 0.9,
  timeStepRescale: 0.5,
  voxelCount: 0,
};

export const initialVanillaSparseStructureState: VanillaSparseStructureState = {
  cfgDuration: [0, 0.4],
  cfgRescale: 0.7,
  cfgStrength: 7.5,
  generatedOccUrl: '',
  progress: 0,
  seed: 42,
  status: 'idle',
  steps: 12,
  timeStepRescale: 0.5,
  voxelCount: 0,
};

export const initialSymShapeState: SymShapeState = {
  cfgDuration: [0, 0.4],
  cfgRescale: 0.5,
  cfgStrength: 7.5,
  generatedShapeUrl: '',
  // MOCK_SYM_SHAPE_OCC_INPUT_START
  // public/mock/occ.glb stands in for the sym_ss output artifact when opening sym_shape directly.
  inputOccUrl: '/mock/occ.glb',
  // MOCK_SYM_SHAPE_OCC_INPUT_END
  maxTokens: 32768,
  mode: '512',
  noiseSymmetryProjectionStrength: 0.2,
  oVoxelGridSize: 0,
  progress: 0,
  seed: 42,
  shapeLatentGridSize: 0,
  status: 'idle',
  steps: 12,
  symmetryProjectionDuration: [0, 0.3],
  symmetryProjectionStrength: 0.9,
  timeStepRescale: 3,
  voxelCount: 0,
};

export const initialVanillaShapeState: VanillaShapeState = {
  cfgDuration: [0, 0.4],
  cfgRescale: 0.5,
  cfgStrength: 7.5,
  generatedShapeUrl: '',
  // MOCK_VANILLA_SHAPE_OCC_INPUT_START
  // public/mock/occ.glb stands in for the vanilla nat_ss output artifact when opening nat_shape directly.
  inputOccUrl: '/mock/occ.glb',
  // MOCK_VANILLA_SHAPE_OCC_INPUT_END
  maxTokens: 32768,
  mode: '512',
  oVoxelGridSize: 0,
  progress: 0,
  seed: 42,
  shapeLatentGridSize: 0,
  status: 'idle',
  steps: 12,
  timeStepRescale: 3,
  voxelCount: 0,
};

export const initialTextureState: TextureState = {
  cfgDuration: [0.1, 0.4],
  cfgRescale: 0,
  cfgStrength: 1,
  generatedTextureUrl: '',
  // MOCK_TEXTURE_SHAPE_INPUT_START
  // public/mock/shape.glb stands in for the shape-generation mesh artifact consumed by texture.
  inputShapeUrl: '/mock/shape.glb',
  // MOCK_TEXTURE_SHAPE_INPUT_END
  progress: 0,
  seed: 42,
  status: 'idle',
  steps: 12,
  timeStepRescale: 3,
};

export function dagStatusForCurrentNode(currentNodeId: NodeId): Record<NodeId, DagStatus> {
  const status = { ...inactiveDagStatus, [currentNodeId]: 'current' as DagStatus };

  if (currentNodeId === 'manual_sym') {
    status.img_cond = 'completed';
  } else if (currentNodeId === 'nat_ss') {
    status.img_cond = 'completed';
  } else if (currentNodeId === 'nat_shape') {
    status.img_cond = 'completed';
    status.nat_ss = 'completed';
  } else if (currentNodeId === 'detect_sym') {
    status.img_cond = 'completed';
    status.nat_ss = 'completed';
    status.nat_shape = 'completed';
  } else if (currentNodeId === 'sym_ss') {
    // TODO: sym_ss predecessor must come from session route state.
    // It can be either manual_sym or detect_sym; current mock defaults to manual_sym.
    status.img_cond = 'completed';
    status.manual_sym = 'completed';
  } else if (currentNodeId === 'sym_shape') {
    status.img_cond = 'completed';
    status.manual_sym = 'completed';
    status.sym_ss = 'completed';
  } else if (currentNodeId === 'texture') {
    // TODO: texture predecessor must come from session route state.
    // It can follow either native shape or symmetry-enforced shape; current mock follows sym_shape.
    status.img_cond = 'completed';
    status.manual_sym = 'completed';
    status.sym_ss = 'completed';
    status.sym_shape = 'completed';
  }

  return status;
}

export function imageConditionInstruction(state: ImageConditionState): string {
  if (!state.uploadedImageName) {
    return 'Upload an input image, then generate the TRELLIS.2 image condition.';
  }

  if (state.conditionStatus === 'ready') {
    return 'Condition is ready. Choose the next DAG node.';
  }

  return 'Generate the image condition before choosing the next DAG node.';
}

export function imageConditionReducer(
  state: ImageConditionState,
  action: ImageConditionAction,
): ImageConditionState {
  switch (action.type) {
    case 'imageUploaded':
      return {
        conditionStatus: 'idle',
        uploadedImageFile: action.file,
        uploadedImageName: action.name,
        uploadedImageUrl: action.url,
      };

    case 'conditionGenerated':
      return { ...state, conditionStatus: 'ready' };
  }
}

export const initialDetectionState: DetectionState = {
  c2Axes: [],
  center: [0, 0, 0],
  family: null,
  finerStatus: 'idle',
  fold: 1,
  labels: [],
  majorAxis: [0, 0, 1],
  majorStatus: 'idle',
  minorAxis: [1, 0, 0],
  overlays: [],
  proposedSymmetry: null,
  reflectionPlanesContainingAxis: [],
  reflectionPlanesPerpendicularToAxis: [],
  rotationAxes: [],
  selectedLabel: '',
  selectedMajorCandidateId: null,
  selectedMinorItemId: null,
  selectedOverlayId: null,
  selectableOverlayIds: [],
  symmetryPreview: null,
};

export function normalizeAxisInput(axis: Vector3): Vector3 {
  const length = Math.hypot(axis[0], axis[1], axis[2]);
  const normalized: Vector3 = [axis[0] / length, axis[1] / length, axis[2] / length];
  const dominantIndex =
    Math.abs(normalized[0]) > Math.abs(normalized[1])
      ? Math.abs(normalized[0]) > Math.abs(normalized[2])
        ? 0
        : 2
      : Math.abs(normalized[1]) > Math.abs(normalized[2])
        ? 1
        : 2;

  if (Math.abs(normalized[dominantIndex]) > 0.985) {
    const snapped: Vector3 = [0, 0, 0];
    snapped[dominantIndex] = 1;
    return snapped;
  }

  if (normalized[dominantIndex] < 0) {
    return [-normalized[0], -normalized[1], -normalized[2]];
  }

  return normalized;
}

export function normalizeCenterInput(): Vector3 {
  return [0, 0, 0];
}

export function minorAxisFromVerticalPlane(planeNormal: Vector3, majorAxis: Vector3): Vector3 {
  return normalizeAxisInput([
    planeNormal[1] * majorAxis[2] - planeNormal[2] * majorAxis[1],
    planeNormal[2] * majorAxis[0] - planeNormal[0] * majorAxis[2],
    planeNormal[0] * majorAxis[1] - planeNormal[1] * majorAxis[0],
  ]);
}

export function minorAxisFromSecondaryAxis(axis: Vector3, majorAxis: Vector3): Vector3 {
  const major = normalizeAxisInput(majorAxis);
  const axisDotMajor = axis[0] * major[0] + axis[1] * major[1] + axis[2] * major[2];

  return normalizeAxisInput([
    axis[0] - axisDotMajor * major[0],
    axis[1] - axisDotMajor * major[1],
    axis[2] - axisDotMajor * major[2],
  ]);
}

export function labelsForFamily(family: SymmetryFamily, fold: number) {
  if (family === 'T') {
    return ['T', 'Td', 'Th'];
  }

  if (family === 'O') {
    return ['O', 'Oh'];
  }

  if (family === 'I') {
    return ['I', 'Ih'];
  }

  return [`C${fold}`, `S${2 * fold}`, `C${fold}h`, `C${fold}v`, `D${fold}`, `D${fold}d`, `D${fold}h`];
}

export function axisShortcutDisabled(axis: Vector3, majorAxis: Vector3) {
  const shortcut = normalizeAxisInput(axis);
  const major = normalizeAxisInput(majorAxis);
  const dot = Math.abs(shortcut[0] * major[0] + shortcut[1] * major[1] + shortcut[2] * major[2]);

  return dot > secondaryAxisParallelThreshold;
}

export function canProposeSymmetry(state: DetectionState) {
  return Boolean(
    state.selectedLabel &&
      state.family &&
      state.majorStatus === 'ready' &&
      (state.family !== 'axial' || state.finerStatus === 'ready'),
  );
}

export function canProposeManualSymmetry(state: ManualSymmetryState) {
  return Boolean(state.selectedLabel && state.family);
}

export function manualSymmetryInstruction(state: ManualSymmetryState): string {
  if (state.proposedSymmetry) {
    return 'Review the locked symmetry tuple and viewer preview, then press Confirm.';
  }

  if (state.family === 'axial') {
    return 'Set the principal axis, center, fold, and point group type, then press Confirm proposed symmetry.';
  }

  return 'Set the major and minor axes for the polyhedral frame, then press Confirm proposed symmetry.';
}

export function symSparseStructureInstruction(state: SymSparseStructureState): string {
  if (state.status === 'running') {
    return 'Generating symmetry enforced sparse structure. Progress follows backend flow time.';
  }

  if (state.status === 'ready' && state.voxelCount === 0) {
    return 'Generated occupancy is empty. Change parameters and generate again.';
  }

  if (state.status === 'ready') {
    return 'Sparse structure is ready. Review the voxel count, then go to the next step.';
  }

  return 'Review the locked symmetry tuple and sparse-structure parameters, then press Confirm and generate.';
}

export function vanillaSparseStructureInstruction(state: VanillaSparseStructureState): string {
  if (state.status === 'running') {
    return 'Generating vanilla sparse structure. Progress follows backend flow time.';
  }

  if (state.status === 'ready' && state.voxelCount === 0) {
    return 'Generated occupancy is empty. Change parameters and generate again.';
  }

  if (state.status === 'ready') {
    return 'Sparse structure is ready. Review the voxel count, then go to the next step.';
  }

  return 'Review vanilla sparse-structure parameters, then press Confirm and generate.';
}

export function estimateBf16FlowPeakGb(maxTokens: number): number {
  return 2.691 + 0.00004342 * maxTokens;
}

export function cascadeShapeLatentGridSize(maxTokens: number): number {
  if (maxTokens >= 32768) {
    return 96;
  }

  if (maxTokens >= 24576) {
    return 88;
  }

  if (maxTokens >= 16384) {
    return 80;
  }

  if (maxTokens >= 8192) {
    return 72;
  }

  return 64;
}

export function symShapeInstruction(state: SymShapeState): string {
  if (state.status === 'running') {
    return 'Generating symmetry enforced shape. The viewer shows the input occupancy until the mesh is ready.';
  }

  if (state.status === 'ready') {
    return 'Shape mesh is ready. Review grid metadata, then go to the next step.';
  }

  return 'Review the locked symmetry tuple and shape-generation parameters, then press Confirm and generate.';
}

export function vanillaShapeInstruction(state: VanillaShapeState): string {
  if (state.status === 'running') {
    return 'Generating vanilla shape. The viewer shows the input occupancy until the mesh is ready.';
  }

  if (state.status === 'ready') {
    return 'Shape mesh is ready. Review grid metadata, then go to the next step.';
  }

  return 'Review vanilla shape-generation parameters, then press Confirm and generate.';
}

export function textureInstruction(state: TextureState): string {
  if (state.status === 'running') {
    return 'Generating texture. The viewer shows the input mesh until the textured mesh is ready.';
  }

  if (state.status === 'ready') {
    return 'Textured mesh is ready. Review the result, then go to the next step.';
  }

  return 'Review texture-generation parameters, then press Confirm and generate.';
}

export function symSparseStructureReducer(
  state: SymSparseStructureState,
  action: SymSparseStructureAction,
): SymSparseStructureState {
  switch (action.type) {
    case 'paramsChanged': {
      const nextState = {
        ...state,
        ...action.params,
        generatedOccUrl: '',
        progress: 0,
        status: 'idle' as const,
        voxelCount: 0,
      };

      if (action.params.seed !== undefined) {
        nextState.seed = Math.trunc(action.params.seed);
      }

      if (action.params.steps !== undefined) {
        nextState.steps = Math.max(1, Math.trunc(action.params.steps));
      }

      return nextState;
    }

    case 'seedRandomized':
      return {
        ...state,
        generatedOccUrl: '',
        progress: 0,
        seed: Math.floor(Math.random() * 2147483647),
        status: 'idle',
        voxelCount: 0,
      };

    case 'generationStarted':
      return { ...state, generatedOccUrl: '', progress: 0, status: 'running', voxelCount: 0 };

    case 'generationProgressed':
      return { ...state, progress: action.progress };

    case 'generationFinished':
      return {
        ...state,
        generatedOccUrl: action.generatedOccUrl,
        progress: 1,
        status: 'ready',
        voxelCount: action.voxelCount,
      };
  }
}

export function vanillaSparseStructureReducer(
  state: VanillaSparseStructureState,
  action: VanillaSparseStructureAction,
): VanillaSparseStructureState {
  switch (action.type) {
    case 'paramsChanged': {
      const nextState = {
        ...state,
        ...action.params,
        generatedOccUrl: '',
        progress: 0,
        status: 'idle' as const,
        voxelCount: 0,
      };

      if (action.params.seed !== undefined) {
        nextState.seed = Math.trunc(action.params.seed);
      }

      if (action.params.steps !== undefined) {
        nextState.steps = Math.max(1, Math.trunc(action.params.steps));
      }

      return nextState;
    }

    case 'seedRandomized':
      return {
        ...state,
        generatedOccUrl: '',
        progress: 0,
        seed: Math.floor(Math.random() * 2147483647),
        status: 'idle',
        voxelCount: 0,
      };

    case 'generationStarted':
      return { ...state, generatedOccUrl: '', progress: 0, status: 'running', voxelCount: 0 };

    case 'generationProgressed':
      return { ...state, progress: action.progress };

    case 'generationFinished':
      return {
        ...state,
        generatedOccUrl: action.generatedOccUrl,
        progress: 1,
        status: 'ready',
        voxelCount: action.voxelCount,
      };
  }
}

export function symShapeReducer(state: SymShapeState, action: SymShapeAction): SymShapeState {
  switch (action.type) {
    case 'paramsChanged': {
      const nextState = {
        ...state,
        ...action.params,
        generatedShapeUrl: '',
        oVoxelGridSize: 0,
        progress: 0,
        shapeLatentGridSize: 0,
        status: 'idle' as const,
        voxelCount: 0,
      };

      if (action.params.seed !== undefined) {
        nextState.seed = Math.trunc(action.params.seed);
      }

      if (action.params.steps !== undefined) {
        nextState.steps = Math.max(1, Math.trunc(action.params.steps));
      }

      if (action.params.maxTokens !== undefined) {
        nextState.maxTokens = Math.min(524288, Math.max(4096, Math.trunc(action.params.maxTokens)));
      }

      return nextState;
    }

    case 'seedRandomized':
      return {
        ...state,
        generatedShapeUrl: '',
        oVoxelGridSize: 0,
        progress: 0,
        seed: Math.floor(Math.random() * 2147483647),
        shapeLatentGridSize: 0,
        status: 'idle',
        voxelCount: 0,
      };

    case 'generationStarted':
      return {
        ...state,
        generatedShapeUrl: '',
        inputOccUrl: action.inputOccUrl,
        oVoxelGridSize: 0,
        progress: 0,
        shapeLatentGridSize: 0,
        status: 'running',
        voxelCount: 0,
      };

    case 'generationProgressed':
      return { ...state, progress: action.progress };

    case 'generationFinished':
      return {
        ...state,
        generatedShapeUrl: action.generatedShapeUrl,
        oVoxelGridSize: action.oVoxelGridSize,
        progress: 1,
        shapeLatentGridSize: action.shapeLatentGridSize,
        status: 'ready',
        voxelCount: action.voxelCount,
      };
  }
}

export function vanillaShapeReducer(
  state: VanillaShapeState,
  action: VanillaShapeAction,
): VanillaShapeState {
  switch (action.type) {
    case 'paramsChanged': {
      const nextState = {
        ...state,
        ...action.params,
        generatedShapeUrl: '',
        oVoxelGridSize: 0,
        progress: 0,
        shapeLatentGridSize: 0,
        status: 'idle' as const,
        voxelCount: 0,
      };

      if (action.params.seed !== undefined) {
        nextState.seed = Math.trunc(action.params.seed);
      }

      if (action.params.steps !== undefined) {
        nextState.steps = Math.max(1, Math.trunc(action.params.steps));
      }

      if (action.params.maxTokens !== undefined) {
        nextState.maxTokens = Math.min(524288, Math.max(4096, Math.trunc(action.params.maxTokens)));
      }

      return nextState;
    }

    case 'seedRandomized':
      return {
        ...state,
        generatedShapeUrl: '',
        oVoxelGridSize: 0,
        progress: 0,
        seed: Math.floor(Math.random() * 2147483647),
        shapeLatentGridSize: 0,
        status: 'idle',
        voxelCount: 0,
      };

    case 'generationStarted':
      return {
        ...state,
        generatedShapeUrl: '',
        inputOccUrl: action.inputOccUrl,
        oVoxelGridSize: 0,
        progress: 0,
        shapeLatentGridSize: 0,
        status: 'running',
        voxelCount: 0,
      };

    case 'generationProgressed':
      return { ...state, progress: action.progress };

    case 'generationFinished':
      return {
        ...state,
        generatedShapeUrl: action.generatedShapeUrl,
        oVoxelGridSize: action.oVoxelGridSize,
        progress: 1,
        shapeLatentGridSize: action.shapeLatentGridSize,
        status: 'ready',
        voxelCount: action.voxelCount,
      };
  }
}

export function textureReducer(state: TextureState, action: TextureAction): TextureState {
  switch (action.type) {
    case 'paramsChanged': {
      const nextState = {
        ...state,
        ...action.params,
        generatedTextureUrl: '',
        progress: 0,
        status: 'idle' as const,
      };

      if (action.params.seed !== undefined) {
        nextState.seed = Math.trunc(action.params.seed);
      }

      if (action.params.steps !== undefined) {
        nextState.steps = Math.max(1, Math.trunc(action.params.steps));
      }

      return nextState;
    }

    case 'seedRandomized':
      return {
        ...state,
        generatedTextureUrl: '',
        progress: 0,
        seed: Math.floor(Math.random() * 2147483647),
        status: 'idle',
      };

    case 'generationStarted':
      return {
        ...state,
        generatedTextureUrl: '',
        inputShapeUrl: action.inputShapeUrl,
        progress: 0,
        status: 'running',
      };

    case 'generationProgressed':
      return { ...state, progress: action.progress };

    case 'generationFinished':
      return {
        ...state,
        generatedTextureUrl: action.generatedTextureUrl,
        progress: 1,
        status: 'ready',
      };
  }
}

export function manualSymmetryReducer(
  state: ManualSymmetryState,
  action: ManualSymmetryAction,
): ManualSymmetryState {
  switch (action.type) {
    case 'majorAxisChanged':
      return { ...state, majorAxis: action.axis, proposedSymmetry: null, symmetryPreview: null };

    case 'minorAxisChanged':
      return { ...state, minorAxis: action.axis, proposedSymmetry: null, symmetryPreview: null };

    case 'centerChanged':
      return { ...state, center: action.center, proposedSymmetry: null, symmetryPreview: null };

    case 'majorAxisShortcutPicked':
      return { ...state, majorAxis: action.axis, proposedSymmetry: null, symmetryPreview: null };

    case 'minorAxisShortcutPicked':
      return { ...state, minorAxis: action.axis, proposedSymmetry: null, symmetryPreview: null };

    case 'familyPicked': {
      const labels = labelsForFamily(action.family, state.fold);

      return {
        ...state,
        family: action.family,
        labels,
        proposedSymmetry: null,
        selectedLabel: labels[0],
        symmetryPreview: null,
      };
    }

    case 'foldChanged': {
      const fold = Math.max(1, Math.trunc(action.fold));
      const labels = labelsForFamily(state.family, fold);

      return {
        ...state,
        fold,
        labels,
        proposedSymmetry: null,
        selectedLabel: labels[0],
        symmetryPreview: null,
      };
    }

    case 'labelPicked':
      return { ...state, proposedSymmetry: null, selectedLabel: action.label, symmetryPreview: null };

    case 'proposeSymmetry': {
      const proposedSymmetry: ProposedSymmetry = {
        center: [state.center[0], state.center[1], state.center[2]],
        label: state.selectedLabel,
        majorAxis: normalizeAxisInput(state.majorAxis),
        minorAxis: normalizeAxisInput(state.minorAxis),
      };

      return { ...state, proposedSymmetry, symmetryPreview: proposedSymmetry };
    }

    case 'confirmSymmetry':
      // TODO: Submit state.proposedSymmetry as the manual_sym node result
      // once backend session/result handling is wired.
      return state;
  }
}

export function detectionInstruction(state: DetectionState): string {
  if (state.majorStatus === 'idle') {
    return 'Press Detect major axis to find rotation symmetry candidates.';
  }

  if (state.majorStatus === 'running') {
    return 'Detecting rotation axes. The viewer will update when candidates are ready.';
  }

  if (state.majorStatus === 'empty') {
    return 'No rotation symmetry detected. Use manual symmetry selection.';
  }

  if (!state.family) {
    return 'Select a major axis in the viewer or edit the axis values. Then choose axial, T, O, or I.';
  }

  if (state.family === 'axial' && state.finerStatus === 'idle') {
    return 'Press Detect finer type to find C2 axes and mirror planes.';
  }

  if (state.finerStatus === 'running') {
    return 'Detecting C2 axes and mirror planes.';
  }

  if (state.proposedSymmetry) {
    return 'Review the locked symmetry tuple and viewer preview, then press Confirm.';
  }

  if (state.family === 'T' || state.family === 'O' || state.family === 'I') {
    const secondaryFold = state.family === 'T' ? 3 : state.family === 'O' ? 4 : 5;

    if (state.selectableOverlayIds.length > 0) {
      return `Select a non-primary C${secondaryFold} axis in the viewer, or keep the current minor axis. Then press Confirm proposed symmetry.`;
    }

    return `No valid secondary C${secondaryFold} axis is available. Edit the minor axis manually, then press Confirm proposed symmetry.`;
  }

  if (state.family !== 'axial') {
    return 'Select a valid secondary axis in the viewer or edit the minor axis. Then press Confirm proposed symmetry.';
  }

  if (state.c2Axes.length > 0) {
    return 'Select a C2 axis in the viewer or keep the current one. Then press Confirm proposed symmetry.';
  }

  if (state.reflectionPlanesContainingAxis.length > 0) {
    return 'Select a mirror plane in the viewer or keep the current one. Then press Confirm proposed symmetry.';
  }

  if (canProposeSymmetry(state)) {
    return 'Press Confirm proposed symmetry to lock the current tuple.';
  }

  return 'Review point group type and axis values.';
}

export function detectionReducer(state: DetectionState, action: DetectionAction): DetectionState {
  switch (action.type) {
    case 'majorDetectionStarted':
      return { ...state, majorStatus: 'running', proposedSymmetry: null, symmetryPreview: null };

    case 'rotationAxesLoaded': {
      if (action.candidates.length === 0) {
        return {
          ...state,
          majorStatus: 'empty',
          overlays: [],
          proposedSymmetry: null,
          rotationAxes: [],
          selectableOverlayIds: [],
          selectedMajorCandidateId: null,
          selectedOverlayId: null,
          symmetryPreview: null,
        };
      }

      const candidate = action.candidates[0];
      const fold = candidate.foldI;

      return {
        ...state,
        center: normalizeCenterInput(),
        family: null,
        finerStatus: 'idle',
        fold,
        labels: [],
        majorAxis: normalizeAxisInput(candidate.axis),
        majorStatus: 'ready',
        overlays: action.candidates.map((item) => ({
          axis: item.axis,
          center: item.center,
          color: item.color,
          fold: item.foldI,
          id: item.id,
          kind: 'rotation_axis',
          label: `${item.foldI}`,
        })),
        proposedSymmetry: null,
        rotationAxes: action.candidates,
        selectableOverlayIds: action.candidates.map((item) => item.id),
        selectedLabel: '',
        selectedMajorCandidateId: candidate.id,
        selectedMinorItemId: null,
        selectedOverlayId: candidate.id,
        symmetryPreview: null,
      };
    }

    case 'majorCandidatePicked': {
      const candidate = state.rotationAxes.find((item) => item.id === action.candidateId)!;
      const labels = state.family ? labelsForFamily(state.family, candidate.foldI) : state.labels;

      return {
        ...state,
        center: normalizeCenterInput(),
        fold: candidate.foldI,
        labels,
        majorAxis: normalizeAxisInput(candidate.axis),
        proposedSymmetry: null,
        selectedLabel: state.family ? labels[0] : state.selectedLabel,
        selectedMajorCandidateId: candidate.id,
        selectedOverlayId: candidate.id,
        symmetryPreview: null,
      };
    }

    case 'majorAxisChanged':
      return { ...state, majorAxis: action.axis, proposedSymmetry: null, symmetryPreview: null };

    case 'centerChanged':
      return { ...state, center: action.center, proposedSymmetry: null, symmetryPreview: null };

    case 'foldChanged': {
      const labels = state.family ? labelsForFamily(state.family, action.fold) : state.labels;

      return {
        ...state,
        fold: action.fold,
        labels,
        proposedSymmetry: null,
        selectedLabel: state.family ? labels[0] : state.selectedLabel,
        symmetryPreview: null,
      };
    }

    case 'majorAxisNormalized':
      return {
        ...state,
        majorAxis: normalizeAxisInput(state.majorAxis),
        proposedSymmetry: null,
        symmetryPreview: null,
      };

    case 'centerNormalized':
      return {
        ...state,
        center: normalizeCenterInput(),
        proposedSymmetry: null,
        symmetryPreview: null,
      };

    case 'familyPicked': {
      const labels = labelsForFamily(action.family, state.fold);

      if (action.family !== 'axial') {
        const secondaryFold = action.family === 'T' ? 3 : action.family === 'O' ? 4 : 5;
        const major = normalizeAxisInput(state.majorAxis);
        const secondaryCandidates = state.rotationAxes.filter((candidate) => {
          const axis = normalizeAxisInput(candidate.axis);
          const dot =
            Math.abs(axis[0] * major[0] + axis[1] * major[1] + axis[2] * major[2]);

          return candidate.foldI === secondaryFold && dot < secondaryAxisParallelThreshold;
        });
        const selectedSecondary = secondaryCandidates[0];

        return {
          ...state,
          c2Axes: [],
          family: action.family,
          finerStatus: 'idle',
          labels,
          minorAxis: selectedSecondary
            ? minorAxisFromSecondaryAxis(selectedSecondary.axis, state.majorAxis)
            : state.minorAxis,
          overlays: secondaryCandidates.map((candidate) => ({
            axis: candidate.axis,
            center: candidate.center,
            color: candidate.color,
            fold: candidate.foldI,
            id: candidate.id,
            kind: 'rotation_axis',
            label: `${candidate.foldI}`,
          })),
          proposedSymmetry: null,
          reflectionPlanesContainingAxis: [],
          reflectionPlanesPerpendicularToAxis: [],
          selectableOverlayIds: secondaryCandidates.map((candidate) => candidate.id),
          selectedLabel: labels[0],
          selectedMinorItemId: selectedSecondary?.id ?? null,
          selectedOverlayId: selectedSecondary?.id ?? null,
          symmetryPreview: null,
        };
      }

      return {
        ...state,
        c2Axes: [],
        family: action.family,
        finerStatus: 'idle',
        labels,
        overlays: [],
        proposedSymmetry: null,
        reflectionPlanesContainingAxis: [],
        reflectionPlanesPerpendicularToAxis: [],
        selectableOverlayIds: [],
        selectedLabel: labels[0],
        selectedMinorItemId: null,
        selectedOverlayId: null,
        symmetryPreview: null,
      };
    }

    case 'finerDetectionStarted':
      return { ...state, finerStatus: 'running', proposedSymmetry: null, symmetryPreview: null };

    case 'finerResultLoaded': {
      const majorOverlay: SymmetryOverlay = {
        axis: state.majorAxis,
        center: state.center,
        color: '#ffffff',
        fold: state.fold,
        id: state.selectedMajorCandidateId ?? 'selected-major-axis',
        kind: 'rotation_axis',
        label: `${state.fold}`,
      };
      const c2Overlays: SymmetryOverlay[] = action.result.c2Axes.map((axis) => ({
        axis: axis.axisCorrected,
        center: axis.centerCorrected,
        color: axis.color,
        fold: 2,
        id: axis.id,
        kind: 'c2_axis',
        label: 'C2',
      }));
      const planeOverlays: SymmetryOverlay[] = [
        ...action.result.reflectionPlanesContainingAxis.map((plane) => ({
          center: state.center,
          color: plane.color,
          id: plane.id,
          kind: 'reflection_plane' as const,
          label: 'mirror',
          majorAxis: state.majorAxis,
          normal: plane.normalCorrected,
          role: plane.role,
        })),
        ...action.result.reflectionPlanesPerpendicularToAxis.map((plane) => ({
          center: state.center,
          color: plane.color,
          id: plane.id,
          kind: 'reflection_plane' as const,
          label: 'mirror',
          majorAxis: state.majorAxis,
          normal: plane.normalCorrected,
          role: plane.role,
        })),
      ];
      const verticalPlane = action.result.reflectionPlanesContainingAxis[0];
      const horizontalPlane = action.result.reflectionPlanesPerpendicularToAxis[0];
      const c2Axis = action.result.c2Axes[0];
      const selectedMinorItemId = c2Axis?.id ?? verticalPlane?.id ?? null;
      const selectableOverlayIds =
        action.result.c2Axes.length > 0
          ? action.result.c2Axes.map((axis) => axis.id)
          : action.result.reflectionPlanesContainingAxis.map((plane) => plane.id);
      const selectedLabel =
        c2Axis && verticalPlane && horizontalPlane
          ? `D${state.fold}h`
          : c2Axis && verticalPlane
            ? `D${state.fold}d`
            : c2Axis
              ? `D${state.fold}`
              : verticalPlane
                ? `C${state.fold}v`
                : horizontalPlane
                  ? `C${state.fold}h`
                  : `C${state.fold}`;

      return {
        ...state,
        c2Axes: action.result.c2Axes,
        finerStatus: 'ready',
        labels: labelsForFamily('axial', state.fold),
        minorAxis: c2Axis
          ? normalizeAxisInput(c2Axis.axisCorrected)
          : verticalPlane
            ? minorAxisFromVerticalPlane(verticalPlane.normalCorrected, state.majorAxis)
            : state.minorAxis,
        overlays: [majorOverlay, ...c2Overlays, ...planeOverlays],
        proposedSymmetry: null,
        reflectionPlanesContainingAxis: action.result.reflectionPlanesContainingAxis,
        reflectionPlanesPerpendicularToAxis: action.result.reflectionPlanesPerpendicularToAxis,
        selectableOverlayIds,
        selectedLabel,
        selectedMinorItemId,
        selectedOverlayId: selectedMinorItemId ?? majorOverlay.id,
        symmetryPreview: null,
      };
    }

    case 'labelPicked':
      return {
        ...state,
        proposedSymmetry: null,
        selectedLabel: action.label,
        symmetryPreview: null,
      };

    case 'minorAxisChanged':
      return { ...state, minorAxis: action.axis, proposedSymmetry: null, symmetryPreview: null };

    case 'minorItemPicked': {
      const c2Axis = state.c2Axes.find((item) => item.id === action.itemId);
      const verticalPlane = state.reflectionPlanesContainingAxis.find((item) => item.id === action.itemId);

      return {
        ...state,
        minorAxis: c2Axis
          ? normalizeAxisInput(c2Axis.axisCorrected)
          : minorAxisFromVerticalPlane(verticalPlane!.normalCorrected, state.majorAxis),
        proposedSymmetry: null,
        selectedMinorItemId: action.itemId,
        selectedOverlayId: action.itemId,
        symmetryPreview: null,
      };
    }

    case 'overlayPicked': {
      const c2Axis = state.c2Axes.find((item) => item.id === action.overlayId);
      const verticalPlane = state.reflectionPlanesContainingAxis.find((item) => item.id === action.overlayId);
      const majorCandidate = state.rotationAxes.find((item) => item.id === action.overlayId);

      if ((state.family === 'T' || state.family === 'O' || state.family === 'I') && majorCandidate) {
        const secondaryFold = state.family === 'T' ? 3 : state.family === 'O' ? 4 : 5;
        const axis = normalizeAxisInput(majorCandidate.axis);
        const major = normalizeAxisInput(state.majorAxis);
        const dot = Math.abs(axis[0] * major[0] + axis[1] * major[1] + axis[2] * major[2]);

        if (majorCandidate.foldI === secondaryFold && dot < secondaryAxisParallelThreshold) {
          return {
            ...state,
            minorAxis: minorAxisFromSecondaryAxis(majorCandidate.axis, state.majorAxis),
            proposedSymmetry: null,
            selectedMinorItemId: majorCandidate.id,
            selectedOverlayId: majorCandidate.id,
            symmetryPreview: null,
          };
        }

        return state;
      }

      if (majorCandidate) {
        const labels = state.family ? labelsForFamily(state.family, majorCandidate.foldI) : state.labels;

        return {
          ...state,
          center: normalizeCenterInput(),
          fold: majorCandidate.foldI,
          labels,
          majorAxis: normalizeAxisInput(majorCandidate.axis),
          proposedSymmetry: null,
          selectedLabel: state.family ? labels[0] : state.selectedLabel,
          selectedMajorCandidateId: majorCandidate.id,
          selectedOverlayId: majorCandidate.id,
          symmetryPreview: null,
        };
      }

      if (c2Axis) {
        return {
          ...state,
          minorAxis: normalizeAxisInput(c2Axis.axisCorrected),
          proposedSymmetry: null,
          selectedMinorItemId: action.overlayId,
          selectedOverlayId: action.overlayId,
          symmetryPreview: null,
        };
      }

      if (verticalPlane && state.c2Axes.length === 0) {
        return {
          ...state,
          minorAxis: minorAxisFromVerticalPlane(verticalPlane.normalCorrected, state.majorAxis),
          proposedSymmetry: null,
          selectedMinorItemId: action.overlayId,
          selectedOverlayId: action.overlayId,
          symmetryPreview: null,
        };
      }

      return state;
    }

    case 'minorAxisNormalized':
      return {
        ...state,
        minorAxis: normalizeAxisInput(state.minorAxis),
        proposedSymmetry: null,
        symmetryPreview: null,
      };

    case 'proposeSymmetry': {
      const proposedSymmetry: ProposedSymmetry = {
        center: [state.center[0], state.center[1], state.center[2]],
        label: state.selectedLabel,
        majorAxis: [state.majorAxis[0], state.majorAxis[1], state.majorAxis[2]],
        minorAxis: [state.minorAxis[0], state.minorAxis[1], state.minorAxis[2]],
      };

      return {
        ...state,
        c2Axes: [],
        overlays: [],
        proposedSymmetry,
        reflectionPlanesContainingAxis: [],
        reflectionPlanesPerpendicularToAxis: [],
        selectableOverlayIds: [],
        selectedMinorItemId: null,
        selectedOverlayId: null,
        symmetryPreview: proposedSymmetry,
      };
    }

    case 'confirmSymmetry':
      // TODO: Submit state.proposedSymmetry as the detect_adjust_symmetry node result
      // once backend session/result handling is wired.
      return state;
  }
}
