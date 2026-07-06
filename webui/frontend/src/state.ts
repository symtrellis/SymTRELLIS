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

export const initialCurrentNodeId: NodeId = 'detect_sym';

export const initialDagStatus: Record<NodeId, DagStatus> = {
  img_cond: 'completed',
  nat_ss: 'completed',
  nat_shape: 'completed',
  detect_sym: 'current',
  manual_sym: 'inactive',
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

export type ProposedSymmetry = SymmetryTuple;

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

export function canProposeSymmetry(state: DetectionState) {
  return Boolean(
    state.selectedLabel &&
      state.family &&
      state.majorStatus === 'ready' &&
      (state.family !== 'axial' || state.finerStatus === 'ready'),
  );
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
