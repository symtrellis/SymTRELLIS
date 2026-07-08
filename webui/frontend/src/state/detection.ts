import type {
  ActionKey,
  FinerSymmetryResult,
  ReflectionPlaneCandidate,
  RotationAxisCandidate,
  SymmetryFamily,
  SymmetryOverlay,
  SymmetryTuple,
  Vector3,
} from '../types';
import {
  familySecondaryFold,
  labelsForFamily,
  minorAxisFromSecondaryAxis,
  minorAxisFromVerticalPlane,
  normalizeAxisInput,
  normalizeCenterInput,
  secondaryAxisParallelThreshold,
} from './symmetry';
import type { ProposedSymmetry } from './symmetry';

export type DetectionStatus = 'idle' | 'running' | 'ready' | 'empty' | 'failed';

export type DetectionState = {
  c2Axes: FinerSymmetryResult['c2Axes'];
  center: Vector3;
  errorMessage: string;
  family: SymmetryFamily | null;
  finerActionKey: ActionKey | null;
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
  rotationActionKey: ActionKey | null;
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
  | { actionKey: ActionKey; candidates: RotationAxisCandidate[]; type: 'rotationAxesLoaded' }
  | { message: string; type: 'majorDetectionFailed' }
  | { candidateId: string; type: 'majorCandidatePicked' }
  | { axis: Vector3; type: 'majorAxisChanged' }
  | { center: Vector3; type: 'centerChanged' }
  | { fold: number; type: 'foldChanged' }
  | { type: 'majorAxisNormalized' }
  | { type: 'centerNormalized' }
  | { family: SymmetryFamily; type: 'familyPicked' }
  | { type: 'finerDetectionStarted' }
  | { actionKey: ActionKey; result: FinerSymmetryResult; type: 'finerResultLoaded' }
  | { message: string; type: 'finerDetectionFailed' }
  | { label: string; type: 'labelPicked' }
  | { axis: Vector3; type: 'minorAxisChanged' }
  | { overlayId: string; type: 'overlayPicked' }
  | { type: 'minorAxisNormalized' }
  | { type: 'proposeSymmetry' }
  | { type: 'reset' };

export const initialDetectionState: DetectionState = {
  c2Axes: [],
  center: [0, 0, 0],
  errorMessage: '',
  family: null,
  finerActionKey: null,
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
  rotationActionKey: null,
  rotationAxes: [],
  selectedLabel: '',
  selectedMajorCandidateId: null,
  selectedMinorItemId: null,
  selectedOverlayId: null,
  selectableOverlayIds: [],
  symmetryPreview: null,
};

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

  if (state.majorStatus === 'failed' || state.finerStatus === 'failed') {
    return state.errorMessage || 'Symmetry detection failed.';
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
    const secondaryFold = familySecondaryFold(state.family);

    if (state.selectableOverlayIds.length > 0) {
      return `Select a non-primary C${secondaryFold} axis in the viewer, or keep the current minor axis. Then press Confirm proposed symmetry.`;
    }

    return `No valid secondary C${secondaryFold} axis is available. Edit the minor axis manually, then press Confirm proposed symmetry.`;
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
  if (action.type === 'reset') {
    return initialDetectionState;
  }

  if (action.type === 'majorDetectionStarted') {
    return {
      ...state,
      c2Axes: [],
      center: [0, 0, 0],
      errorMessage: '',
      family: null,
      finerActionKey: null,
      finerStatus: 'idle',
      labels: [],
      majorAxis: [0, 0, 1],
      majorStatus: 'running',
      overlays: [],
      proposedSymmetry: null,
      reflectionPlanesContainingAxis: [],
      reflectionPlanesPerpendicularToAxis: [],
      rotationActionKey: null,
      rotationAxes: [],
      selectableOverlayIds: [],
      selectedLabel: '',
      selectedMajorCandidateId: null,
      selectedMinorItemId: null,
      selectedOverlayId: null,
      symmetryPreview: null,
    };
  }

  if (action.type === 'rotationAxesLoaded') {
    if (action.candidates.length === 0) {
      return {
        ...state,
        majorStatus: 'empty',
        overlays: [],
        proposedSymmetry: null,
        rotationActionKey: action.actionKey,
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
      errorMessage: '',
      family: null,
      finerActionKey: null,
      finerStatus: 'idle',
      fold,
      labels: [],
      majorAxis: normalizeAxisInput(candidate.axis),
      majorStatus: 'ready',
      overlays: action.candidates.map(rotationAxisOverlay),
      proposedSymmetry: null,
      rotationActionKey: action.actionKey,
      rotationAxes: action.candidates,
      selectableOverlayIds: action.candidates.map((item) => item.id),
      selectedLabel: '',
      selectedMajorCandidateId: candidate.id,
      selectedMinorItemId: null,
      selectedOverlayId: candidate.id,
      symmetryPreview: null,
    };
  }

  if (action.type === 'majorDetectionFailed') {
    return {
      ...state,
      c2Axes: [],
      errorMessage: action.message,
      family: null,
      finerActionKey: null,
      finerStatus: 'idle',
      labels: [],
      majorStatus: 'failed',
      overlays: [],
      proposedSymmetry: null,
      reflectionPlanesContainingAxis: [],
      reflectionPlanesPerpendicularToAxis: [],
      rotationActionKey: null,
      rotationAxes: [],
      selectableOverlayIds: [],
      selectedLabel: '',
      selectedMajorCandidateId: null,
      selectedMinorItemId: null,
      selectedOverlayId: null,
      symmetryPreview: null,
    };
  }

  if (action.type === 'majorCandidatePicked') {
    return pickMajorCandidate(state, action.candidateId);
  }

  if (action.type === 'majorAxisChanged') {
    return { ...state, majorAxis: action.axis, proposedSymmetry: null, symmetryPreview: null };
  }

  if (action.type === 'centerChanged') {
    return { ...state, center: action.center, proposedSymmetry: null, symmetryPreview: null };
  }

  if (action.type === 'foldChanged') {
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

  if (action.type === 'majorAxisNormalized') {
    return {
      ...state,
      majorAxis: normalizeAxisInput(state.majorAxis),
      proposedSymmetry: null,
      symmetryPreview: null,
    };
  }

  if (action.type === 'centerNormalized') {
    return {
      ...state,
      center: normalizeCenterInput(),
      proposedSymmetry: null,
      symmetryPreview: null,
    };
  }

  if (action.type === 'familyPicked') {
    return pickFamily(state, action.family);
  }

  if (action.type === 'finerDetectionStarted') {
    return {
      ...state,
      c2Axes: [],
      errorMessage: '',
      finerActionKey: null,
      finerStatus: 'running',
      overlays: [],
      proposedSymmetry: null,
      reflectionPlanesContainingAxis: [],
      reflectionPlanesPerpendicularToAxis: [],
      selectableOverlayIds: [],
      selectedMinorItemId: null,
      selectedOverlayId: null,
      symmetryPreview: null,
    };
  }

  if (action.type === 'finerResultLoaded') {
    return loadFinerResult(state, action.result, action.actionKey);
  }

  if (action.type === 'finerDetectionFailed') {
    return {
      ...state,
      c2Axes: [],
      errorMessage: action.message,
      finerActionKey: null,
      finerStatus: 'failed',
      overlays: [],
      proposedSymmetry: null,
      reflectionPlanesContainingAxis: [],
      reflectionPlanesPerpendicularToAxis: [],
      selectableOverlayIds: [],
      selectedMinorItemId: null,
      selectedOverlayId: null,
      symmetryPreview: null,
    };
  }

  if (action.type === 'labelPicked') {
    return { ...state, proposedSymmetry: null, selectedLabel: action.label, symmetryPreview: null };
  }

  if (action.type === 'minorAxisChanged') {
    return { ...state, minorAxis: action.axis, proposedSymmetry: null, symmetryPreview: null };
  }

  if (action.type === 'overlayPicked') {
    return pickOverlay(state, action.overlayId);
  }

  if (action.type === 'minorAxisNormalized') {
    return {
      ...state,
      minorAxis: normalizeAxisInput(state.minorAxis),
      proposedSymmetry: null,
      symmetryPreview: null,
    };
  }

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

function pickMajorCandidate(state: DetectionState, candidateId: string): DetectionState {
  const candidate = state.rotationAxes.find((item) => item.id === candidateId);

  if (!candidate) {
    return state;
  }

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

function pickFamily(state: DetectionState, family: SymmetryFamily): DetectionState {
  const labels = labelsForFamily(family, state.fold);

  if (family !== 'axial') {
    const secondaryFold = familySecondaryFold(family);
    const major = normalizeAxisInput(state.majorAxis);
    const secondaryCandidates = state.rotationAxes.filter((candidate) => {
      const axis = normalizeAxisInput(candidate.axis);
      const dot = Math.abs(axis[0] * major[0] + axis[1] * major[1] + axis[2] * major[2]);

      return candidate.foldI === secondaryFold && dot < secondaryAxisParallelThreshold;
    });
    const selectedSecondary = secondaryCandidates[0];

    return {
      ...state,
      c2Axes: [],
      family,
      finerStatus: 'idle',
      labels,
      minorAxis: selectedSecondary
        ? minorAxisFromSecondaryAxis(selectedSecondary.axis, state.majorAxis)
        : state.minorAxis,
      overlays: secondaryCandidates.map(rotationAxisOverlay),
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
    family,
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

function loadFinerResult(
  state: DetectionState,
  result: FinerSymmetryResult,
  actionKey: ActionKey,
): DetectionState {
  const majorOverlay: SymmetryOverlay = {
    axis: state.majorAxis,
    center: state.center,
    color: '#ffffff',
    fold: state.fold,
    id: state.selectedMajorCandidateId ?? 'selected-major-axis',
    kind: 'rotation_axis',
    label: `${state.fold}`,
  };
  const c2Overlays: SymmetryOverlay[] = result.c2Axes.map((axis) => ({
    axis: axis.axisCorrected,
    center: axis.centerCorrected,
    color: axis.color,
    fold: 2,
    id: axis.id,
    kind: 'c2_axis',
    label: 'C2',
  }));
  const planeOverlays: SymmetryOverlay[] = [
    ...result.reflectionPlanesContainingAxis.map((plane) => reflectionPlaneOverlay(plane, state)),
    ...result.reflectionPlanesPerpendicularToAxis.map((plane) => reflectionPlaneOverlay(plane, state)),
  ];
  const verticalPlane = result.reflectionPlanesContainingAxis[0];
  const horizontalPlane = result.reflectionPlanesPerpendicularToAxis[0];
  const c2Axis = result.c2Axes[0];
  const selectedMinorItemId = c2Axis?.id ?? verticalPlane?.id ?? null;
  const selectableOverlayIds =
    result.c2Axes.length > 0
      ? result.c2Axes.map((axis) => axis.id)
      : result.reflectionPlanesContainingAxis.map((plane) => plane.id);
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
    c2Axes: result.c2Axes,
    finerActionKey: actionKey,
    finerStatus: 'ready',
    labels: labelsForFamily('axial', state.fold),
    minorAxis: c2Axis
      ? normalizeAxisInput(c2Axis.axisCorrected)
      : verticalPlane
        ? minorAxisFromVerticalPlane(verticalPlane.normalCorrected, state.majorAxis)
        : state.minorAxis,
    overlays: [majorOverlay, ...c2Overlays, ...planeOverlays],
    proposedSymmetry: null,
    reflectionPlanesContainingAxis: result.reflectionPlanesContainingAxis,
    reflectionPlanesPerpendicularToAxis: result.reflectionPlanesPerpendicularToAxis,
    selectableOverlayIds,
    selectedLabel,
    selectedMinorItemId,
    selectedOverlayId: selectedMinorItemId ?? majorOverlay.id,
    symmetryPreview: null,
  };
}

function pickOverlay(state: DetectionState, overlayId: string): DetectionState {
  const c2Axis = state.c2Axes.find((item) => item.id === overlayId);
  const verticalPlane = state.reflectionPlanesContainingAxis.find((item) => item.id === overlayId);
  const majorCandidate = state.rotationAxes.find((item) => item.id === overlayId);

  if ((state.family === 'T' || state.family === 'O' || state.family === 'I') && majorCandidate) {
    const secondaryFold = familySecondaryFold(state.family);
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
    return pickMajorCandidate(state, majorCandidate.id);
  }

  if (c2Axis) {
    return {
      ...state,
      minorAxis: normalizeAxisInput(c2Axis.axisCorrected),
      proposedSymmetry: null,
      selectedMinorItemId: overlayId,
      selectedOverlayId: overlayId,
      symmetryPreview: null,
    };
  }

  if (verticalPlane && state.c2Axes.length === 0) {
    return {
      ...state,
      minorAxis: minorAxisFromVerticalPlane(verticalPlane.normalCorrected, state.majorAxis),
      proposedSymmetry: null,
      selectedMinorItemId: overlayId,
      selectedOverlayId: overlayId,
      symmetryPreview: null,
    };
  }

  return state;
}

function rotationAxisOverlay(item: RotationAxisCandidate): SymmetryOverlay {
  return {
    axis: item.axis,
    center: item.center,
    color: item.color,
    fold: item.foldI,
    id: item.id,
    kind: 'rotation_axis',
    label: `${item.foldI}`,
  };
}

function reflectionPlaneOverlay(plane: ReflectionPlaneCandidate, state: DetectionState): SymmetryOverlay {
  return {
    center: state.center,
    color: plane.color,
    id: plane.id,
    kind: 'reflection_plane',
    label: 'mirror',
    majorAxis: state.majorAxis,
    normal: plane.normalCorrected,
    role: plane.role,
  };
}
