import type {
  ActionKey,
  C2AxisCandidate,
  FinerSymmetryDetectionResult,
  FinerSymmetryResult,
  ReflectionPlaneDetectionResult,
  ReflectionPlaneCandidate,
  RotationAxisDetectionResult,
  RotationAxisCandidate,
  SymmetryFamily,
  SymmetryOverlay,
  SymmetryTuple,
  Vector3,
} from '../types';
import type { WorkflowActionRun } from './workflow';
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

export type ReflectionDetectionCandidate = {
  center: Vector3;
  color: string;
  dbscanLabel: number;
  foldIValidation: number;
  id: string;
  normal: Vector3;
  ratio: number;
  rmse: number;
};

export type DetectionState = {
  activeDetectionKind: 'reflection' | 'rotation' | null;
  c2AxesPerpendicularToAxis: FinerSymmetryResult['c2AxesPerpendicularToAxis'];
  center: Vector3;
  confirmationError: string;
  confirming: boolean;
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
  reflectionActionKey: ActionKey | null;
  reflectionCenter: Vector3;
  reflectionNormal: Vector3;
  reflectionPlanesContainingAxis: ReflectionPlaneCandidate[];
  reflectionPlanesPerpendicularToAxis: ReflectionPlaneCandidate[];
  reflectionStatus: DetectionStatus;
  rotationActionKey: ActionKey | null;
  rotationAxes: RotationAxisCandidate[];
  selectedLabel: string;
  selectedMajorCandidateId: string | null;
  selectedMinorItemId: string | null;
  selectedOverlayId: string | null;
  selectableOverlayIds: string[];
  standaloneReflectionPlanes: ReflectionDetectionCandidate[];
  symmetryPreview: SymmetryTuple | null;
};

export type DetectionAction =
  | { type: 'majorDetectionStarted' }
  | { actionKey: ActionKey; candidates: RotationAxisDetectionResult[]; type: 'rotationAxesLoaded' }
  | { message: string; type: 'majorDetectionFailed' }
  | { candidateId: string; type: 'majorCandidatePicked' }
  | { axis: Vector3; type: 'majorAxisChanged' }
  | { center: Vector3; type: 'centerChanged' }
  | { fold: number; type: 'foldChanged' }
  | { type: 'majorAxisNormalized' }
  | { type: 'centerNormalized' }
  | { family: SymmetryFamily; type: 'familyPicked' }
  | { type: 'finerDetectionStarted' }
  | { actionKey: ActionKey; result: FinerSymmetryDetectionResult; type: 'finerResultLoaded' }
  | { message: string; type: 'finerDetectionFailed' }
  | { type: 'reflectionDetectionStarted' }
  | {
      actionKey: ActionKey;
      candidates: ReflectionPlaneDetectionResult[];
      type: 'reflectionPlanesLoaded';
    }
  | { message: string; type: 'reflectionDetectionFailed' }
  | { normal: Vector3; type: 'reflectionNormalChanged' }
  | { center: Vector3; type: 'reflectionCenterChanged' }
  | { type: 'reflectionNormalNormalized' }
  | { type: 'reflectionCenterNormalized' }
  | { label: string; type: 'labelPicked' }
  | { axis: Vector3; type: 'minorAxisChanged' }
  | { overlayId: string; type: 'overlayPicked' }
  | { type: 'minorAxisNormalized' }
  | { type: 'proposeSymmetry' }
  | { type: 'confirmationStarted' }
  | { message: string; type: 'confirmationFailed' }
  | { type: 'confirmationCompleted' }
  | {
      confirmedSymmetry: SymmetryTuple | null;
      finerAction: WorkflowActionRun | null;
      reflectionAction: WorkflowActionRun | null;
      rotationAction: WorkflowActionRun | null;
      type: 'detectionRestored';
    }
  | { type: 'reset' };

export const initialDetectionState: DetectionState = {
  activeDetectionKind: null,
  c2AxesPerpendicularToAxis: [],
  center: [0, 0, 0],
  confirmationError: '',
  confirming: false,
  errorMessage: '',
  family: 'axial',
  finerActionKey: null,
  finerStatus: 'idle',
  fold: 1,
  labels: labelsForFamily('axial', 1),
  majorAxis: [0, 0, 1],
  majorStatus: 'idle',
  minorAxis: [1, 0, 0],
  overlays: [],
  proposedSymmetry: null,
  reflectionActionKey: null,
  reflectionCenter: [0, 0, 0],
  reflectionNormal: [0, 0, 1],
  reflectionPlanesContainingAxis: [],
  reflectionPlanesPerpendicularToAxis: [],
  reflectionStatus: 'idle',
  rotationActionKey: null,
  rotationAxes: [],
  selectedLabel: 'C1',
  selectedMajorCandidateId: null,
  selectedMinorItemId: null,
  selectedOverlayId: null,
  selectableOverlayIds: [],
  standaloneReflectionPlanes: [],
  symmetryPreview: null,
};

export function canProposeSymmetry(state: DetectionState) {
  if (state.activeDetectionKind === 'reflection') {
    return state.reflectionStatus === 'ready';
  }

  return Boolean(
    state.selectedLabel &&
      state.family &&
      state.activeDetectionKind === 'rotation' &&
      state.majorStatus === 'ready',
  );
}

export function detectionInstruction(state: DetectionState): string {
  if (state.confirming) {
    return 'Confirming symmetry.';
  }

  if (state.confirmationError) {
    return state.confirmationError;
  }

  if (state.majorStatus === 'running') {
    return 'Detecting rotation axes. The viewer will update when candidates are ready.';
  }

  if (state.reflectionStatus === 'running') {
    return 'Detecting reflection planes. The viewer will update when candidates are ready.';
  }

  if (state.finerStatus === 'running') {
    return 'Detecting C2 axes and mirror planes.';
  }

  if (
    (state.activeDetectionKind === 'rotation' &&
      (state.majorStatus === 'failed' || state.finerStatus === 'failed')) ||
    (state.activeDetectionKind === 'reflection' && state.reflectionStatus === 'failed')
  ) {
    return state.errorMessage || 'Symmetry detection failed.';
  }

  if (state.proposedSymmetry) {
    return 'Review the locked symmetry tuple and viewer preview, then press Confirm.';
  }

  if (state.activeDetectionKind === 'reflection') {
    if (state.reflectionStatus === 'empty') {
      return 'No reflection plane detected. Try rotation symmetry detection or specify symmetry manually.';
    }

    if (state.reflectionStatus === 'ready') {
      return 'Select a reflection plane in the viewer or edit its normal and center, then visualize the specified symmetry.';
    }
  }

  if (state.majorStatus === 'idle') {
    return 'Press Detect major rotation axis or Detect reflection planes.';
  }

  if (state.majorStatus === 'empty') {
    return 'No rotation symmetry detected. Try reflection plane detection or specify symmetry manually.';
  }

  if (state.family === 'axial' && state.finerStatus === 'idle') {
    return 'Review the major rotation axis, or press Detect finer type to find C2 axes and mirror planes.';
  }

  if (state.family === 'T' || state.family === 'O' || state.family === 'I') {
    const secondaryFold = familySecondaryFold(state.family);

    if (state.selectableOverlayIds.length > 0) {
      return `Select a non-primary C${secondaryFold} axis in the viewer, or keep the current minor axis. Then visualize the specified symmetry.`;
    }

    return `No valid secondary C${secondaryFold} axis is available. Edit the minor axis manually, then visualize the specified symmetry.`;
  }

  if (state.c2AxesPerpendicularToAxis.length > 0) {
    return 'Select a C2 axis in the viewer or keep the current one. Then visualize the specified symmetry.';
  }

  if (state.reflectionPlanesContainingAxis.length > 0) {
    return 'Select a mirror plane in the viewer or keep the current one. Then visualize the specified symmetry.';
  }

  if (canProposeSymmetry(state)) {
    return 'Visualize the specified symmetry when the current parameters are ready.';
  }

  return 'Review point group type and axis values.';
}

export function detectionReducer(state: DetectionState, action: DetectionAction): DetectionState {
  if (action.type === 'reset') {
    return initialDetectionState;
  }

  if (action.type === 'detectionRestored') {
    let restored = initialDetectionState;

    if (action.reflectionAction) {
      restored = detectionReducer(restored, {
        actionKey: action.reflectionAction.key,
        candidates: action.reflectionAction.jsonResult as ReflectionPlaneDetectionResult[],
        type: 'reflectionPlanesLoaded',
      });
    } else if (action.rotationAction) {
      restored = detectionReducer(restored, {
        actionKey: action.rotationAction.key,
        candidates: action.rotationAction.jsonResult as RotationAxisDetectionResult[],
        type: 'rotationAxesLoaded',
      });

      if (action.finerAction) {
        restored = {
          ...restored,
          center: action.finerAction.params.center as Vector3,
          majorAxis: action.finerAction.params.majorAxis as Vector3,
        };
        restored = detectionReducer(restored, {
          actionKey: action.finerAction.key,
          result: action.finerAction.jsonResult as FinerSymmetryDetectionResult,
          type: 'finerResultLoaded',
        });
      }
    }

    if (action.confirmedSymmetry) {
      let family: SymmetryFamily = 'axial';
      let fold = Number.parseInt(action.confirmedSymmetry.label.slice(1), 10);

      if (action.confirmedSymmetry.label === 'S1') {
        fold = 1;
      } else if (action.confirmedSymmetry.label.startsWith('T')) {
        family = 'T';
        fold = 3;
      } else if (action.confirmedSymmetry.label.startsWith('O')) {
        family = 'O';
        fold = 4;
      } else if (action.confirmedSymmetry.label.startsWith('I')) {
        family = 'I';
        fold = 5;
      } else if (action.confirmedSymmetry.label.startsWith('S')) {
        fold /= 2;
      }

      restored = {
        ...restored,
        activeDetectionKind:
          action.confirmedSymmetry.label === 'S1' ? 'reflection' : 'rotation',
        center: action.confirmedSymmetry.center,
        confirmationError: '',
        confirming: false,
        family,
        fold,
        labels:
          action.confirmedSymmetry.label === 'S1'
            ? ['S1']
            : labelsForFamily(family, fold),
        majorAxis: action.confirmedSymmetry.majorAxis,
        minorAxis: action.confirmedSymmetry.minorAxis,
        proposedSymmetry: action.confirmedSymmetry,
        reflectionCenter: action.confirmedSymmetry.center,
        reflectionNormal: action.confirmedSymmetry.majorAxis,
        selectedLabel: action.confirmedSymmetry.label,
        symmetryPreview: action.confirmedSymmetry,
      };
    }

    return restored;
  }

  if (action.type === 'confirmationStarted') {
    return { ...state, confirmationError: '', confirming: true };
  }

  if (action.type === 'confirmationFailed') {
    return { ...state, confirmationError: action.message, confirming: false };
  }

  if (action.type === 'confirmationCompleted') {
    return { ...state, confirmationError: '', confirming: false };
  }

  if (state.confirmationError) {
    state = { ...state, confirmationError: '' };
  }

  if (action.type === 'majorDetectionStarted') {
    return {
      ...state,
      activeDetectionKind: 'rotation',
      c2AxesPerpendicularToAxis: [],
      center: [0, 0, 0],
      errorMessage: '',
      family: 'axial',
      finerActionKey: null,
      finerStatus: 'idle',
      fold: 1,
      labels: labelsForFamily('axial', 1),
      majorAxis: [0, 0, 1],
      majorStatus: 'running',
      overlays: [],
      proposedSymmetry: null,
      reflectionPlanesContainingAxis: [],
      reflectionPlanesPerpendicularToAxis: [],
      rotationActionKey: null,
      rotationAxes: [],
      selectableOverlayIds: [],
      selectedLabel: 'C1',
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
        activeDetectionKind: 'rotation',
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

    const rotationAxes: RotationAxisCandidate[] = action.candidates.map((candidate, index) => ({
      axis: candidate.axis,
      center: candidate.q,
      color: `hsl(${(index * 137.508) % 360}, 72%, 48%)`,
      dbscanLabel: candidate.dbscan_label,
      foldE: candidate.fold_e,
      foldI: candidate.fold_i,
      id: `${action.actionKey}:rotation:${index}`,
      ratio: candidate.ratio,
      rmse: candidate.rmse,
    }));
    const candidate = rotationAxes[0];
    const fold = candidate.foldI;

    return {
      ...state,
      activeDetectionKind: 'rotation',
      center: normalizeCenterInput(),
      errorMessage: '',
      family: 'axial',
      finerActionKey: null,
      finerStatus: 'idle',
      fold,
      labels: labelsForFamily('axial', fold),
      majorAxis: normalizeAxisInput(candidate.axis),
      majorStatus: 'ready',
      overlays: rotationAxes.map(rotationAxisOverlay),
      proposedSymmetry: null,
      rotationActionKey: action.actionKey,
      rotationAxes,
      selectableOverlayIds: rotationAxes.map((item) => item.id),
      selectedLabel: `C${fold}`,
      selectedMajorCandidateId: candidate.id,
      selectedMinorItemId: null,
      selectedOverlayId: candidate.id,
      symmetryPreview: null,
    };
  }

  if (action.type === 'majorDetectionFailed') {
    return {
      ...state,
      activeDetectionKind: 'rotation',
      c2AxesPerpendicularToAxis: [],
      errorMessage: action.message,
      family: 'axial',
      finerActionKey: null,
      finerStatus: 'idle',
      labels: labelsForFamily('axial', 1),
      majorStatus: 'failed',
      overlays: [],
      proposedSymmetry: null,
      reflectionPlanesContainingAxis: [],
      reflectionPlanesPerpendicularToAxis: [],
      rotationActionKey: null,
      rotationAxes: [],
      selectableOverlayIds: [],
      selectedLabel: 'C1',
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
    return {
      ...state,
      activeDetectionKind: 'rotation',
      majorAxis: action.axis,
      proposedSymmetry: null,
      symmetryPreview: null,
    };
  }

  if (action.type === 'centerChanged') {
    return {
      ...state,
      activeDetectionKind: 'rotation',
      center: action.center,
      proposedSymmetry: null,
      symmetryPreview: null,
    };
  }

  if (action.type === 'foldChanged') {
    const labels = state.family ? labelsForFamily(state.family, action.fold) : state.labels;

    return {
      ...state,
      activeDetectionKind: 'rotation',
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
      activeDetectionKind: 'rotation',
      majorAxis: normalizeAxisInput(state.majorAxis),
      proposedSymmetry: null,
      symmetryPreview: null,
    };
  }

  if (action.type === 'centerNormalized') {
    return {
      ...state,
      activeDetectionKind: 'rotation',
      center: normalizeCenterInput(),
      proposedSymmetry: null,
      symmetryPreview: null,
    };
  }

  if (action.type === 'familyPicked') {
    return pickFamily(state, action.family);
  }

  if (action.type === 'finerDetectionStarted') {
    const majorCandidate = state.rotationAxes.find(
      (candidate) => candidate.id === state.selectedMajorCandidateId,
    );

    return {
      ...state,
      activeDetectionKind: 'rotation',
      c2AxesPerpendicularToAxis: [],
      errorMessage: '',
      finerActionKey: null,
      finerStatus: 'running',
      overlays: majorCandidate ? [rotationAxisOverlay(majorCandidate)] : [],
      proposedSymmetry: null,
      reflectionPlanesContainingAxis: [],
      reflectionPlanesPerpendicularToAxis: [],
      selectableOverlayIds: [],
      selectedMinorItemId: null,
      selectedOverlayId: majorCandidate?.id ?? null,
      symmetryPreview: null,
    };
  }

  if (action.type === 'finerResultLoaded') {
    return loadFinerResult(state, action.result, action.actionKey);
  }

  if (action.type === 'finerDetectionFailed') {
    return {
      ...state,
      activeDetectionKind: 'rotation',
      c2AxesPerpendicularToAxis: [],
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

  if (action.type === 'reflectionDetectionStarted') {
    return {
      ...state,
      activeDetectionKind: 'reflection',
      errorMessage: '',
      overlays: [],
      proposedSymmetry: null,
      reflectionActionKey: null,
      reflectionStatus: 'running',
      selectedOverlayId: null,
      selectableOverlayIds: [],
      standaloneReflectionPlanes: [],
      symmetryPreview: null,
    };
  }

  if (action.type === 'reflectionPlanesLoaded') {
    if (action.candidates.length === 0) {
      return {
        ...state,
        activeDetectionKind: 'reflection',
        overlays: [],
        proposedSymmetry: null,
        reflectionActionKey: action.actionKey,
        reflectionStatus: 'empty',
        selectedOverlayId: null,
        selectableOverlayIds: [],
        standaloneReflectionPlanes: [],
        symmetryPreview: null,
      };
    }

    const standaloneReflectionPlanes: ReflectionDetectionCandidate[] = action.candidates.map(
      (candidate, index) => ({
        center: [
          candidate.n[0] * candidate.c,
          candidate.n[1] * candidate.c,
          candidate.n[2] * candidate.c,
        ],
        color: `hsl(${(index * 137.508) % 360}, 72%, 48%)`,
        dbscanLabel: candidate.dbscan_label,
        foldIValidation: candidate.fold_i_val,
        id: `${action.actionKey}:reflection:${index}`,
        normal: candidate.n,
        ratio: candidate.ratio,
        rmse: candidate.rmse,
      }),
    );
    const candidate = standaloneReflectionPlanes[0];
    const overlays: SymmetryOverlay[] = standaloneReflectionPlanes.map((plane) => ({
      center: plane.center,
      color: plane.color,
      id: plane.id,
      kind: 'reflection_plane',
      label: 'mirror',
      majorAxis: [0, 0, 1],
      normal: plane.normal,
      role: 'perpendicular_to_major_axis',
      shape: 'disk',
    }));

    return {
      ...state,
      activeDetectionKind: 'reflection',
      errorMessage: '',
      overlays,
      proposedSymmetry: null,
      reflectionActionKey: action.actionKey,
      reflectionCenter: normalizeCenterInput(),
      reflectionNormal: normalizeAxisInput(candidate.normal),
      reflectionStatus: 'ready',
      selectedOverlayId: candidate.id,
      selectableOverlayIds: standaloneReflectionPlanes.map((plane) => plane.id),
      standaloneReflectionPlanes,
      symmetryPreview: null,
    };
  }

  if (action.type === 'reflectionDetectionFailed') {
    return {
      ...state,
      activeDetectionKind: 'reflection',
      errorMessage: action.message,
      overlays: [],
      proposedSymmetry: null,
      reflectionActionKey: null,
      reflectionStatus: 'failed',
      selectedOverlayId: null,
      selectableOverlayIds: [],
      standaloneReflectionPlanes: [],
      symmetryPreview: null,
    };
  }

  if (action.type === 'reflectionNormalChanged') {
    return {
      ...state,
      activeDetectionKind: 'reflection',
      proposedSymmetry: null,
      reflectionNormal: action.normal,
      symmetryPreview: null,
    };
  }

  if (action.type === 'reflectionCenterChanged') {
    return {
      ...state,
      activeDetectionKind: 'reflection',
      proposedSymmetry: null,
      reflectionCenter: action.center,
      symmetryPreview: null,
    };
  }

  if (action.type === 'reflectionNormalNormalized') {
    return {
      ...state,
      activeDetectionKind: 'reflection',
      proposedSymmetry: null,
      reflectionNormal: normalizeAxisInput(state.reflectionNormal),
      symmetryPreview: null,
    };
  }

  if (action.type === 'reflectionCenterNormalized') {
    return {
      ...state,
      activeDetectionKind: 'reflection',
      proposedSymmetry: null,
      reflectionCenter: normalizeCenterInput(),
      symmetryPreview: null,
    };
  }

  if (action.type === 'labelPicked') {
    return {
      ...state,
      activeDetectionKind: 'rotation',
      proposedSymmetry: null,
      selectedLabel: action.label,
      symmetryPreview: null,
    };
  }

  if (action.type === 'minorAxisChanged') {
    return {
      ...state,
      activeDetectionKind: 'rotation',
      minorAxis: action.axis,
      proposedSymmetry: null,
      symmetryPreview: null,
    };
  }

  if (action.type === 'overlayPicked') {
    return pickOverlay(state, action.overlayId);
  }

  if (action.type === 'minorAxisNormalized') {
    return {
      ...state,
      activeDetectionKind: 'rotation',
      minorAxis: normalizeAxisInput(state.minorAxis),
      proposedSymmetry: null,
      symmetryPreview: null,
    };
  }

  let proposedSymmetry: ProposedSymmetry;

  if (state.activeDetectionKind === 'reflection') {
    const majorAxis = normalizeAxisInput(state.reflectionNormal);
    const referenceAxis: Vector3 = Math.abs(majorAxis[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];

    proposedSymmetry = {
      center: [
        state.reflectionCenter[0],
        state.reflectionCenter[1],
        state.reflectionCenter[2],
      ],
      label: 'S1',
      majorAxis,
      minorAxis: minorAxisFromSecondaryAxis(referenceAxis, majorAxis),
    };
  } else {
    proposedSymmetry = {
      center: [state.center[0], state.center[1], state.center[2]],
      label: state.selectedLabel,
      majorAxis: [state.majorAxis[0], state.majorAxis[1], state.majorAxis[2]],
      minorAxis: [state.minorAxis[0], state.minorAxis[1], state.minorAxis[2]],
    };
  }

  return {
    ...state,
    c2AxesPerpendicularToAxis: [],
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
    activeDetectionKind: 'rotation',
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
    const majorCandidate = state.rotationAxes.find(
      (candidate) => candidate.id === state.selectedMajorCandidateId,
    );
    const secondaryCandidates = state.rotationAxes.filter((candidate) => {
      const axis = normalizeAxisInput(candidate.axis);
      const dot = Math.abs(axis[0] * major[0] + axis[1] * major[1] + axis[2] * major[2]);

      return (
        candidate.id !== state.selectedMajorCandidateId &&
        candidate.foldI === secondaryFold &&
        dot < secondaryAxisParallelThreshold
      );
    });
    const selectedSecondary = secondaryCandidates[0];

    return {
      ...state,
      activeDetectionKind: 'rotation',
      c2AxesPerpendicularToAxis: [],
      family,
      finerStatus: 'idle',
      labels,
      minorAxis: selectedSecondary
        ? minorAxisFromSecondaryAxis(selectedSecondary.axis, state.majorAxis)
        : state.minorAxis,
      overlays: [
        ...(majorCandidate ? [rotationAxisOverlay(majorCandidate)] : []),
        ...secondaryCandidates.map(rotationAxisOverlay),
      ],
      proposedSymmetry: null,
      reflectionPlanesContainingAxis: [],
      reflectionPlanesPerpendicularToAxis: [],
      selectableOverlayIds: secondaryCandidates.map((candidate) => candidate.id),
      selectedLabel: labels[0],
      selectedMinorItemId: selectedSecondary?.id ?? null,
      selectedOverlayId: selectedSecondary?.id ?? majorCandidate?.id ?? null,
      symmetryPreview: null,
    };
  }

  return {
    ...state,
    activeDetectionKind: 'rotation',
    c2AxesPerpendicularToAxis: [],
    family,
    finerStatus: 'idle',
    labels,
    overlays: state.rotationAxes.map(rotationAxisOverlay),
    proposedSymmetry: null,
    reflectionPlanesContainingAxis: [],
    reflectionPlanesPerpendicularToAxis: [],
    selectableOverlayIds: state.rotationAxes.map((candidate) => candidate.id),
    selectedLabel: labels[0],
    selectedMinorItemId: null,
    selectedOverlayId: state.selectedMajorCandidateId,
    symmetryPreview: null,
  };
}

function loadFinerResult(
  state: DetectionState,
  result: FinerSymmetryDetectionResult,
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
  const c2AxesPerpendicularToAxis: C2AxisCandidate[] = result.c2_axes_perpendicular_to_axis.map(
    (axis, index) => ({
      axis: axis.axis,
      axisCorrected: axis.axis_cor,
      center: axis.q,
      centerCorrected: axis.q_cor,
      color: `hsl(${(index * 137.508) % 360}, 72%, 48%)`,
      dbscanLabel: axis.dbscan_label,
      foldC2: axis.fold_c2,
      foldIValidation: axis.fold_i_val,
      id: `${actionKey}:c2:${index}`,
      ratio: axis.ratio,
      rmse: axis.rmse,
    }),
  );
  const reflectionPlanesContainingAxis: ReflectionPlaneCandidate[] =
    result.reflection_planes_containing_axis.map((plane, index) => ({
      color: `hsl(${((index + c2AxesPerpendicularToAxis.length) * 137.508) % 360}, 72%, 48%)`,
      dbscanLabel: plane.dbscan_label,
      foldIValidation: plane.fold_i_val,
      foldPred: plane.fold_pred,
      id: `${actionKey}:plane-containing:${index}`,
      normal: plane.n,
      normalCorrected: plane.n_cor,
      ratio: plane.ratio,
      rmse: plane.rmse,
      role: 'contains_major_axis',
    }));
  const reflectionPlanesPerpendicularToAxis: ReflectionPlaneCandidate[] =
    result.reflection_planes_perpendicular_to_axis.map((plane, index) => ({
      color: `hsl(${
        ((index + c2AxesPerpendicularToAxis.length + reflectionPlanesContainingAxis.length) *
          137.508) %
        360
      }, 72%, 48%)`,
      dbscanLabel: plane.dbscan_label,
      foldIValidation: plane.fold_i_val,
      id: `${actionKey}:plane-perpendicular:${index}`,
      normal: plane.n,
      normalCorrected: plane.n_cor,
      ratio: plane.ratio,
      rmse: plane.rmse,
      role: 'perpendicular_to_major_axis',
    }));
  const c2Overlays: SymmetryOverlay[] = c2AxesPerpendicularToAxis.map((axis) => ({
    axis: axis.axisCorrected,
    center: axis.centerCorrected,
    color: axis.color,
    fold: 2,
    id: axis.id,
    kind: 'c2_axis',
    label: 'C2',
  }));
  const planeOverlays: SymmetryOverlay[] = [
    ...reflectionPlanesContainingAxis.map((plane) => reflectionPlaneOverlay(plane, state)),
    ...reflectionPlanesPerpendicularToAxis.map((plane) => reflectionPlaneOverlay(plane, state)),
  ];
  const verticalPlane = reflectionPlanesContainingAxis[0];
  const horizontalPlane = reflectionPlanesPerpendicularToAxis[0];
  const c2Axis = c2AxesPerpendicularToAxis[0];
  const selectedMinorItemId = c2Axis?.id ?? verticalPlane?.id ?? null;
  const selectableOverlayIds =
    c2AxesPerpendicularToAxis.length > 0
      ? c2AxesPerpendicularToAxis.map((axis) => axis.id)
      : reflectionPlanesContainingAxis.map((plane) => plane.id);
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
    activeDetectionKind: 'rotation',
    c2AxesPerpendicularToAxis,
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
    reflectionPlanesContainingAxis,
    reflectionPlanesPerpendicularToAxis,
    selectableOverlayIds,
    selectedLabel,
    selectedMinorItemId,
    selectedOverlayId: selectedMinorItemId ?? majorOverlay.id,
    symmetryPreview: null,
  };
}

function pickOverlay(state: DetectionState, overlayId: string): DetectionState {
  const standalonePlane = state.standaloneReflectionPlanes.find((item) => item.id === overlayId);

  if (state.activeDetectionKind === 'reflection' && standalonePlane) {
    return {
      ...state,
      reflectionCenter: standalonePlane.center,
      reflectionNormal: standalonePlane.normal,
      proposedSymmetry: null,
      selectedOverlayId: overlayId,
      symmetryPreview: null,
    };
  }

  const c2Axis = state.c2AxesPerpendicularToAxis.find((item) => item.id === overlayId);
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
        activeDetectionKind: 'rotation',
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
      activeDetectionKind: 'rotation',
      minorAxis: normalizeAxisInput(c2Axis.axisCorrected),
      proposedSymmetry: null,
      selectedMinorItemId: overlayId,
      selectedOverlayId: overlayId,
      symmetryPreview: null,
    };
  }

  if (verticalPlane && state.c2AxesPerpendicularToAxis.length === 0) {
    return {
      ...state,
      activeDetectionKind: 'rotation',
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
    shape: plane.role === 'perpendicular_to_major_axis' ? 'disk' : 'square',
  };
}
