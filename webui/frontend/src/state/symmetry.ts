import type { SymmetryFamily, SymmetryTuple, Vector3 } from '../types';

export type ProposedSymmetry = SymmetryTuple;
export type ManualSymmetryFamily = 'reflection' | SymmetryFamily;

export type ManualSymmetryState = {
  center: Vector3;
  confirmationError: string;
  confirming: boolean;
  family: ManualSymmetryFamily;
  fold: number;
  labels: string[];
  majorAxis: Vector3;
  minorAxis: Vector3;
  proposedSymmetry: ProposedSymmetry | null;
  selectedLabel: string;
  symmetryPreview: SymmetryTuple | null;
};

export type ManualSymmetryAction =
  | { axis: Vector3; type: 'majorAxisChanged' }
  | { axis: Vector3; type: 'minorAxisChanged' }
  | { center: Vector3; type: 'centerChanged' }
  | { axis: Vector3; type: 'majorAxisShortcutPicked' }
  | { axis: Vector3; type: 'minorAxisShortcutPicked' }
  | { family: ManualSymmetryFamily; type: 'familyPicked' }
  | { fold: number; type: 'foldChanged' }
  | { label: string; type: 'labelPicked' }
  | { type: 'proposeSymmetry' }
  | { type: 'confirmationStarted' }
  | { message: string; type: 'confirmationFailed' }
  | { type: 'confirmationCompleted' }
  | { type: 'reset' };

export const rotationSymmetryFamilies: SymmetryFamily[] = ['axial', 'T', 'O', 'I'];
export const manualSymmetryFamilies: ManualSymmetryFamily[] = [
  'reflection',
  ...rotationSymmetryFamilies,
];

export const axisShortcuts: Array<{ axis: Vector3; label: string }> = [
  { axis: [1, 0, 0], label: 'X' },
  { axis: [0, 1, 0], label: 'Y' },
  { axis: [0, 0, 1], label: 'Z' },
];

export const secondaryAxisParallelThreshold = 0.98;

export const initialManualSymmetryState: ManualSymmetryState = {
  center: [0, 0, 0],
  confirmationError: '',
  confirming: false,
  family: 'axial',
  fold: 2,
  labels: labelsForFamily('axial', 2),
  majorAxis: [0, 0, 1],
  minorAxis: [1, 0, 0],
  proposedSymmetry: null,
  selectedLabel: 'C2',
  symmetryPreview: null,
};

export function labelsForFamily(family: ManualSymmetryFamily, fold: number) {
  if (family === 'reflection') {
    return ['S1'];
  }

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

export function axisShortcutDisabled(axis: Vector3, majorAxis: Vector3) {
  const shortcut = normalizeAxisInput(axis);
  const major = normalizeAxisInput(majorAxis);
  const dot = Math.abs(shortcut[0] * major[0] + shortcut[1] * major[1] + shortcut[2] * major[2]);

  return dot > secondaryAxisParallelThreshold;
}

export function familySecondaryFold(family: SymmetryFamily) {
  if (family === 'T') {
    return 3;
  }

  if (family === 'O') {
    return 4;
  }

  if (family === 'I') {
    return 5;
  }

  return 2;
}

export function vectorWithValue(vector: Vector3, index: number, value: number): Vector3 {
  return [
    index === 0 ? value : vector[0],
    index === 1 ? value : vector[1],
    index === 2 ? value : vector[2],
  ];
}

export function formatVector(vector: Vector3) {
  return `[${vector.map((value) => Number(value.toFixed(4)).toString()).join(', ')}]`;
}

export function canProposeManualSymmetry(state: ManualSymmetryState) {
  return Boolean(state.selectedLabel && state.family);
}

export function manualSymmetryInstruction(state: ManualSymmetryState): string {
  if (state.confirming) {
    return 'Confirming symmetry.';
  }

  if (state.confirmationError) {
    return state.confirmationError;
  }

  if (state.proposedSymmetry) {
    return 'Review the locked symmetry tuple and viewer preview, then press Confirm.';
  }

  if (state.family === 'reflection') {
    return 'Set the mirror-plane normal and center, then visualize the specified symmetry.';
  }

  if (state.family === 'axial') {
    return 'Set the principal axis, center, fold, and point group type, then press Confirm proposed symmetry.';
  }

  return 'Set the major and minor axes for the polyhedral frame, then press Confirm proposed symmetry.';
}

export function manualSymmetryReducer(
  state: ManualSymmetryState,
  action: ManualSymmetryAction,
): ManualSymmetryState {
  if (action.type === 'reset') {
    return initialManualSymmetryState;
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

  if (action.type === 'majorAxisChanged') {
    return { ...state, majorAxis: action.axis, proposedSymmetry: null, symmetryPreview: null };
  }

  if (action.type === 'minorAxisChanged') {
    return { ...state, minorAxis: action.axis, proposedSymmetry: null, symmetryPreview: null };
  }

  if (action.type === 'centerChanged') {
    return { ...state, center: action.center, proposedSymmetry: null, symmetryPreview: null };
  }

  if (action.type === 'majorAxisShortcutPicked') {
    return { ...state, majorAxis: action.axis, proposedSymmetry: null, symmetryPreview: null };
  }

  if (action.type === 'minorAxisShortcutPicked') {
    return { ...state, minorAxis: action.axis, proposedSymmetry: null, symmetryPreview: null };
  }

  if (action.type === 'familyPicked') {
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

  if (action.type === 'foldChanged') {
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

  if (action.type === 'labelPicked') {
    return { ...state, proposedSymmetry: null, selectedLabel: action.label, symmetryPreview: null };
  }

  const proposedSymmetry: ProposedSymmetry = {
    center: [state.center[0], state.center[1], state.center[2]],
    label: state.selectedLabel,
    majorAxis: normalizeAxisInput(state.majorAxis),
    minorAxis: normalizeAxisInput(state.minorAxis),
  };

  return { ...state, proposedSymmetry, symmetryPreview: proposedSymmetry };
}
