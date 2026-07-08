import type { ArtifactRef, NodeRunRef } from '../types';

export type ImageConditionStatus = 'idle' | 'uploading' | 'generating' | 'ready' | 'failed';

export type ImageConditionState = {
  conditionArtifact: ArtifactRef | null;
  errorMessage: string;
  inputArtifact: ArtifactRef | null;
  nodeRun: NodeRunRef | null;
  previewFile: Blob | File | null;
  previewName: string;
  previewUrl: string;
  status: ImageConditionStatus;
};

export type ImageConditionAction =
  | { file: Blob | File; name: string; type: 'imageSelected'; url: string }
  | { type: 'conditionGenerationStarted' }
  | { artifact: ArtifactRef; type: 'inputUploaded' }
  | { conditionArtifact: ArtifactRef; nodeRun: NodeRunRef; type: 'conditionGenerated' }
  | { message: string; type: 'conditionGenerationFailed' }
  | { type: 'conditionResultCleared' }
  | { type: 'reset' };

export const initialImageConditionState: ImageConditionState = {
  conditionArtifact: null,
  errorMessage: '',
  inputArtifact: null,
  nodeRun: null,
  previewFile: null,
  previewName: '',
  previewUrl: '',
  status: 'idle',
};

export function imageConditionReducer(
  state: ImageConditionState,
  action: ImageConditionAction,
): ImageConditionState {
  if (action.type === 'imageSelected') {
    return {
      ...state,
      conditionArtifact: null,
      errorMessage: '',
      inputArtifact: null,
      nodeRun: null,
      previewFile: action.file,
      previewName: action.name,
      previewUrl: action.url,
      status: 'idle',
    };
  }

  if (action.type === 'conditionGenerationStarted') {
    return {
      ...state,
      conditionArtifact: null,
      errorMessage: '',
      inputArtifact: null,
      nodeRun: null,
      status: 'uploading',
    };
  }

  if (action.type === 'inputUploaded') {
    return { ...state, inputArtifact: action.artifact, status: 'generating' };
  }

  if (action.type === 'conditionGenerated') {
    return {
      ...state,
      conditionArtifact: action.conditionArtifact,
      errorMessage: '',
      nodeRun: action.nodeRun,
      status: 'ready',
    };
  }

  if (action.type === 'conditionResultCleared') {
    return {
      ...state,
      conditionArtifact: null,
      errorMessage: '',
      nodeRun: null,
      status: 'idle',
    };
  }

  if (action.type === 'reset') {
    return initialImageConditionState;
  }

  return { ...state, errorMessage: action.message, status: 'failed' };
}

export function imageConditionInstruction(state: ImageConditionState) {
  if (!state.previewFile) {
    return 'Choose, drop, or paste an input image. The viewer remains available for camera checks.';
  }

  if (state.status === 'uploading') {
    return 'Uploading the selected image as an input artifact before condition generation.';
  }

  if (state.status === 'generating') {
    return 'Generating the image condition from the uploaded artifact.';
  }

  if (state.status === 'ready') {
    return 'Image condition is ready. Choose the next route in the workflow.';
  }

  if (state.status === 'failed') {
    return state.errorMessage || 'Image condition generation failed.';
  }

  return 'Generate the image condition, then choose manual symmetry or vanilla generation.';
}
