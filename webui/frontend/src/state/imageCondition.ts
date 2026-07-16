import type { NodeRunResult, UploadRef } from '../types';

export type ImageConditionStatus = 'idle' | 'uploading' | 'generating' | 'ready' | 'failed';

export type ImageConditionState = {
  errorMessage: string;
  file: Blob | File | null;
  previewName: string;
  previewUrl: string;
  run: NodeRunResult | null;
  status: ImageConditionStatus;
  upload: UploadRef | null;
};

export type ImageConditionAction =
  | {
      file: Blob | File;
      name: string;
      type: 'imageSelected';
      url: string;
    }
  | {
      type: 'conditionGenerationStarted';
    }
  | {
      type: 'inputUploaded';
      upload: UploadRef;
    }
  | {
      run: NodeRunResult;
      type: 'conditionGenerated';
    }
  | {
      filename: string;
      previewUrl: string;
      run: NodeRunResult;
      type: 'conditionRestored';
    }
  | {
      message: string;
      type: 'conditionGenerationFailed';
    }
  | {
      type: 'resetToNodeStart';
    }
  | {
      type: 'resetSession';
    };

export function createInitialImageConditionState(): ImageConditionState {
  return {
    errorMessage: '',
    file: null,
    previewName: '',
    previewUrl: '',
    run: null,
    status: 'idle',
    upload: null,
  };
}

export function imageConditionReducer(
  state: ImageConditionState,
  action: ImageConditionAction,
): ImageConditionState {
  switch (action.type) {
    case 'imageSelected':
      return {
        errorMessage: '',
        file: action.file,
        previewName: action.name,
        previewUrl: action.url,
        run: null,
        status: 'idle',
        upload: null,
      };

    case 'conditionGenerationStarted':
      return {
        ...state,
        errorMessage: '',
        run: null,
        status: 'uploading',
        upload: null,
      };

    case 'inputUploaded':
      return {
        ...state,
        status: 'generating',
        upload: action.upload,
      };

    case 'conditionGenerated':
      return {
        ...state,
        errorMessage: '',
        run: action.run,
        status: 'ready',
      };

    case 'conditionRestored':
      return {
        errorMessage: '',
        file: null,
        previewName: action.filename,
        previewUrl: action.previewUrl,
        run: action.run,
        status: 'ready',
        upload: null,
      };

    case 'conditionGenerationFailed':
      return {
        ...state,
        errorMessage: action.message,
        status: 'failed',
      };

    case 'resetToNodeStart':
      return {
        ...state,
        errorMessage: '',
        run: null,
        status: 'idle',
        upload: null,
      };

    case 'resetSession':
      return createInitialImageConditionState();
  }
}

export function imageConditionInstruction(state: ImageConditionState): string {
  if (state.status === 'ready') {
    return 'Image condition is ready. Choose the next route.';
  }

  if (!state.file) {
    return 'Choose, drop, or paste an input image.';
  }

  if (state.status === 'uploading') {
    return 'Uploading image.';
  }

  if (state.status === 'generating') {
    return 'Generating image condition.';
  }

  if (state.status === 'failed') {
    return state.errorMessage;
  }

  return 'Generate image condition before choosing the next route.';
}
