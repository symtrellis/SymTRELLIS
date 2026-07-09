import type { NodeRunResult, RequestId } from '../types';

export type DurationRange = [number, number];

export type CommonGenerationParams = {
  cfgDuration: DurationRange;
  cfgRescale: number;
  cfgStrength: number;
  seed: number;
  steps: number;
  timeStepRescale: number;
};

export type SymmetryProjectionParams = {
  noiseSymmetryProjectionStrength: number;
  symmetryProjectionDuration: DurationRange;
  symmetryProjectionStrength: number;
};

export type GenerationStatus = 'idle' | 'running' | 'ready' | 'failed';

export type GenerationRunState = {
  errorMessage: string;
  progress: number;
  requestId: RequestId | null;
  result: NodeRunResult | null;
  status: GenerationStatus;
};

export type GenerationState<Params extends CommonGenerationParams, Metadata> = {
  metadata: Metadata;
  params: Params;
  run: GenerationRunState;
};

export type GenerationAction<Params extends CommonGenerationParams, Metadata> =
  | {
      params: Partial<Params>;
      type: 'paramsChanged';
    }
  | {
      type: 'seedRandomized';
    }
  | {
      requestId: RequestId;
      type: 'generationStarted';
    }
  | {
      progress: number;
      requestId: RequestId;
      type: 'generationProgressUpdated';
    }
  | {
      metadata: Metadata;
      result: NodeRunResult;
      type: 'generationCompleted';
    }
  | {
      message: string;
      requestId: RequestId;
      type: 'generationFailed';
    }
  | {
      metadata: Metadata;
      params: Params;
      type: 'resetToNodeStart';
    };

export function createInitialGenerationState<Params extends CommonGenerationParams, Metadata>(
  params: Params,
  metadata: Metadata,
): GenerationState<Params, Metadata> {
  return {
    metadata,
    params,
    run: {
      errorMessage: '',
      progress: 0,
      requestId: null,
      result: null,
      status: 'idle',
    },
  };
}

export function generationReducer<Params extends CommonGenerationParams, Metadata>(
  state: GenerationState<Params, Metadata>,
  action: GenerationAction<Params, Metadata>,
): GenerationState<Params, Metadata> {
  switch (action.type) {
    case 'paramsChanged':
      return {
        ...state,
        params: {
          ...state.params,
          ...action.params,
        },
        run: {
          errorMessage: '',
          progress: 0,
          requestId: null,
          result: null,
          status: 'idle',
        },
      };

    case 'seedRandomized':
      return {
        ...state,
        params: {
          ...state.params,
          seed: Math.floor(Math.random() * 2147483647),
        },
        run: {
          errorMessage: '',
          progress: 0,
          requestId: null,
          result: null,
          status: 'idle',
        },
      };

    case 'generationStarted':
      return {
        ...state,
        run: {
          errorMessage: '',
          progress: 0,
          requestId: action.requestId,
          result: null,
          status: 'running',
        },
      };

    case 'generationProgressUpdated':
      if (action.requestId !== state.run.requestId) {
        return state;
      }

      return {
        ...state,
        run: {
          ...state.run,
          progress: Math.min(1, Math.max(0, action.progress)),
        },
      };

    case 'generationCompleted':
      return {
        ...state,
        metadata: action.metadata,
        run: {
          errorMessage: '',
          progress: 1,
          requestId: null,
          result: action.result,
          status: 'ready',
        },
      };

    case 'generationFailed':
      if (action.requestId !== state.run.requestId) {
        return state;
      }

      return {
        ...state,
        run: {
          errorMessage: action.message,
          progress: 0,
          requestId: null,
          result: null,
          status: 'failed',
        },
      };

    case 'resetToNodeStart':
      return createInitialGenerationState(action.params, action.metadata);
  }
}
