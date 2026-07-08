import type { ArtifactRef, NodeRunRef } from '../types';

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
  nodeRun: NodeRunRef | null;
  outputArtifact: ArtifactRef | null;
  progress: number;
  status: GenerationStatus;
};

export type GenerationState<Params extends CommonGenerationParams, Metadata> = {
  metadata: Metadata;
  params: Params;
  run: GenerationRunState;
};

export type GenerationAction<Params extends CommonGenerationParams, Metadata> =
  | { params: Partial<Params>; type: 'paramsChanged' }
  | { type: 'seedRandomized' }
  | { type: 'generationStarted' }
  | { progress: number; type: 'generationProgressed' }
  | {
      metadata: Metadata;
      nodeRun: NodeRunRef;
      outputArtifact: ArtifactRef | null;
      type: 'generationFinished';
    }
  | { message: string; type: 'generationFailed' }
  | { state: GenerationState<Params, Metadata>; type: 'reset' };

export function generationInitialState<Params extends CommonGenerationParams, Metadata>(
  params: Params,
  metadata: Metadata,
): GenerationState<Params, Metadata> {
  return {
    metadata,
    params,
    run: idleGenerationRun(),
  };
}

export function generationReducer<Params extends CommonGenerationParams, Metadata>(
  state: GenerationState<Params, Metadata>,
  action: GenerationAction<Params, Metadata>,
): GenerationState<Params, Metadata> {
  if (action.type === 'paramsChanged') {
    const params = { ...state.params, ...action.params };

    if (action.params.seed !== undefined) {
      params.seed = Math.trunc(action.params.seed);
    }

    if (action.params.steps !== undefined) {
      params.steps = Math.max(1, Math.trunc(action.params.steps));
    }

    return {
      ...state,
      params,
      run: idleGenerationRun(),
    };
  }

  if (action.type === 'seedRandomized') {
    return {
      ...state,
      params: {
        ...state.params,
        seed: Math.floor(Math.random() * 2147483647),
      },
      run: idleGenerationRun(),
    };
  }

  if (action.type === 'generationStarted') {
    return {
      ...state,
      run: {
        ...idleGenerationRun(),
        status: 'running',
      },
    };
  }

  if (action.type === 'generationProgressed') {
    return {
      ...state,
      run: {
        ...state.run,
        progress: action.progress,
      },
    };
  }

  if (action.type === 'generationFinished') {
    return {
      ...state,
      metadata: action.metadata,
      run: {
        errorMessage: '',
        nodeRun: action.nodeRun,
        outputArtifact: action.outputArtifact,
        progress: 1,
        status: 'ready',
      },
    };
  }

  if (action.type === 'generationFailed') {
    return {
      ...state,
      run: {
        ...state.run,
        errorMessage: action.message,
        progress: 0,
        status: 'failed',
      },
    };
  }

  return action.state;
}

function idleGenerationRun(): GenerationRunState {
  return {
    errorMessage: '',
    nodeRun: null,
    outputArtifact: null,
    progress: 0,
    status: 'idle',
  };
}
