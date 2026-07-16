import type { Dispatch } from 'react';
import type {
  Trellis2VanillaSparseStructureAction,
  Trellis2VanillaSparseStructureState,
} from '../models/trellis2';
import {
  GenerationActions,
  GenerationParameters,
  GenerationStatusBlock,
} from './generationControls';

type Trellis2VanillaSparseStructurePanelProps = {
  dispatch: Dispatch<Trellis2VanillaSparseStructureAction>;
  nextActions: Array<{ label: string; onClick: () => void }>;
  onGenerate: () => void;
  state: Trellis2VanillaSparseStructureState;
};

function trellis2VanillaSparseStructureInstruction(
  state: Trellis2VanillaSparseStructureState,
): string {
  if (state.run.status === 'running') {
    return 'Generating vanilla sparse structure. Progress follows backend flow time.';
  }

  if (state.run.status === 'ready' && state.metadata.voxelCount === 0) {
    return 'Generated occupancy is empty. Change parameters and generate again.';
  }

  if (state.run.status === 'ready') {
    return 'Sparse structure is ready. Review the voxel count, then go to the next step.';
  }

  if (state.run.status === 'failed') {
    return state.run.errorMessage || 'Vanilla sparse structure generation failed.';
  }

  return 'Review vanilla sparse-structure parameters, then press Confirm and generate.';
}

export function Trellis2VanillaSparseStructurePanel({
  dispatch,
  nextActions,
  onGenerate,
  state,
}: Trellis2VanillaSparseStructurePanelProps) {
  const running = state.run.status === 'running';
  const readyForNext = state.run.status === 'ready' && state.metadata.voxelCount > 0;

  return (
    <div className="node-panel-stack">
      <GenerationParameters
        disabled={running}
        onParamsChange={(params) => dispatch({ params, type: 'paramsChanged' })}
        onSeedRandomized={() => dispatch({ type: 'seedRandomized' })}
        params={state.params}
      />

      <GenerationStatusBlock
        metadata={[{ label: 'voxel count', value: state.metadata.voxelCount }]}
        progress={state.run.progress}
        stage={state.run.stage}
        status={state.run.status}
      />

      <GenerationActions
        nextActions={nextActions}
        onGenerate={onGenerate}
        readyForNext={readyForNext}
        running={running}
      />
    </div>
  );
}

Trellis2VanillaSparseStructurePanel.instruction = trellis2VanillaSparseStructureInstruction;
