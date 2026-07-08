import type { Dispatch } from 'react';
import type {
  Trellis2SymmetrySparseStructureAction,
  Trellis2SymmetrySparseStructureState,
} from '../models/trellis2';
import type { SymmetryTuple } from '../types';
import {
  GenerationActions,
  GenerationParameters,
  GenerationStatusBlock,
  SymmetryGenerationParameters,
} from './generationControls';
import { ProposedSymmetryBlock } from './symmetryControls';

type Trellis2SymmetrySparseStructurePanelProps = {
  dispatch: Dispatch<Trellis2SymmetrySparseStructureAction>;
  nextActions: Array<{ label: string; onClick: () => void }>;
  onGenerate: () => void;
  state: Trellis2SymmetrySparseStructureState;
  symmetryTuple: SymmetryTuple | null;
};

function trellis2SymmetrySparseStructureInstruction(
  state: Trellis2SymmetrySparseStructureState,
): string {
  if (state.run.status === 'running') {
    return 'Generating symmetry enforced sparse structure. Progress follows backend flow time.';
  }

  if (state.run.status === 'ready' && state.metadata.voxelCount === 0) {
    return 'Generated occupancy is empty. Change parameters and generate again.';
  }

  if (state.run.status === 'ready') {
    return 'Sparse structure is ready. Review the voxel count, then go to the next step.';
  }

  if (state.run.status === 'failed') {
    return state.run.errorMessage || 'Symmetry enforced sparse structure generation failed.';
  }

  return 'Review the locked symmetry tuple and sparse-structure parameters, then press Confirm and generate.';
}

export function Trellis2SymmetrySparseStructurePanel({
  dispatch,
  nextActions,
  onGenerate,
  state,
  symmetryTuple,
}: Trellis2SymmetrySparseStructurePanelProps) {
  const running = state.run.status === 'running';
  const readyForNext = state.run.status === 'ready' && state.metadata.voxelCount > 0;

  return (
    <div className="node-panel-stack">
      <ProposedSymmetryBlock symmetry={symmetryTuple} />

      <SymmetryGenerationParameters
        disabled={running}
        onParamsChange={(params) => dispatch({ params, type: 'paramsChanged' })}
        params={state.params}
      />

      <div className="node-section-divider" aria-hidden="true" />

      <GenerationParameters
        disabled={running}
        onParamsChange={(params) => dispatch({ params, type: 'paramsChanged' })}
        onSeedRandomized={() => dispatch({ type: 'seedRandomized' })}
        params={state.params}
      />

      <GenerationStatusBlock
        metadata={[{ label: 'voxel count', value: state.metadata.voxelCount }]}
        progress={state.run.progress}
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

Trellis2SymmetrySparseStructurePanel.instruction = trellis2SymmetrySparseStructureInstruction;
