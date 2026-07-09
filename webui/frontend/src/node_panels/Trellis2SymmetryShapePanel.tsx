import type { Dispatch } from 'react';
import type {
  Trellis2ExportParams,
  Trellis2SymmetryShapeAction,
  Trellis2SymmetryShapeState,
} from '../models/trellis2';
import type { SymmetryTuple } from '../types';
import type { ExportState } from './exportControls';
import { ExportControls } from './exportControls';
import {
  GenerationActions,
  GenerationParameters,
  GenerationStatusBlock,
  SymmetryGenerationParameters,
} from './generationControls';
import { ProposedSymmetryBlock } from './symmetryControls';
import { Trellis2ShapeModeControl } from './trellis2ShapeControls';

type Trellis2SymmetryShapePanelProps = {
  dispatch: Dispatch<Trellis2SymmetryShapeAction>;
  exportState: ExportState;
  nextActions: Array<{ label: string; onClick: () => void }>;
  onExport: () => void;
  onExportParamsChange: (params: Partial<Trellis2ExportParams>) => void;
  onGenerate: () => void;
  state: Trellis2SymmetryShapeState;
  symmetryTuple: SymmetryTuple | null;
};

function trellis2SymmetryShapeInstruction(state: Trellis2SymmetryShapeState): string {
  if (state.run.status === 'running') {
    return 'Generating symmetry enforced shape. The viewer shows the input occupancy until the mesh is ready.';
  }

  if (state.run.status === 'ready') {
    return 'Shape mesh is ready. Review grid metadata, then go to the next step.';
  }

  if (state.run.status === 'failed') {
    return state.run.errorMessage || 'Symmetry enforced shape generation failed.';
  }

  return 'Review the locked symmetry tuple and shape-generation parameters, then press Confirm and generate.';
}

export function Trellis2SymmetryShapePanel({
  dispatch,
  exportState,
  nextActions,
  onExport,
  onExportParamsChange,
  onGenerate,
  state,
  symmetryTuple,
}: Trellis2SymmetryShapePanelProps) {
  const running = state.run.status === 'running';
  const readyForNext = state.run.status === 'ready' && Boolean(state.run.result);

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

      <Trellis2ShapeModeControl
        disabled={running}
        onParamsChange={(params) => dispatch({ params, type: 'paramsChanged' })}
        params={state.params}
      />

      <GenerationStatusBlock
        metadata={[
          { label: 'voxel count', value: state.metadata.voxelCount },
          { label: 'shape latent grid size', value: state.metadata.shapeLatentGridSize },
          { label: 'o-voxel grid size', value: state.metadata.oVoxelGridSize },
        ]}
        progress={state.run.progress}
        status={state.run.status}
      />

      <GenerationActions
        nextActions={nextActions}
        onGenerate={onGenerate}
        readyForNext={readyForNext}
        running={running}
      />

      <ExportControls
        disabled={state.run.status !== 'ready' || !state.run.result}
        onExport={onExport}
        onParamsChange={onExportParamsChange}
        state={exportState}
      />
    </div>
  );
}

Trellis2SymmetryShapePanel.instruction = trellis2SymmetryShapeInstruction;
