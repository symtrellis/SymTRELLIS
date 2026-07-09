import type { Dispatch } from 'react';
import type {
  Trellis2ExportParams,
  Trellis2VanillaShapeAction,
  Trellis2VanillaShapeState,
} from '../models/trellis2';
import type { ExportState } from './exportControls';
import { ExportControls } from './exportControls';
import {
  GenerationActions,
  GenerationParameters,
  GenerationStatusBlock,
} from './generationControls';
import { Trellis2ShapeModeControl } from './trellis2ShapeControls';

type Trellis2VanillaShapePanelProps = {
  dispatch: Dispatch<Trellis2VanillaShapeAction>;
  exportState: ExportState;
  nextActions: Array<{ label: string; onClick: () => void }>;
  onExport: () => void;
  onExportParamsChange: (params: Partial<Trellis2ExportParams>) => void;
  onGenerate: () => void;
  state: Trellis2VanillaShapeState;
};

function trellis2VanillaShapeInstruction(state: Trellis2VanillaShapeState): string {
  if (state.run.status === 'running') {
    return 'Generating vanilla shape. The viewer shows the input occupancy until the mesh is ready.';
  }

  if (state.run.status === 'ready') {
    return 'Shape mesh is ready. Review grid metadata, then go to the next step.';
  }

  if (state.run.status === 'failed') {
    return state.run.errorMessage || 'Vanilla shape generation failed.';
  }

  return 'Review vanilla shape-generation parameters, then press Confirm and generate.';
}

export function Trellis2VanillaShapePanel({
  dispatch,
  exportState,
  nextActions,
  onExport,
  onExportParamsChange,
  onGenerate,
  state,
}: Trellis2VanillaShapePanelProps) {
  const running = state.run.status === 'running';
  const readyForNext = state.run.status === 'ready' && Boolean(state.run.result);

  return (
    <div className="node-panel-stack">
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

Trellis2VanillaShapePanel.instruction = trellis2VanillaShapeInstruction;
