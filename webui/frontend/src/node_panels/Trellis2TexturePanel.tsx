import type { Dispatch } from 'react';
import type {
  Trellis2ExportParams,
  Trellis2TextureAction,
  Trellis2TextureState,
} from '../models/trellis2';
import type { ExportState } from './exportControls';
import { ExportControls } from './exportControls';
import {
  GenerationActions,
  GenerationParameters,
  GenerationStatusBlock,
} from './generationControls';

type Trellis2TexturePanelProps = {
  dispatch: Dispatch<Trellis2TextureAction>;
  exportState: ExportState;
  nextActions: Array<{ label: string; onClick: () => void }>;
  onExport: () => void;
  onExportParamsChange: (params: Partial<Trellis2ExportParams>) => void;
  onGenerate: () => void;
  state: Trellis2TextureState;
};

function trellis2TextureInstruction(state: Trellis2TextureState): string {
  if (state.run.status === 'running') {
    return 'Generating texture. The viewer shows the input mesh until the textured mesh is ready.';
  }

  if (state.run.status === 'ready') {
    return 'Textured mesh is ready. Review the result.';
  }

  if (state.run.status === 'failed') {
    return state.run.errorMessage || 'Texture generation failed.';
  }

  return 'Review texture-generation parameters, then press Confirm and generate.';
}

export function Trellis2TexturePanel({
  dispatch,
  exportState,
  nextActions,
  onExport,
  onExportParamsChange,
  onGenerate,
  state,
}: Trellis2TexturePanelProps) {
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

      <GenerationStatusBlock metadata={[]} progress={state.run.progress} status={state.run.status} />

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

Trellis2TexturePanel.instruction = trellis2TextureInstruction;
