import type { ActionResult, RequestId } from '../types';
import type { Trellis2ExportParams } from '../models/trellis2';
import { trellis2OutputRoleCandidates, trellis2ExportDefaults } from '../models/trellis2';
import { outputByRole } from '../api/storage';
import { NumberField } from './generationControls';

export type ExportStatus = 'idle' | 'running' | 'ready' | 'failed';

export type ExportState = {
  errorMessage: string;
  log: string;
  params: Trellis2ExportParams;
  progress: number;
  requestId: RequestId | null;
  result: ActionResult | null;
  status: ExportStatus;
};

type ExportControlsProps = {
  disabled?: boolean;
  onExport: () => void;
  onParamsChange: (params: Partial<Trellis2ExportParams>) => void;
  state: ExportState;
};

const initialExportState: ExportState = {
  errorMessage: '',
  log: '',
  params: trellis2ExportDefaults,
  progress: 0,
  requestId: null,
  result: null,
  status: 'idle',
};

export function ExportControls({
  disabled = false,
  onExport,
  onParamsChange,
  state,
}: ExportControlsProps) {
  const running = state.status === 'running';
  const canExtract = !disabled && !running;
  const glbOutput = outputByRole(state.result?.outputs ?? {}, trellis2OutputRoleCandidates.exportGlb);
  const bundleOutput = outputByRole(state.result?.outputs ?? {}, trellis2OutputRoleCandidates.exportBundle);
  const progressPercent = Math.round(state.progress * 100);

  return (
    <section className="node-section export-section">
      <NumberField
        disabled={running}
        label="face decimation target"
        onChange={(faceDecimationTarget) =>
          onParamsChange({ faceDecimationTarget: Math.trunc(faceDecimationTarget) })
        }
        step={1000}
        value={state.params.faceDecimationTarget}
      />

      <NumberField
        disabled={running}
        label="texture size"
        onChange={(textureSize) => onParamsChange({ textureSize: Math.trunc(textureSize) })}
        step={1}
        value={state.params.textureSize}
      />

      <label className="export-checkbox-row">
        <span className="field-label">remesh</span>
        <input
          checked={state.params.remesh}
          disabled={running}
          onChange={(event) => onParamsChange({ remesh: event.currentTarget.checked })}
          type="checkbox"
        />
      </label>

      <NumberField
        disabled={running}
        label="remesh band"
        onChange={(remeshBand) => onParamsChange({ remeshBand })}
        step={0.01}
        value={state.params.remeshBand}
      />

      <NumberField
        disabled={running}
        label="remesh project"
        onChange={(remeshProject) => onParamsChange({ remeshProject })}
        step={0.01}
        value={state.params.remeshProject}
      />

      <div className="export-progress">
        <div className="generation-progress-row">
          <span>{progressPercent}%</span>
          <span>{state.status}</span>
        </div>
        <div className="generation-progress-track">
          <div className="generation-progress-fill" style={{ width: `${progressPercent}%` }} />
        </div>
        <p>{state.log || 'Export has not started.'}</p>
      </div>

      <button className="button button-neutral" disabled={!canExtract} onClick={onExport} type="button">
        Extract GLB
      </button>

      {state.status === 'ready' && glbOutput ? (
        <a className="button button-primary export-download" href={glbOutput.url}>
          Download GLB
        </a>
      ) : (
        <button className="button button-primary" disabled type="button">
          Download GLB
        </button>
      )}

      {state.status === 'ready' && bundleOutput ? (
        <a className="button button-primary export-download" href={bundleOutput.url}>
          Download GLB and all latents
        </a>
      ) : (
        <button className="button button-primary" disabled type="button">
          Download GLB and all latents
        </button>
      )}

      {state.status === 'failed' ? (
        <p className="export-error">{state.errorMessage || 'Export failed.'}</p>
      ) : null}
    </section>
  );
}

ExportControls.initialState = initialExportState;
