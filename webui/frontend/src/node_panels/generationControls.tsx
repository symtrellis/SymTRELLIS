import type {
  CommonGenerationParams,
  GenerationStatus,
  SymmetryProjectionParams,
} from '../state/generation';
import { DurationRangeControl, IntegerStepperField } from './controls';

type NumberFieldProps = {
  disabled?: boolean;
  label: string;
  min?: number;
  onChange: (value: number) => void;
  step?: number;
  value: number;
};

type GenerationParametersProps = {
  disabled?: boolean;
  onParamsChange: (params: Partial<CommonGenerationParams>) => void;
  onSeedRandomized: () => void;
  params: CommonGenerationParams;
};

type SymmetryGenerationParametersProps = {
  disabled?: boolean;
  onParamsChange: (params: Partial<SymmetryProjectionParams>) => void;
  params: SymmetryProjectionParams;
};

type GenerationStatusBlockProps = {
  metadata: Array<{ label: string; value: number | string }>;
  progress: number;
  stage?: string;
  status: GenerationStatus;
};

type GenerationNextAction = {
  label: string;
  onClick: () => void;
};

type GenerationActionsProps = {
  nextActions: GenerationNextAction[];
  onGenerate: () => void;
  readyForNext: boolean;
  running: boolean;
};

export function NumberField({
  disabled = false,
  label,
  min,
  onChange,
  step,
  value,
}: NumberFieldProps) {
  return (
    <label className="field-row generation-field-row">
      <span className="field-label">{label}</span>
      <input
        disabled={disabled}
        min={min}
        onChange={(event) => onChange(Number(event.currentTarget.value))}
        step={step}
        type="number"
        value={value}
      />
    </label>
  );
}

export function SymmetryGenerationParameters({
  disabled = false,
  onParamsChange,
  params,
}: SymmetryGenerationParametersProps) {
  return (
    <section className="node-section">
      <NumberField
        disabled={disabled}
        label="noise symmetry projection strength"
        onChange={(noiseSymmetryProjectionStrength) => onParamsChange({ noiseSymmetryProjectionStrength })}
        step={0.01}
        value={params.noiseSymmetryProjectionStrength}
      />
      <NumberField
        disabled={disabled}
        label="symmetry projection strength"
        onChange={(symmetryProjectionStrength) => onParamsChange({ symmetryProjectionStrength })}
        step={0.01}
        value={params.symmetryProjectionStrength}
      />
      <DurationRangeControl
        disabled={disabled}
        label="symmetry projection duration"
        onChange={(symmetryProjectionDuration) => onParamsChange({ symmetryProjectionDuration })}
        value={params.symmetryProjectionDuration}
      />
    </section>
  );
}

export function GenerationParameters({
  disabled = false,
  onParamsChange,
  onSeedRandomized,
  params,
}: GenerationParametersProps) {
  return (
    <section className="node-section">
      <IntegerStepperField
        disabled={disabled}
        label="seed"
        onChange={(seed) => onParamsChange({ seed })}
        sideAction={
          <button
            className="button button-neutral button-compact integer-stepper-side-action"
            disabled={disabled}
            onClick={onSeedRandomized}
            type="button"
          >
            random
          </button>
        }
        value={params.seed}
      />
      <IntegerStepperField
        disabled={disabled}
        label="steps"
        min={1}
        onChange={(steps) => onParamsChange({ steps })}
        value={params.steps}
      />
      <NumberField
        disabled={disabled}
        label="time step rescale"
        onChange={(timeStepRescale) => onParamsChange({ timeStepRescale })}
        step={0.01}
        value={params.timeStepRescale}
      />
      <NumberField
        disabled={disabled}
        label="classifier free guidance strength"
        onChange={(cfgStrength) => onParamsChange({ cfgStrength })}
        step={0.1}
        value={params.cfgStrength}
      />
      <DurationRangeControl
        disabled={disabled}
        label="classifier free guidance duration"
        onChange={(cfgDuration) => onParamsChange({ cfgDuration })}
        value={params.cfgDuration}
      />
      <NumberField
        disabled={disabled}
        label="classifier free guidance rescale"
        onChange={(cfgRescale) => onParamsChange({ cfgRescale })}
        step={0.01}
        value={params.cfgRescale}
      />
    </section>
  );
}

export function GenerationStatusBlock({
  metadata,
  progress,
  stage,
  status,
}: GenerationStatusBlockProps) {
  if (status === 'idle') {
    return null;
  }

  return (
    <section className="node-section generation-status">
      <div className="generation-progress-row">
        <div className="generation-progress-track">
          <div
            className="generation-progress-fill"
            style={{ width: `${Math.round(progress * 100)}%` }}
          />
        </div>
        <span>{Math.round(progress * 100)}%</span>
      </div>

      <p>{stage ?? status}</p>

      {status === 'ready' && metadata.length > 0 ? (
        <dl className="generation-metadata">
          {metadata.map((item) => (
            <div key={item.label}>
              <dt>{item.label}</dt>
              <dd>{item.value}</dd>
            </div>
          ))}
        </dl>
      ) : null}
    </section>
  );
}

export function GenerationActions({
  nextActions,
  onGenerate,
  readyForNext,
  running,
}: GenerationActionsProps) {
  return (
    <>
      <button className="button button-neutral" disabled={running} onClick={onGenerate} type="button">
        Confirm and generate
      </button>

      {nextActions.map((action) => (
        <button
          className="button button-primary"
          disabled={!readyForNext}
          key={action.label}
          onClick={action.onClick}
          type="button"
        >
          {action.label}
        </button>
      ))}
    </>
  );
}
