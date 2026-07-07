import * as Slider from '@radix-ui/react-slider';
import type { Dispatch } from 'react';
import type {
  DurationRange,
  VanillaSparseStructureAction,
  VanillaSparseStructureState,
} from '../state';

type ScalarFieldProps = {
  disabled?: boolean;
  label: string;
  onChange: (value: number) => void;
  step?: number;
  value: number;
};

function ScalarField({ disabled = false, label, onChange, step, value }: ScalarFieldProps) {
  return (
    <label className="vanilla-ss-field-row">
      <span className="field-label">{label}</span>
      <input
        disabled={disabled}
        onChange={(event) => onChange(Number(event.currentTarget.value))}
        step={step}
        type="number"
        value={value}
      />
    </label>
  );
}

type IntegerStepperFieldProps = {
  disabled?: boolean;
  label: string;
  min?: number;
  onChange: (value: number) => void;
  sideAction?: {
    label: string;
    onClick: () => void;
  };
  value: number;
};

function IntegerStepperField({
  disabled = false,
  label,
  min,
  onChange,
  sideAction,
  value,
}: IntegerStepperFieldProps) {
  const decrementDisabled = disabled || (min !== undefined && value <= min);

  return (
    <label className="integer-stepper-row">
      <span className="field-label">{label}</span>
      <span className="integer-stepper-wrap">
        {sideAction ? (
          <button
            className="button button-neutral button-compact integer-stepper-side-action"
            disabled={disabled}
            onClick={sideAction.onClick}
            type="button"
          >
            {sideAction.label}
          </button>
        ) : null}
        <span className="integer-stepper">
          <button
            className="button button-neutral button-compact"
            disabled={decrementDisabled}
            onClick={() => onChange(min === undefined ? value - 1 : Math.max(min, value - 1))}
            type="button"
          >
            -
          </button>
          <input
            disabled={disabled}
            onChange={(event) => onChange(Number(event.currentTarget.value))}
            step={1}
            type="number"
            value={value}
          />
          <button
            className="button button-neutral button-compact"
            disabled={disabled}
            onClick={() => onChange(value + 1)}
            type="button"
          >
            +
          </button>
        </span>
      </span>
    </label>
  );
}

type DurationRangeControlProps = {
  disabled?: boolean;
  label: string;
  onChange: (value: DurationRange) => void;
  value: DurationRange;
};

function DurationRangeControl({
  disabled = false,
  label,
  onChange,
  value,
}: DurationRangeControlProps) {
  return (
    <div className="duration-row">
      <span className="field-label">{label}</span>
      <div className="duration-control">
        <div className="duration-inputs">
          <input
            disabled={disabled}
            max={1}
            min={0}
            onChange={(event) =>
              onChange([Math.min(Number(event.currentTarget.value), value[1]), value[1]])
            }
            step={0.01}
            type="number"
            value={value[0]}
          />
          <input
            disabled={disabled}
            max={1}
            min={0}
            onChange={(event) =>
              onChange([value[0], Math.max(Number(event.currentTarget.value), value[0])])
            }
            step={0.01}
            type="number"
            value={value[1]}
          />
        </div>
        <Slider.Root
          className="duration-slider"
          disabled={disabled}
          max={1}
          min={0}
          onValueChange={(nextValue) => onChange([nextValue[0], nextValue[1]])}
          step={0.01}
          value={value}
        >
          <Slider.Track className="duration-slider-track">
            <Slider.Range className="duration-slider-range" />
          </Slider.Track>
          <Slider.Thumb aria-label={`${label} start`} className="duration-slider-thumb" />
          <Slider.Thumb aria-label={`${label} end`} className="duration-slider-thumb" />
        </Slider.Root>
      </div>
    </div>
  );
}

type VanillaSparseStructurePanelProps = {
  dispatch: Dispatch<VanillaSparseStructureAction>;
  onGenerate: () => void;
  state: VanillaSparseStructureState;
};

export function VanillaSparseStructurePanel({
  dispatch,
  onGenerate,
  state,
}: VanillaSparseStructurePanelProps) {
  const running = state.status === 'running';
  const readyForNext = state.status === 'ready' && state.voxelCount > 0;

  return (
    <div className="symmetry-panel vanilla-ss-panel">
      <section className="symmetry-section">
        <IntegerStepperField
          disabled={running}
          label="seed"
          onChange={(seed) => dispatch({ params: { seed }, type: 'paramsChanged' })}
          sideAction={{
            label: 'random',
            onClick: () => dispatch({ type: 'seedRandomized' }),
          }}
          value={state.seed}
        />
        <IntegerStepperField
          disabled={running}
          label="steps"
          min={1}
          onChange={(steps) => dispatch({ params: { steps }, type: 'paramsChanged' })}
          value={state.steps}
        />
        <ScalarField
          disabled={running}
          label="time step rescale"
          onChange={(timeStepRescale) =>
            dispatch({ params: { timeStepRescale }, type: 'paramsChanged' })
          }
          step={0.01}
          value={state.timeStepRescale}
        />
        <ScalarField
          disabled={running}
          label="classifier free guidance strength"
          onChange={(cfgStrength) => dispatch({ params: { cfgStrength }, type: 'paramsChanged' })}
          step={0.1}
          value={state.cfgStrength}
        />
        <DurationRangeControl
          disabled={running}
          label="classifier free guidance duration"
          onChange={(cfgDuration) => dispatch({ params: { cfgDuration }, type: 'paramsChanged' })}
          value={state.cfgDuration}
        />
        <ScalarField
          disabled={running}
          label="classifier free guidance rescale"
          onChange={(cfgRescale) => dispatch({ params: { cfgRescale }, type: 'paramsChanged' })}
          step={0.01}
          value={state.cfgRescale}
        />
      </section>

      {state.status !== 'idle' ? (
        <section className="symmetry-section generation-status">
          <div className="generation-progress-row">
            <div className="generation-progress-track">
              <div
                className="generation-progress-fill"
                style={{ width: `${Math.round(state.progress * 100)}%` }}
              />
            </div>
            <span>{Math.round(state.progress * 100)}%</span>
          </div>
          {state.status === 'ready' ? (
            <dl className="generation-metadata">
              <div>
                <dt>voxel count</dt>
                <dd>{state.voxelCount}</dd>
              </div>
            </dl>
          ) : null}
        </section>
      ) : null}

      <button
        className="button button-neutral"
        disabled={running}
        onClick={onGenerate}
        type="button"
      >
        Confirm and generate
      </button>

      <button className="button button-primary" disabled={!readyForNext} type="button">
        Go to next step
      </button>
    </div>
  );
}
