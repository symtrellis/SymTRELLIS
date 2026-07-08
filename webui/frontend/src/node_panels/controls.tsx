import * as Slider from '@radix-ui/react-slider';
import type { ReactNode } from 'react';
import type { DurationRange } from '../state/generation';
import type { Vector3 } from '../types';
import { vectorWithValue } from '../state/symmetry';

type VectorFieldProps = {
  action?: ReactNode;
  disabled?: boolean;
  label: string;
  onChange: (value: Vector3) => void;
  value: Vector3;
};

type ScalarFieldProps = {
  disabled?: boolean;
  label: string;
  min?: number;
  onChange: (value: number) => void;
  step?: number;
  value: number;
};

type IntegerStepperFieldProps = ScalarFieldProps & {
  sideAction?: ReactNode;
};

type DurationRangeControlProps = {
  disabled?: boolean;
  label: string;
  onChange: (value: DurationRange) => void;
  value: DurationRange;
};

export function VectorField({ action, disabled = false, label, onChange, value }: VectorFieldProps) {
  return (
    <label className={`field-row${action ? '' : ' field-row--no-action'}`}>
      <span className="field-label">{label}</span>
      <span className="vector-inputs">
        {[0, 1, 2].map((index) => (
          <input
            disabled={disabled}
            key={index}
            onChange={(event) => onChange(vectorWithValue(value, index, Number(event.currentTarget.value)))}
            type="number"
            value={value[index]}
          />
        ))}
      </span>
      {action}
    </label>
  );
}

export function ScalarField({ disabled = false, label, min, onChange, step, value }: ScalarFieldProps) {
  return (
    <label className="field-row field-row--scalar">
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

export function IntegerStepperField({
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
        {sideAction}
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

export function DurationRangeControl({
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
