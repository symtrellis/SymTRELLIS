import * as Slider from '@radix-ui/react-slider';
import {
  estimateTrellis2Bf16FlowPeakGb,
  type Trellis2ShapeParams,
} from '../models/trellis2';

type Trellis2ShapeModeControlProps = {
  disabled?: boolean;
  onParamsChange: (params: Partial<Trellis2ShapeParams>) => void;
  params: Trellis2ShapeParams;
};

const shapeModes: Array<{ label: string; mode: Trellis2ShapeParams['mode']; path: string }> = [
  { label: '512', mode: '512', path: '32^3 -> 512^3' },
  { label: 'cascade', mode: 'cascade', path: '32^3 -> cascade upscale -> 1536^3' },
];

export function Trellis2ShapeModeControl({
  disabled = false,
  onParamsChange,
  params,
}: Trellis2ShapeModeControlProps) {
  return (
    <section className="node-section">
      <div className="shape-mode-options">
        {shapeModes.map((item) => (
          <button
            className={`choice-button${params.mode === item.mode ? ' choice-button--selected' : ''}`}
            disabled={disabled}
            key={item.mode}
            onClick={() => onParamsChange({ mode: item.mode })}
            type="button"
          >
            <span>{item.label}</span>
            <small>{item.path}</small>
          </button>
        ))}
      </div>

      {params.mode === 'cascade' ? (
        <div className="shape-token-panel">
          <label className="shape-token-row">
            <span className="field-label">max tokens</span>
            <input
              disabled={disabled}
              max={524288}
              min={4096}
              onChange={(event) =>
                onParamsChange({
                  maxTokens: Math.min(524288, Math.max(4096, Math.trunc(Number(event.currentTarget.value)))),
                })
              }
              step={1024}
              type="number"
              value={params.maxTokens}
            />
            <span className="shape-vram-est">
              VRAM est bf16 {estimateTrellis2Bf16FlowPeakGb(params.maxTokens).toFixed(2)} GB
            </span>
          </label>
          <Slider.Root
            className="shape-token-slider"
            disabled={disabled}
            max={524288}
            min={4096}
            onValueChange={(nextValue) => onParamsChange({ maxTokens: Math.trunc(nextValue[0]) })}
            step={1024}
            value={[params.maxTokens]}
          >
            <Slider.Track className="duration-slider-track">
              <Slider.Range className="duration-slider-range" />
            </Slider.Track>
            <Slider.Thumb aria-label="max tokens" className="duration-slider-thumb" />
          </Slider.Root>
        </div>
      ) : null}
    </section>
  );
}
