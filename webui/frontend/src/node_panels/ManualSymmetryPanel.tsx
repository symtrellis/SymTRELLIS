import type { Dispatch } from 'react';
import {
  axisShortcutDisabled,
  canProposeManualSymmetry,
} from '../state';
import type { ManualSymmetryAction, ManualSymmetryState } from '../state';
import type { SymmetryFamily, Vector3 } from '../types';

const axisShortcuts: Array<{ axis: Vector3; label: string }> = [
  { axis: [1, 0, 0], label: 'X' },
  { axis: [0, 1, 0], label: 'Y' },
  { axis: [0, 0, 1], label: 'Z' },
];

const families: SymmetryFamily[] = ['axial', 'T', 'O', 'I'];

function formatVector(vector: Vector3) {
  return `[${vector.map((value) => Number(value.toFixed(4)).toString()).join(', ')}]`;
}

function vectorWithValue(vector: Vector3, index: number, value: number): Vector3 {
  return [
    index === 0 ? value : vector[0],
    index === 1 ? value : vector[1],
    index === 2 ? value : vector[2],
  ];
}

type ManualSymmetryPanelProps = {
  dispatch: Dispatch<ManualSymmetryAction>;
  state: ManualSymmetryState;
};

export function ManualSymmetryPanel({ dispatch, state }: ManualSymmetryPanelProps) {
  const canPropose = canProposeManualSymmetry(state);

  return (
    <div className="symmetry-panel">
      <section className="symmetry-section">
        <div className="field-stack">
          <label className="field-row">
            <span className="field-label">major axis</span>
            <span className="vector-inputs">
              <input
                onChange={(event) =>
                  dispatch({
                    axis: vectorWithValue(state.majorAxis, 0, Number(event.currentTarget.value)),
                    type: 'majorAxisChanged',
                  })
                }
                type="number"
                value={state.majorAxis[0]}
              />
              <input
                onChange={(event) =>
                  dispatch({
                    axis: vectorWithValue(state.majorAxis, 1, Number(event.currentTarget.value)),
                    type: 'majorAxisChanged',
                  })
                }
                type="number"
                value={state.majorAxis[1]}
              />
              <input
                onChange={(event) =>
                  dispatch({
                    axis: vectorWithValue(state.majorAxis, 2, Number(event.currentTarget.value)),
                    type: 'majorAxisChanged',
                  })
                }
                type="number"
                value={state.majorAxis[2]}
              />
            </span>
            <span className="axis-shortcuts">
              {axisShortcuts.map((shortcut) => (
                <button
                  className="button button-neutral axis-shortcut"
                  key={shortcut.label}
                  onClick={() =>
                    dispatch({ axis: shortcut.axis, type: 'majorAxisShortcutPicked' })
                  }
                  type="button"
                >
                  {shortcut.label}
                </button>
              ))}
            </span>
          </label>

          <label className="field-row">
            <span className="field-label">minor axis</span>
            <span className="vector-inputs">
              <input
                onChange={(event) =>
                  dispatch({
                    axis: vectorWithValue(state.minorAxis, 0, Number(event.currentTarget.value)),
                    type: 'minorAxisChanged',
                  })
                }
                type="number"
                value={state.minorAxis[0]}
              />
              <input
                onChange={(event) =>
                  dispatch({
                    axis: vectorWithValue(state.minorAxis, 1, Number(event.currentTarget.value)),
                    type: 'minorAxisChanged',
                  })
                }
                type="number"
                value={state.minorAxis[1]}
              />
              <input
                onChange={(event) =>
                  dispatch({
                    axis: vectorWithValue(state.minorAxis, 2, Number(event.currentTarget.value)),
                    type: 'minorAxisChanged',
                  })
                }
                type="number"
                value={state.minorAxis[2]}
              />
            </span>
            <span className="axis-shortcuts">
              {axisShortcuts.map((shortcut) => (
                <button
                  className="button button-neutral axis-shortcut"
                  disabled={axisShortcutDisabled(shortcut.axis, state.majorAxis)}
                  key={shortcut.label}
                  onClick={() =>
                    dispatch({ axis: shortcut.axis, type: 'minorAxisShortcutPicked' })
                  }
                  type="button"
                >
                  {shortcut.label}
                </button>
              ))}
            </span>
          </label>

          <label className="field-row field-row--no-action">
            <span className="field-label">center</span>
            <span className="vector-inputs">
              <input
                onChange={(event) =>
                  dispatch({
                    center: vectorWithValue(state.center, 0, Number(event.currentTarget.value)),
                    type: 'centerChanged',
                  })
                }
                type="number"
                value={state.center[0]}
              />
              <input
                onChange={(event) =>
                  dispatch({
                    center: vectorWithValue(state.center, 1, Number(event.currentTarget.value)),
                    type: 'centerChanged',
                  })
                }
                type="number"
                value={state.center[1]}
              />
              <input
                onChange={(event) =>
                  dispatch({
                    center: vectorWithValue(state.center, 2, Number(event.currentTarget.value)),
                    type: 'centerChanged',
                  })
                }
                type="number"
                value={state.center[2]}
              />
            </span>
          </label>
        </div>
      </section>

      <section className="symmetry-section">
        <div className="family-options">
          {families.map((family) => (
            <button
              className={`choice-button${state.family === family ? ' choice-button--selected' : ''}`}
              key={family}
              onClick={() => dispatch({ family, type: 'familyPicked' })}
              type="button"
            >
              {family}
            </button>
          ))}
        </div>

        {state.family === 'axial' ? (
          <label className="field-row field-row--scalar">
            <span className="field-label">fold</span>
            <input
              min={1}
              onChange={(event) =>
                dispatch({ fold: Number(event.currentTarget.value), type: 'foldChanged' })
              }
              step={1}
              type="number"
              value={state.fold}
            />
          </label>
        ) : null}
      </section>

      <section className="symmetry-section">
        <label className="select-row">
          <span className="field-label">Point group type</span>
          <select
            className="point-group-select"
            onChange={(event) =>
              dispatch({ label: event.currentTarget.value, type: 'labelPicked' })
            }
            value={state.selectedLabel}
          >
            {state.labels.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
      </section>

      <button
        className="button button-neutral"
        disabled={!canPropose || Boolean(state.proposedSymmetry)}
        onClick={() => dispatch({ type: 'proposeSymmetry' })}
        type="button"
      >
        Confirm proposed symmetry
      </button>

      {state.proposedSymmetry ? (
        <section className="symmetry-section proposed-symmetry">
          <dl className="proposed-tuple">
            <div className="proposed-row">
              <dt>Point group label</dt>
              <dd>{state.proposedSymmetry.label}</dd>
            </div>
            <div className="proposed-row">
              <dt>major axis</dt>
              <dd>{formatVector(state.proposedSymmetry.majorAxis)}</dd>
            </div>
            <div className="proposed-row">
              <dt>minor axis</dt>
              <dd>{formatVector(state.proposedSymmetry.minorAxis)}</dd>
            </div>
            <div className="proposed-row">
              <dt>center</dt>
              <dd>{formatVector(state.proposedSymmetry.center)}</dd>
            </div>
          </dl>
        </section>
      ) : null}

      <button
        className="button button-primary"
        disabled={!state.proposedSymmetry}
        onClick={() => dispatch({ type: 'confirmSymmetry' })}
        type="button"
      >
        Confirm
      </button>
    </div>
  );
}
