import type { Dispatch } from 'react';
import { canProposeSymmetry } from '../state';
import type { DetectionAction, DetectionState } from '../state';

function formatVector(vector: DetectionState['majorAxis']) {
  return `[${vector.map((value) => Number(value.toFixed(4)).toString()).join(', ')}]`;
}

type DetectAdjustSymmetryPanelProps = {
  dispatch: Dispatch<DetectionAction>;
  onDetectFinerSymmetry: () => Promise<void>;
  onDetectMajorAxis: () => Promise<void>;
  state: DetectionState;
};

export function DetectAdjustSymmetryPanel({
  dispatch,
  onDetectFinerSymmetry,
  onDetectMajorAxis,
  state,
}: DetectAdjustSymmetryPanelProps) {
  const majorReady = state.majorStatus === 'ready';
  const finerRunning = state.finerStatus === 'running';
  const majorRunning = state.majorStatus === 'running';
  const canDetectFiner = majorReady && state.family === 'axial' && !finerRunning;
  const canPropose = canProposeSymmetry(state);

  return (
    <div className="detect-panel">
      <section className="detect-section">
        <button
          className="button button-neutral"
          disabled={majorRunning}
          onClick={() => void onDetectMajorAxis()}
          type="button"
        >
          {majorRunning ? 'Detecting major axis' : 'Detect major axis'}
        </button>

        {state.majorStatus === 'empty' ? (
          <div className="empty-state">no symmetry detected</div>
        ) : null}

        <div className="field-stack">
          <label className="field-row">
            <span className="field-label">axis</span>
            <span className="vector-inputs">
              <input
                disabled={!majorReady}
                onChange={(event) =>
                  dispatch({
                    axis: [Number(event.currentTarget.value), state.majorAxis[1], state.majorAxis[2]],
                    type: 'majorAxisChanged',
                  })
                }
                type="number"
                value={state.majorAxis[0]}
              />
              <input
                disabled={!majorReady}
                onChange={(event) =>
                  dispatch({
                    axis: [state.majorAxis[0], Number(event.currentTarget.value), state.majorAxis[2]],
                    type: 'majorAxisChanged',
                  })
                }
                type="number"
                value={state.majorAxis[1]}
              />
              <input
                disabled={!majorReady}
                onChange={(event) =>
                  dispatch({
                    axis: [state.majorAxis[0], state.majorAxis[1], Number(event.currentTarget.value)],
                    type: 'majorAxisChanged',
                  })
                }
                type="number"
                value={state.majorAxis[2]}
              />
            </span>
            <button
              className="button button-neutral button-compact"
              disabled={!majorReady}
              onClick={() => dispatch({ type: 'majorAxisNormalized' })}
              type="button"
            >
              normalize
            </button>
          </label>

          <label className="field-row">
            <span className="field-label">center</span>
            <span className="vector-inputs">
              <input
                disabled={!majorReady}
                onChange={(event) =>
                  dispatch({
                    center: [Number(event.currentTarget.value), state.center[1], state.center[2]],
                    type: 'centerChanged',
                  })
                }
                type="number"
                value={state.center[0]}
              />
              <input
                disabled={!majorReady}
                onChange={(event) =>
                  dispatch({
                    center: [state.center[0], Number(event.currentTarget.value), state.center[2]],
                    type: 'centerChanged',
                  })
                }
                type="number"
                value={state.center[1]}
              />
              <input
                disabled={!majorReady}
                onChange={(event) =>
                  dispatch({
                    center: [state.center[0], state.center[1], Number(event.currentTarget.value)],
                    type: 'centerChanged',
                  })
                }
                type="number"
                value={state.center[2]}
              />
            </span>
            <button
              className="button button-neutral button-compact"
              disabled={!majorReady}
              onClick={() => dispatch({ type: 'centerNormalized' })}
              type="button"
            >
              normalize
            </button>
          </label>

          <label className="field-row field-row--scalar">
            <span className="field-label">fold</span>
            <input
              disabled={!majorReady}
              min={1}
              onChange={(event) =>
                dispatch({ fold: Number(event.currentTarget.value), type: 'foldChanged' })
              }
              type="number"
              value={state.fold}
            />
          </label>
        </div>
      </section>

      <section className="detect-section">
        <div className="family-options">
          <button
            className={`choice-button${state.family === 'axial' ? ' choice-button--selected' : ''}`}
            disabled={!majorReady}
            onClick={() => dispatch({ family: 'axial', type: 'familyPicked' })}
            type="button"
          >
            axial
          </button>
          <button
            className={`choice-button${state.family === 'T' ? ' choice-button--selected' : ''}`}
            disabled={!majorReady}
            onClick={() => dispatch({ family: 'T', type: 'familyPicked' })}
            type="button"
          >
            T
          </button>
          <button
            className={`choice-button${state.family === 'O' ? ' choice-button--selected' : ''}`}
            disabled={!majorReady}
            onClick={() => dispatch({ family: 'O', type: 'familyPicked' })}
            type="button"
          >
            O
          </button>
          <button
            className={`choice-button${state.family === 'I' ? ' choice-button--selected' : ''}`}
            disabled={!majorReady}
            onClick={() => dispatch({ family: 'I', type: 'familyPicked' })}
            type="button"
          >
            I
          </button>
        </div>

        {state.family === 'axial' ? (
          <button
            className="button button-neutral"
            disabled={!canDetectFiner}
            onClick={() => void onDetectFinerSymmetry()}
            type="button"
          >
            {finerRunning ? 'Detecting finer type' : 'Detect finer type'}
          </button>
        ) : null}
      </section>

      {state.labels.length > 0 ? (
        <section className="detect-section">
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
      ) : null}

      {state.family ? (
        <section className="detect-section">
          <label className="field-row">
            <span className="field-label">minor</span>
            <span className="vector-inputs">
              <input
                onChange={(event) =>
                  dispatch({
                    axis: [Number(event.currentTarget.value), state.minorAxis[1], state.minorAxis[2]],
                    type: 'minorAxisChanged',
                  })
                }
                type="number"
                value={state.minorAxis[0]}
              />
              <input
                onChange={(event) =>
                  dispatch({
                    axis: [state.minorAxis[0], Number(event.currentTarget.value), state.minorAxis[2]],
                    type: 'minorAxisChanged',
                  })
                }
                type="number"
                value={state.minorAxis[1]}
              />
              <input
                onChange={(event) =>
                  dispatch({
                    axis: [state.minorAxis[0], state.minorAxis[1], Number(event.currentTarget.value)],
                    type: 'minorAxisChanged',
                  })
                }
                type="number"
                value={state.minorAxis[2]}
              />
            </span>
            <button
              className="button button-neutral button-compact"
              onClick={() => dispatch({ type: 'minorAxisNormalized' })}
              type="button"
            >
              normalize
            </button>
          </label>
        </section>
      ) : null}

      <button
        className="button button-neutral"
        disabled={!canPropose || Boolean(state.proposedSymmetry)}
        onClick={() => dispatch({ type: 'proposeSymmetry' })}
        type="button"
      >
        Confirm proposed symmetry
      </button>

      {state.proposedSymmetry ? (
        <section className="detect-section proposed-symmetry">
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
