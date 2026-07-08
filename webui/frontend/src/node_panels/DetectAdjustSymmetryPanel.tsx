import type { Dispatch } from 'react';
import { canProposeSymmetry } from '../state/detection';
import type { DetectionAction, DetectionState } from '../state/detection';
import { FamilyPicker, PointGroupSelect, ProposedSymmetryBlock } from './symmetryControls';
import { IntegerStepperField, VectorField } from './controls';

type DetectAdjustSymmetryPanelProps = {
  dispatch: Dispatch<DetectionAction>;
  nodeReady: boolean;
  onConfirm: () => void;
  onDetectFinerSymmetry: () => void;
  onDetectMajorAxis: () => void;
  onNext?: () => void;
  state: DetectionState;
};

export function DetectAdjustSymmetryPanel({
  dispatch,
  nodeReady,
  onConfirm,
  onDetectFinerSymmetry,
  onDetectMajorAxis,
  onNext,
  state,
}: DetectAdjustSymmetryPanelProps) {
  const majorReady = state.majorStatus === 'ready';
  const finerRunning = state.finerStatus === 'running';
  const majorRunning = state.majorStatus === 'running';
  const canDetectFiner = majorReady && state.family === 'axial' && !finerRunning;
  const canPropose = canProposeSymmetry(state);

  return (
    <div className="node-panel-stack">
      <section className="node-section">
        <button
          className="button button-neutral"
          disabled={majorRunning}
          onClick={onDetectMajorAxis}
          type="button"
        >
          {majorRunning ? 'Detecting major axis' : 'Detect major axis'}
        </button>

        {state.majorStatus === 'empty' ? (
          <div className="empty-state">no symmetry detected</div>
        ) : null}

        <div className="field-stack">
          <VectorField
            action={
              <button
                className="button button-neutral button-compact"
                disabled={!majorReady}
                onClick={() => dispatch({ type: 'majorAxisNormalized' })}
                type="button"
              >
                normalize
              </button>
            }
            disabled={!majorReady}
            label="axis"
            onChange={(axis) => dispatch({ axis, type: 'majorAxisChanged' })}
            value={state.majorAxis}
          />

          <VectorField
            action={
              <button
                className="button button-neutral button-compact"
                disabled={!majorReady}
                onClick={() => dispatch({ type: 'centerNormalized' })}
                type="button"
              >
                normalize
              </button>
            }
            disabled={!majorReady}
            label="center"
            onChange={(center) => dispatch({ center, type: 'centerChanged' })}
            value={state.center}
          />

          <IntegerStepperField
            disabled={!majorReady}
            label="fold"
            min={1}
            onChange={(fold) => dispatch({ fold, type: 'foldChanged' })}
            value={state.fold}
          />
        </div>
      </section>

      <section className="node-section">
        <FamilyPicker disabled={!majorReady} dispatch={dispatch} value={state.family} />

        {state.family === 'axial' ? (
          <button
            className="button button-neutral"
            disabled={!canDetectFiner}
            onClick={onDetectFinerSymmetry}
            type="button"
          >
            {finerRunning ? 'Detecting finer type' : 'Detect finer type'}
          </button>
        ) : null}
      </section>

      {state.labels.length > 0 ? (
        <section className="node-section">
          <PointGroupSelect dispatch={dispatch} labels={state.labels} value={state.selectedLabel} />
        </section>
      ) : null}

      {state.family ? (
        <section className="node-section">
          <VectorField
            action={
              <button
                className="button button-neutral button-compact"
                onClick={() => dispatch({ type: 'minorAxisNormalized' })}
                type="button"
              >
                normalize
              </button>
            }
            label="minor"
            onChange={(axis) => dispatch({ axis, type: 'minorAxisChanged' })}
            value={state.minorAxis}
          />
        </section>
      ) : null}

      <button
        className="button button-neutral"
        disabled={!canPropose || Boolean(state.proposedSymmetry)}
        onClick={() => dispatch({ type: 'proposeSymmetry' })}
        type="button"
      >
        Visualize specified symmetry
      </button>

      <ProposedSymmetryBlock symmetry={state.proposedSymmetry} />

      <button
        className="button button-primary"
        disabled={!state.proposedSymmetry || nodeReady}
        onClick={onConfirm}
        type="button"
      >
        Confirm
      </button>

      {onNext ? (
        <button className="button button-primary" disabled={!nodeReady} onClick={onNext} type="button">
          Start symmetry enforced generation
        </button>
      ) : null}
    </div>
  );
}
