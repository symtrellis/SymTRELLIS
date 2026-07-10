import type { Dispatch } from 'react';
import { canProposeSymmetry } from '../state/detection';
import type { DetectionAction, DetectionState } from '../state/detection';
import { rotationSymmetryFamilies } from '../state/symmetry';
import { FamilyPicker, PointGroupSelect, ProposedSymmetryBlock } from './symmetryControls';
import { IntegerStepperField, VectorField } from './controls';

type DetectAdjustSymmetryPanelProps = {
  dispatch: Dispatch<DetectionAction>;
  nodeReady: boolean;
  onConfirm: () => void;
  onDetectFinerSymmetry: () => void;
  onDetectMajorAxis: () => void;
  onDetectReflectionPlanes: () => void;
  onNext?: () => void;
  state: DetectionState;
};

export function DetectAdjustSymmetryPanel({
  dispatch,
  nodeReady,
  onConfirm,
  onDetectFinerSymmetry,
  onDetectMajorAxis,
  onDetectReflectionPlanes,
  onNext,
  state,
}: DetectAdjustSymmetryPanelProps) {
  const majorReady = state.majorStatus === 'ready';
  const finerRunning = state.finerStatus === 'running';
  const majorRunning = state.majorStatus === 'running';
  const reflectionReady = state.reflectionStatus === 'ready';
  const reflectionRunning = state.reflectionStatus === 'running';
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
          {majorRunning ? 'Detecting major rotation axis' : 'Detect major rotation axis'}
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
        <FamilyPicker
          families={rotationSymmetryFamilies}
          onChange={(family) => dispatch({ family, type: 'familyPicked' })}
          value={state.family}
        />

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

      <section className="node-section">
        <PointGroupSelect dispatch={dispatch} labels={state.labels} value={state.selectedLabel} />
      </section>

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

      <div className="node-section-divider" aria-hidden="true" />

      <section className="node-section">
        <button
          className="button button-neutral"
          disabled={reflectionRunning}
          onClick={onDetectReflectionPlanes}
          type="button"
        >
          {reflectionRunning ? 'Detecting reflection planes' : 'Detect reflection planes'}
        </button>

        {state.reflectionStatus === 'empty' ? (
          <div className="empty-state">no reflection plane detected</div>
        ) : null}

        <div className="field-stack">
          <VectorField
            action={
              <button
                className="button button-neutral button-compact"
                disabled={!reflectionReady}
                onClick={() => dispatch({ type: 'reflectionNormalNormalized' })}
                type="button"
              >
                normalize
              </button>
            }
            disabled={!reflectionReady}
            label="normal"
            onChange={(normal) => dispatch({ normal, type: 'reflectionNormalChanged' })}
            value={state.reflectionNormal}
          />

          <VectorField
            action={
              <button
                className="button button-neutral button-compact"
                disabled={!reflectionReady}
                onClick={() => dispatch({ type: 'reflectionCenterNormalized' })}
                type="button"
              >
                normalize
              </button>
            }
            disabled={!reflectionReady}
            label="center"
            onChange={(center) => dispatch({ center, type: 'reflectionCenterChanged' })}
            value={state.reflectionCenter}
          />
        </div>
      </section>

      <div className="node-section-divider" aria-hidden="true" />

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
        disabled={!state.proposedSymmetry || nodeReady || state.confirming}
        onClick={onConfirm}
        type="button"
      >
        {state.confirming ? 'Confirming' : 'Confirm'}
      </button>

      {onNext ? (
        <button className="button button-primary" disabled={!nodeReady} onClick={onNext} type="button">
          Start symmetry enforced generation
        </button>
      ) : null}
    </div>
  );
}
