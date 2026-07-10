import type { Dispatch } from 'react';
import {
  axisShortcutDisabled,
  axisShortcuts,
  canProposeManualSymmetry,
  manualSymmetryFamilies,
} from '../state/symmetry';
import type { ManualSymmetryAction, ManualSymmetryState } from '../state/symmetry';
import { FamilyPicker, PointGroupSelect, ProposedSymmetryBlock } from './symmetryControls';
import { IntegerStepperField, VectorField } from './controls';

type ManualSymmetryPanelProps = {
  dispatch: Dispatch<ManualSymmetryAction>;
  nodeReady: boolean;
  onConfirm: () => void;
  onNext?: () => void;
  state: ManualSymmetryState;
};

export function ManualSymmetryPanel({
  dispatch,
  nodeReady,
  onConfirm,
  onNext,
  state,
}: ManualSymmetryPanelProps) {
  const canPropose = canProposeManualSymmetry(state);

  return (
    <div className="node-panel-stack">
      <section className="node-section">
        <div className="field-stack">
          <VectorField
            action={
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
            }
            label="major axis"
            onChange={(axis) => dispatch({ axis, type: 'majorAxisChanged' })}
            value={state.majorAxis}
          />

          <VectorField
            action={
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
            }
            label="minor axis"
            onChange={(axis) => dispatch({ axis, type: 'minorAxisChanged' })}
            value={state.minorAxis}
          />

          <VectorField
            label="center"
            onChange={(center) => dispatch({ center, type: 'centerChanged' })}
            value={state.center}
          />
        </div>
      </section>

      <section className="node-section manual-family-section">
        <FamilyPicker
          displayNames={{ reflection: 'reflect' }}
          families={manualSymmetryFamilies}
          onChange={(family) => dispatch({ family, type: 'familyPicked' })}
          value={state.family}
        />

        {state.family === 'axial' ? (
          <IntegerStepperField
            label="fold"
            min={1}
            onChange={(fold) => dispatch({ fold, type: 'foldChanged' })}
            value={state.fold}
          />
        ) : null}
      </section>

      <section className="node-section">
        {state.family === 'reflection' ? (
          <div className="select-row">
            <span className="field-label">Point group type</span>
            <output className="point-group-value">S1</output>
          </div>
        ) : (
          <PointGroupSelect dispatch={dispatch} labels={state.labels} value={state.selectedLabel} />
        )}
      </section>

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
