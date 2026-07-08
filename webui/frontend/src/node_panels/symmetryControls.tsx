import type { Dispatch } from 'react';
import type { SymmetryFamily, SymmetryTuple } from '../types';
import { formatVector, symmetryFamilies } from '../state/symmetry';

type FamilyPickerAction = {
  family: SymmetryFamily;
  type: 'familyPicked';
};

type LabelPickerAction = {
  label: string;
  type: 'labelPicked';
};

type FamilyPickerProps = {
  disabled?: boolean;
  dispatch: Dispatch<FamilyPickerAction>;
  value: SymmetryFamily | null;
};

type PointGroupSelectProps = {
  dispatch: Dispatch<LabelPickerAction>;
  labels: string[];
  value: string;
};

type ProposedSymmetryBlockProps = {
  symmetry: SymmetryTuple | null;
};

export function FamilyPicker({ disabled = false, dispatch, value }: FamilyPickerProps) {
  return (
    <div className="family-options">
      {symmetryFamilies.map((family) => (
        <button
          className={`choice-button${value === family ? ' choice-button--selected' : ''}`}
          disabled={disabled}
          key={family}
          onClick={() => dispatch({ family, type: 'familyPicked' })}
          type="button"
        >
          {family}
        </button>
      ))}
    </div>
  );
}

export function PointGroupSelect({ dispatch, labels, value }: PointGroupSelectProps) {
  return (
    <label className="select-row">
      <span className="field-label">Point group type</span>
      <select
        className="point-group-select"
        onChange={(event) => dispatch({ label: event.currentTarget.value, type: 'labelPicked' })}
        value={value}
      >
        {labels.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}

export function ProposedSymmetryBlock({ symmetry }: ProposedSymmetryBlockProps) {
  if (!symmetry) {
    return null;
  }

  return (
    <section className="node-section proposed-symmetry">
      <dl className="proposed-tuple">
        <div className="proposed-row">
          <dt>Point group label</dt>
          <dd>{symmetry.label}</dd>
        </div>
        <div className="proposed-row">
          <dt>major axis</dt>
          <dd>{formatVector(symmetry.majorAxis)}</dd>
        </div>
        <div className="proposed-row">
          <dt>minor axis</dt>
          <dd>{formatVector(symmetry.minorAxis)}</dd>
        </div>
        <div className="proposed-row">
          <dt>center</dt>
          <dd>{formatVector(symmetry.center)}</dd>
        </div>
      </dl>
    </section>
  );
}
