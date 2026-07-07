import type { ModelId } from '../types';

const modelOptions: Array<{ disabled: boolean; id: ModelId; label: string }> = [
  { disabled: true, id: 'trellis', label: 'TRELLIS' },
  { disabled: false, id: 'trellis2', label: 'TRELLIS.2' },
  { disabled: true, id: 'sam3d_object', label: 'SAM-3D Object' },
];

type ModelSelectionPanelProps = {
  onConfirm: () => void;
  selectedModelId: ModelId;
};

export function ModelSelectionPanel({ onConfirm, selectedModelId }: ModelSelectionPanelProps) {
  return (
    <section className="pre-dag-panel glass-panel" aria-label="Model selection">
      <header className="pre-dag-panel-header">
        <h1>Select model</h1>
        <p className="node-instruction">Choose the model DAG to load.</p>
      </header>

      <div className="pre-dag-panel-body">
        <div className="model-options">
          {modelOptions.map((option) => (
            <button
              className={`model-option${option.id === selectedModelId ? ' model-option--selected' : ''}`}
              disabled={option.disabled}
              key={option.id}
              type="button"
            >
              {option.label}
            </button>
          ))}
        </div>

        <button className="button button-primary" onClick={onConfirm} type="button">
          Confirm and next
        </button>
      </div>
    </section>
  );
}
