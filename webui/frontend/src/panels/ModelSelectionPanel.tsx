import type { ModelId, ModelOption } from '../models/types';

type ModelSelectionPanelProps = {
  modelOptions: ModelOption[];
  onConfirm: () => void;
  selectedModelId: ModelId;
};

export function ModelSelectionPanel({
  modelOptions,
  onConfirm,
  selectedModelId,
}: ModelSelectionPanelProps) {
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
