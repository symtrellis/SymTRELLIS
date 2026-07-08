import type { ChangeEvent, ClipboardEvent, DragEvent } from 'react';
import type { ImageConditionState } from '../state/imageCondition';

type ImageInput = {
  file: Blob | File;
  name: string;
};

type ImageConditionPanelProps = {
  onGenerateCondition: () => void;
  onImageSelected: (file: Blob | File, name: string) => void;
  routeActions: Array<{ label: string; onClick: () => void }>;
  state: ImageConditionState;
};

function imageInputFromClipboard(event: ClipboardEvent<HTMLElement>): ImageInput | null {
  const imageItem = Array.from(event.clipboardData.items).find((item) =>
    item.type.startsWith('image/'),
  );
  const file = imageItem?.getAsFile();

  if (!file) {
    return null;
  }

  return { file, name: file.name || 'pasted-image.png' };
}

export function ImageConditionPanel({
  onGenerateCondition,
  onImageSelected,
  routeActions,
  state,
}: ImageConditionPanelProps) {
  const hasImage = Boolean(state.previewName);
  const conditionReady = state.status === 'ready';
  const running = state.status === 'uploading' || state.status === 'generating';
  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.currentTarget.files?.[0];

    if (file) {
      onImageSelected(file, file.name);
      event.currentTarget.value = '';
    }
  };
  const handleDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();

    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      onImageSelected(file, file.name);
    }
  };
  const handleDragOver = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
  };
  const handlePaste = (event: ClipboardEvent<HTMLLabelElement>) => {
    const imageInput = imageInputFromClipboard(event);

    if (imageInput) {
      event.preventDefault();
      onImageSelected(imageInput.file, imageInput.name);
    }
  };

  return (
    <div className="node-panel-stack">
      <section className="node-section">
        <label
          className="upload-box"
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onPaste={handlePaste}
          tabIndex={0}
        >
          <input accept="image/*" onChange={handleFileChange} type="file" />
          {state.previewUrl ? (
            <span className="upload-preview">
              <img alt={state.previewName} src={state.previewUrl} />
            </span>
          ) : null}
          <span className="upload-box-title">{hasImage ? state.previewName : 'Upload image'}</span>
          <span className="upload-box-detail">
            {hasImage ? 'ready for condition generation' : 'drop, choose, or paste an input image'}
          </span>
        </label>

        <button
          className="button button-neutral"
          disabled={!hasImage || running}
          onClick={onGenerateCondition}
          type="button"
        >
          Generate condition
        </button>
      </section>

      {routeActions.length > 0 ? (
        <section className="node-section">
          {routeActions.map((action) => (
            <button
              className="button button-neutral"
              disabled={!conditionReady}
              key={action.label}
              onClick={action.onClick}
              type="button"
            >
              {action.label}
            </button>
          ))}
        </section>
      ) : null}
    </div>
  );
}
