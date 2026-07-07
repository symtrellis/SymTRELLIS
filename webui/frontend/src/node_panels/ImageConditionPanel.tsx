import type { ChangeEvent, ClipboardEvent, DragEvent } from 'react';
import type { ImageConditionState } from '../state';

type ImageInput = {
  file: Blob | File;
  name: string;
};

type ImageConditionPanelProps = {
  onEnterManualSymmetry: () => void;
  onGenerateCondition: () => void;
  onImageSelected: (file: Blob | File, name: string) => void;
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
  onEnterManualSymmetry,
  onGenerateCondition,
  onImageSelected,
  state,
}: ImageConditionPanelProps) {
  const hasImage = Boolean(state.uploadedImageName);
  const conditionReady = state.conditionStatus === 'ready';
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
    <div className="symmetry-panel">
      <section className="symmetry-section">
        <label
          className="upload-box"
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onPaste={handlePaste}
          tabIndex={0}
        >
          <input
            accept="image/*"
            onChange={handleFileChange}
            type="file"
          />
          {state.uploadedImageUrl ? (
            <span className="upload-preview">
              <img alt={state.uploadedImageName} src={state.uploadedImageUrl} />
            </span>
          ) : null}
          <span className="upload-box-title">{hasImage ? state.uploadedImageName : 'Upload image'}</span>
          <span className="upload-box-detail">
            {hasImage ? 'ready for condition generation' : 'drop, choose, or paste an input image'}
          </span>
        </label>

        <button
          className="button button-neutral"
          disabled={!hasImage}
          onClick={onGenerateCondition}
          type="button"
        >
          Generate condition
        </button>
      </section>

      {conditionReady ? (
        <section className="symmetry-section">
          <button
            className="button button-neutral"
            onClick={onEnterManualSymmetry}
            type="button"
          >
            Manually specify symmetry
          </button>
          <button className="button button-neutral" disabled type="button">
            Native generation and detect
          </button>
        </section>
      ) : null}
    </div>
  );
}
