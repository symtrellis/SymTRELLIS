import type { ReactNode } from 'react';
import { ArrowLeft } from 'lucide-react';

type NodePanelProps = {
  children: ReactNode;
  footer?: ReactNode;
  instruction?: ReactNode;
  onBack?: () => void;
  title: string;
};

export function NodePanel({ children, footer, instruction, onBack, title }: NodePanelProps) {
  return (
    <section className="node-panel glass-panel" aria-label={title}>
      <header className="node-panel-header">
        <div className="node-panel-title-row">
          <button
            aria-label="Back to previous node"
            className="node-back-button"
            disabled={!onBack}
            onClick={onBack}
            type="button"
          >
            <ArrowLeft size={16} strokeWidth={2.2} />
          </button>
          <h1>{title}</h1>
        </div>
        {instruction ? <p className="node-instruction">{instruction}</p> : null}
      </header>

      <div className="node-panel-body">{children}</div>

      {footer ? <footer className="node-panel-footer">{footer}</footer> : null}
    </section>
  );
}
