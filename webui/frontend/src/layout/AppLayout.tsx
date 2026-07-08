import * as Switch from '@radix-ui/react-switch';
import { Moon, Sun } from 'lucide-react';
import type { ReactNode } from 'react';
import type { DagStatusByNode, ModelDagEdge, ModelDagLayout, ModelDagNode, NodeInstanceId } from '../models/types';
import type { ThemeMode } from '../types';
import { DagPanel } from '../panels/DagPanel';
import { ThreeViewer } from '../viewer/ThreeViewer';
import type { ViewerContent } from '../viewer/viewerTypes';

type AppLayoutProps = {
  chosenEdgeIds: string[];
  currentNodeId: NodeInstanceId | null;
  dagEdges: ModelDagEdge[];
  dagLayout: ModelDagLayout;
  dagNodes: ModelDagNode[];
  dagStatus: DagStatusByNode;
  nodePanel: ReactNode;
  onOverlayPicked?: (overlayId: string) => void;
  onThemeChange: (theme: ThemeMode) => void;
  theme: ThemeMode;
  viewerContent: ViewerContent;
};

export function AppLayout({
  chosenEdgeIds,
  currentNodeId,
  dagEdges,
  dagLayout,
  dagNodes,
  dagStatus,
  nodePanel,
  onOverlayPicked,
  onThemeChange,
  theme,
  viewerContent,
}: AppLayoutProps) {
  return (
    <div className="app-shell" data-theme={theme}>
      <ThreeViewer
        content={viewerContent}
        dagVisible={Boolean(currentNodeId)}
        onOverlayPicked={onOverlayPicked}
        theme={theme}
      />

      <div className="node-panel-anchor">{nodePanel}</div>

      <div className="theme-switch-anchor">
        <Switch.Root
          aria-label="Toggle color theme"
          checked={theme === 'dark'}
          className="theme-switch"
          onCheckedChange={(checked) => onThemeChange(checked ? 'dark' : 'light')}
        >
          <span className="theme-switch-icon theme-switch-icon--sun" aria-hidden="true">
            <Sun size={13} strokeWidth={2.2} />
          </span>
          <span className="theme-switch-icon theme-switch-icon--moon" aria-hidden="true">
            <Moon size={13} strokeWidth={2.2} />
          </span>
          <Switch.Thumb className="theme-switch-thumb" />
        </Switch.Root>
      </div>

      {currentNodeId ? (
        <div className="dag-anchor">
          <DagPanel
            chosenEdgeIds={chosenEdgeIds}
            currentNodeId={currentNodeId}
            edges={dagEdges}
            layout={dagLayout}
            nodes={dagNodes}
            statusByNode={dagStatus}
          />
        </div>
      ) : null}
    </div>
  );
}
