import { DetectAdjustSymmetryPanel } from '../node_panels/DetectAdjustSymmetryPanel';
import { DagPanel } from '../panels/DagPanel';
import { NodePanel } from '../panels/NodePanel';
import { ThreeViewer } from '../viewer/ThreeViewer';
import * as Switch from '@radix-ui/react-switch';
import { Moon, Sun } from 'lucide-react';
import type { Dispatch } from 'react';
import { detectionInstruction } from '../state';
import type { DetectionAction, DetectionState } from '../state';
import type { DagEdge, DagNode, DagStatus, NodeId, ThemeMode } from '../types';

type AppLayoutProps = {
  currentNodeId: NodeId;
  dagEdges: DagEdge[];
  dagNodes: DagNode[];
  dagStatus: Record<NodeId, DagStatus>;
  detectionState: DetectionState;
  onDetectFinerSymmetry: () => Promise<void>;
  onDetectMajorAxis: () => Promise<void>;
  onDetectionAction: Dispatch<DetectionAction>;
  onThemeChange: (theme: ThemeMode) => void;
  theme: ThemeMode;
};

export function AppLayout({
  currentNodeId,
  dagEdges,
  dagNodes,
  dagStatus,
  detectionState,
  onDetectFinerSymmetry,
  onDetectMajorAxis,
  onDetectionAction,
  onThemeChange,
  theme,
}: AppLayoutProps) {
  return (
    <div className="app-shell" data-theme={theme}>
      <ThreeViewer
        onOverlayPick={(overlayId) => onDetectionAction({ overlayId, type: 'overlayPicked' })}
        selectableOverlayIds={detectionState.selectableOverlayIds}
        overlays={detectionState.overlays}
        selectedOverlayId={detectionState.selectedOverlayId}
        symmetryPreview={detectionState.symmetryPreview}
        theme={theme}
      />

      <div className="node-panel-anchor">
        <NodePanel
          instruction={detectionInstruction(detectionState)}
          title="Detect and adjust symmetry"
        >
          <DetectAdjustSymmetryPanel
            dispatch={onDetectionAction}
            onDetectFinerSymmetry={onDetectFinerSymmetry}
            onDetectMajorAxis={onDetectMajorAxis}
            state={detectionState}
          />
        </NodePanel>
      </div>

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

      <div className="dag-anchor">
        <DagPanel
          currentNodeId={currentNodeId}
          edges={dagEdges}
          nodes={dagNodes}
          statusByNode={dagStatus}
        />
      </div>
    </div>
  );
}
