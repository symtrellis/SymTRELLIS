import { outputByRole } from '../api/storage';
import type { ModelDagNode, ModelSpec } from '../models/types';
import type { DetectionState } from '../state/detection';
import type { ManualSymmetryState } from '../state/symmetry';
import type { WorkflowState } from '../state/workflow';
import type { SymmetryTuple } from '../types';
import { emptyViewerContent, type ViewerContent } from '../viewer/viewerTypes';

type ViewerContentForWorkflowInput = {
  confirmedSymmetry: SymmetryTuple | null;
  currentNode: ModelDagNode | undefined;
  detectionState: DetectionState;
  manualSymmetryState: ManualSymmetryState;
  modelSpec: ModelSpec;
  workflow: WorkflowState;
};

export function viewerContentForWorkflow({
  confirmedSymmetry,
  currentNode,
  detectionState,
  manualSymmetryState,
  modelSpec,
  workflow,
}: ViewerContentForWorkflowInput): ViewerContent {
  if (!currentNode) {
    return emptyViewerContent;
  }

  const rule = modelSpec.viewer[currentNode.id];
  let glb: ViewerContent['glb'] = null;

  if (rule) {
    for (const candidate of rule.outputCandidates) {
      const nodeRun = workflow.nodeRunsByNode[candidate.nodeId];
      const output = nodeRun ? outputByRole(nodeRun.outputs, candidate.roles) : null;

      if (output) {
        glb = {
          material: candidate.material,
          url: output.url,
        };
        break;
      }
    }
  }

  const baseContent: ViewerContent = {
    ...emptyViewerContent,
    glb,
  };

  if (currentNode.kind === 'detect_adjust_symmetry') {
    return {
      ...baseContent,
      overlays: detectionState.overlays,
      selectableOverlayIds: detectionState.selectableOverlayIds,
      selectedOverlayId: detectionState.selectedOverlayId,
      symmetryPreview: detectionState.symmetryPreview,
    };
  }

  if (currentNode.kind === 'manual_symmetry') {
    return {
      ...baseContent,
      symmetryPreview: manualSymmetryState.symmetryPreview,
    };
  }

  if (rule?.showConfirmedSymmetryPreview) {
    return {
      ...baseContent,
      symmetryPreview: confirmedSymmetry,
    };
  }

  return baseContent;
}
