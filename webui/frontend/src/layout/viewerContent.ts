import { artifactByRole } from '../api/artifacts';
import type { ModelDagNode, ModelSpec } from '../models/types';
import type { DetectionState } from '../state/detection';
import type { ManualSymmetryState } from '../state/symmetry';
import type { WorkflowState } from '../state/workflow';
import type { ArtifactRef, SymmetryTuple } from '../types';
import { emptyViewerContent } from '../viewer/viewerTypes';
import type { ViewerContent } from '../viewer/viewerTypes';

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

  if (currentNode.kind === 'detect_adjust_symmetry') {
    return {
      ...contentForModelRule(modelSpec, workflow, currentNode.id),
      overlays: detectionState.overlays,
      selectedOverlayId: detectionState.selectedOverlayId,
      selectableOverlayIds: detectionState.selectableOverlayIds,
      symmetryPreview: detectionState.symmetryPreview,
    };
  }

  if (currentNode.kind === 'manual_symmetry') {
    return {
      ...emptyViewerContent,
      symmetryPreview: manualSymmetryState.symmetryPreview,
    };
  }

  const content = contentForModelRule(modelSpec, workflow, currentNode.id);
  if (modelSpec.viewer[currentNode.id]?.showConfirmedSymmetryPreview) {
    return {
      ...content,
      symmetryPreview: confirmedSymmetry,
    };
  }

  return content;
}

function contentForModelRule(
  modelSpec: ModelSpec,
  workflow: WorkflowState,
  nodeId: string,
): ViewerContent {
  const rule = modelSpec.viewer[nodeId];
  if (!rule) {
    return emptyViewerContent;
  }

  const glb = rule.artifactCandidates.reduce<ViewerContent['glb']>((currentGlb, candidate) => {
    if (currentGlb) {
      return currentGlb;
    }

    return glbContentForArtifact(
      artifactByRole(workflow.nodeRunsByNode[candidate.nodeId]?.artifactRefs ?? {}, candidate.roles),
      candidate.material,
    );
  }, null);

  return {
    ...emptyViewerContent,
    glb,
  };
}

function glbContentForArtifact(artifact: ArtifactRef | null, material: 'neutral' | 'source') {
  return artifact ? { material, url: artifact.url } : null;
}
