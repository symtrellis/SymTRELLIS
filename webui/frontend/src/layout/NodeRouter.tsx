import type { Dispatch } from 'react';
import type { ModelDagNode, NodeInstanceId } from '../models/types';
import type {
  Trellis2ExportParams,
  Trellis2SymmetryShapeAction,
  Trellis2SymmetryShapeState,
  Trellis2SymmetrySparseStructureAction,
  Trellis2SymmetrySparseStructureState,
  Trellis2TextureAction,
  Trellis2TextureState,
  Trellis2VanillaShapeAction,
  Trellis2VanillaShapeState,
  Trellis2VanillaSparseStructureAction,
  Trellis2VanillaSparseStructureState,
} from '../models/trellis2';
import { ExportControls, type ExportState } from '../node_panels/exportControls';
import type { ImageConditionAction, ImageConditionState } from '../state/imageCondition';
import type { DetectionAction, DetectionState } from '../state/detection';
import type { ManualSymmetryAction, ManualSymmetryState } from '../state/symmetry';
import type { WorkflowSuccessorRoute } from '../state/workflow';
import { imageConditionInstruction } from '../state/imageCondition';
import { detectionInstruction } from '../state/detection';
import { manualSymmetryInstruction } from '../state/symmetry';
import { DetectAdjustSymmetryPanel } from '../node_panels/DetectAdjustSymmetryPanel';
import { ImageConditionPanel } from '../node_panels/ImageConditionPanel';
import { ManualSymmetryPanel } from '../node_panels/ManualSymmetryPanel';
import { Trellis2SymmetryShapePanel } from '../node_panels/Trellis2SymmetryShapePanel';
import { Trellis2SymmetrySparseStructurePanel } from '../node_panels/Trellis2SymmetrySparseStructurePanel';
import { Trellis2TexturePanel } from '../node_panels/Trellis2TexturePanel';
import { Trellis2VanillaShapePanel } from '../node_panels/Trellis2VanillaShapePanel';
import { Trellis2VanillaSparseStructurePanel } from '../node_panels/Trellis2VanillaSparseStructurePanel';
import { NodePanel } from '../panels/NodePanel';
import type { SymmetryTuple } from '../types';

type NodeRouterProps = {
  currentNode: ModelDagNode | undefined;
  currentNodeCompleted: boolean;
  detectionState: DetectionState;
  exportStates: Record<NodeInstanceId, ExportState>;
  imageConditionState: ImageConditionState;
  manualSymmetryState: ManualSymmetryState;
  onChooseNextNode: (nodeId: NodeInstanceId) => void;
  onConfirmDetectedSymmetry: () => void;
  onConfirmManualSymmetry: () => void;
  onDetectFinerSymmetry: () => void;
  onDetectMajorAxis: () => void;
  onDetectionAction: Dispatch<DetectionAction>;
  onExportGlb: (nodeId: NodeInstanceId) => void;
  onExportParamsChange: (nodeId: NodeInstanceId, params: Partial<Trellis2ExportParams>) => void;
  onGenerateImageCondition: () => void;
  onGenerateTrellis2SymmetryShape: () => void;
  onGenerateTrellis2SymmetrySparseStructure: () => void;
  onGenerateTrellis2Texture: () => void;
  onGenerateTrellis2VanillaShape: () => void;
  onGenerateTrellis2VanillaSparseStructure: () => void;
  onGoBack?: () => void;
  onImageConditionAction: Dispatch<ImageConditionAction>;
  onManualSymmetryAction: Dispatch<ManualSymmetryAction>;
  onTrellis2SymmetryShapeAction: Dispatch<Trellis2SymmetryShapeAction>;
  onTrellis2SymmetrySparseStructureAction: Dispatch<Trellis2SymmetrySparseStructureAction>;
  onTrellis2TextureAction: Dispatch<Trellis2TextureAction>;
  onTrellis2VanillaShapeAction: Dispatch<Trellis2VanillaShapeAction>;
  onTrellis2VanillaSparseStructureAction: Dispatch<Trellis2VanillaSparseStructureAction>;
  successorRoutes: WorkflowSuccessorRoute[];
  symmetryTuple: SymmetryTuple | null;
  trellis2SymmetryShapeState: Trellis2SymmetryShapeState;
  trellis2SymmetrySparseStructureState: Trellis2SymmetrySparseStructureState;
  trellis2TextureState: Trellis2TextureState;
  trellis2VanillaShapeState: Trellis2VanillaShapeState;
  trellis2VanillaSparseStructureState: Trellis2VanillaSparseStructureState;
};

export function NodeRouter({
  currentNode,
  currentNodeCompleted,
  detectionState,
  exportStates,
  imageConditionState,
  manualSymmetryState,
  onChooseNextNode,
  onConfirmDetectedSymmetry,
  onConfirmManualSymmetry,
  onDetectFinerSymmetry,
  onDetectMajorAxis,
  onDetectionAction,
  onExportGlb,
  onExportParamsChange,
  onGenerateImageCondition,
  onGenerateTrellis2SymmetryShape,
  onGenerateTrellis2SymmetrySparseStructure,
  onGenerateTrellis2Texture,
  onGenerateTrellis2VanillaShape,
  onGenerateTrellis2VanillaSparseStructure,
  onGoBack,
  onImageConditionAction,
  onManualSymmetryAction,
  onTrellis2SymmetryShapeAction,
  onTrellis2SymmetrySparseStructureAction,
  onTrellis2TextureAction,
  onTrellis2VanillaShapeAction,
  onTrellis2VanillaSparseStructureAction,
  successorRoutes,
  symmetryTuple,
  trellis2SymmetryShapeState,
  trellis2SymmetrySparseStructureState,
  trellis2TextureState,
  trellis2VanillaShapeState,
  trellis2VanillaSparseStructureState,
}: NodeRouterProps) {
  if (!currentNode) {
    return null;
  }

  const nextActions = successorRoutes.map((route) => ({
    label: route.edge.routeLabel ?? route.label,
    onClick: () => onChooseNextNode(route.node.id),
  }));

  if (currentNode.kind === 'trellis2_image_condition') {
    return (
      <NodePanel instruction={imageConditionInstruction(imageConditionState)} onBack={onGoBack} title={currentNode.label}>
        <ImageConditionPanel
          onGenerateCondition={onGenerateImageCondition}
          onImageSelected={(file, name) =>
            onImageConditionAction({
              file,
              name,
              type: 'imageSelected',
              url: URL.createObjectURL(file),
            })
          }
          routeActions={successorRoutes.map((route) => ({
            label: route.label,
            onClick: () => onChooseNextNode(route.node.id),
          }))}
          state={imageConditionState}
        />
      </NodePanel>
    );
  }

  if (currentNode.kind === 'trellis2_vanilla_sparse_structure') {
    return (
      <NodePanel
        instruction={Trellis2VanillaSparseStructurePanel.instruction(trellis2VanillaSparseStructureState)}
        onBack={onGoBack}
        title={currentNode.label}
      >
        <Trellis2VanillaSparseStructurePanel
          dispatch={onTrellis2VanillaSparseStructureAction}
          nextActions={nextActions}
          onGenerate={onGenerateTrellis2VanillaSparseStructure}
          state={trellis2VanillaSparseStructureState}
        />
      </NodePanel>
    );
  }

  if (currentNode.kind === 'trellis2_vanilla_shape') {
    return (
      <NodePanel
        instruction={Trellis2VanillaShapePanel.instruction(trellis2VanillaShapeState)}
        onBack={onGoBack}
        title={currentNode.label}
      >
        <Trellis2VanillaShapePanel
          dispatch={onTrellis2VanillaShapeAction}
          exportState={exportStates[currentNode.id] ?? ExportControls.initialState}
          nextActions={nextActions}
          onExport={() => onExportGlb(currentNode.id)}
          onExportParamsChange={(params) => onExportParamsChange(currentNode.id, params)}
          onGenerate={onGenerateTrellis2VanillaShape}
          state={trellis2VanillaShapeState}
        />
      </NodePanel>
    );
  }

  if (currentNode.kind === 'manual_symmetry') {
    return (
      <NodePanel instruction={manualSymmetryInstruction(manualSymmetryState)} onBack={onGoBack} title={currentNode.label}>
        <ManualSymmetryPanel
          dispatch={onManualSymmetryAction}
          nodeReady={currentNodeCompleted}
          onConfirm={onConfirmManualSymmetry}
          onNext={nextActions[0]?.onClick}
          state={manualSymmetryState}
        />
      </NodePanel>
    );
  }

  if (currentNode.kind === 'detect_adjust_symmetry') {
    return (
      <NodePanel instruction={detectionInstruction(detectionState)} onBack={onGoBack} title={currentNode.label}>
        <DetectAdjustSymmetryPanel
          dispatch={onDetectionAction}
          nodeReady={currentNodeCompleted}
          onConfirm={onConfirmDetectedSymmetry}
          onDetectFinerSymmetry={onDetectFinerSymmetry}
          onDetectMajorAxis={onDetectMajorAxis}
          onNext={nextActions[0]?.onClick}
          state={detectionState}
        />
      </NodePanel>
    );
  }

  if (currentNode.kind === 'trellis2_symmetry_sparse_structure') {
    return (
      <NodePanel
        instruction={Trellis2SymmetrySparseStructurePanel.instruction(trellis2SymmetrySparseStructureState)}
        onBack={onGoBack}
        title={currentNode.label}
      >
        <Trellis2SymmetrySparseStructurePanel
          dispatch={onTrellis2SymmetrySparseStructureAction}
          nextActions={nextActions}
          onGenerate={onGenerateTrellis2SymmetrySparseStructure}
          state={trellis2SymmetrySparseStructureState}
          symmetryTuple={symmetryTuple}
        />
      </NodePanel>
    );
  }

  if (currentNode.kind === 'trellis2_symmetry_shape') {
    return (
      <NodePanel
        instruction={Trellis2SymmetryShapePanel.instruction(trellis2SymmetryShapeState)}
        onBack={onGoBack}
        title={currentNode.label}
      >
        <Trellis2SymmetryShapePanel
          dispatch={onTrellis2SymmetryShapeAction}
          exportState={exportStates[currentNode.id] ?? ExportControls.initialState}
          nextActions={nextActions}
          onExport={() => onExportGlb(currentNode.id)}
          onExportParamsChange={(params) => onExportParamsChange(currentNode.id, params)}
          onGenerate={onGenerateTrellis2SymmetryShape}
          state={trellis2SymmetryShapeState}
          symmetryTuple={symmetryTuple}
        />
      </NodePanel>
    );
  }

  if (currentNode.kind === 'trellis2_texture') {
    return (
      <NodePanel
        instruction={Trellis2TexturePanel.instruction(trellis2TextureState)}
        onBack={onGoBack}
        title={currentNode.label}
      >
        <Trellis2TexturePanel
          dispatch={onTrellis2TextureAction}
          exportState={exportStates[currentNode.id] ?? ExportControls.initialState}
          nextActions={nextActions}
          onExport={() => onExportGlb(currentNode.id)}
          onExportParamsChange={(params) => onExportParamsChange(currentNode.id, params)}
          onGenerate={onGenerateTrellis2Texture}
          state={trellis2TextureState}
        />
      </NodePanel>
    );
  }

  return null;
}
