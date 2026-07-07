import { DetectAdjustSymmetryPanel } from '../node_panels/DetectAdjustSymmetryPanel';
import { ImageConditionPanel } from '../node_panels/ImageConditionPanel';
import { ManualSymmetryPanel } from '../node_panels/ManualSymmetryPanel';
import { SymShapePanel } from '../node_panels/SymShapePanel';
import { SymSparseStructurePanel } from '../node_panels/SymSparseStructurePanel';
import { TexturePanel } from '../node_panels/TexturePanel';
import { VanillaShapePanel } from '../node_panels/VanillaShapePanel';
import { VanillaSparseStructurePanel } from '../node_panels/VanillaSparseStructurePanel';
import { DagPanel } from '../panels/DagPanel';
import { ModelSelectionPanel } from '../panels/ModelSelectionPanel';
import { NodePanel } from '../panels/NodePanel';
import { ThreeViewer } from '../viewer/ThreeViewer';
import * as Switch from '@radix-ui/react-switch';
import { Moon, Sun } from 'lucide-react';
import type { Dispatch } from 'react';
import {
  detectionInstruction,
  imageConditionInstruction,
  manualSymmetryInstruction,
  symShapeInstruction,
  symSparseStructureInstruction,
  textureInstruction,
  vanillaShapeInstruction,
  vanillaSparseStructureInstruction,
} from '../state';
import type {
  DetectionAction,
  DetectionState,
  ImageConditionState,
  ManualSymmetryAction,
  ManualSymmetryState,
  SymShapeAction,
  SymShapeState,
  SymSparseStructureAction,
  SymSparseStructureState,
  TextureAction,
  TextureState,
  VanillaShapeAction,
  VanillaShapeState,
  VanillaSparseStructureAction,
  VanillaSparseStructureState,
} from '../state';
import type {
  DagEdge,
  DagNode,
  DagStatus,
  ModelId,
  NodeId,
  SymmetryTuple,
  ThemeMode,
  ViewerContent,
} from '../types';

type AppLayoutProps = {
  currentNodeId: NodeId | null;
  dagEdges: DagEdge[];
  dagNodes: DagNode[];
  dagStatus: Record<NodeId, DagStatus>;
  detectionState: DetectionState;
  imageConditionState: ImageConditionState;
  manualState: ManualSymmetryState;
  onDetectFinerSymmetry: () => Promise<void>;
  onDetectMajorAxis: () => Promise<void>;
  onDetectionAction: Dispatch<DetectionAction>;
  onEnterImageCondition: () => void;
  onEnterManualSymmetry: () => void;
  onGenerateCondition: () => void;
  onGenerateSymShape: () => void;
  onGenerateSymSparseStructure: () => void;
  onGenerateTexture: () => void;
  onGenerateVanillaShape: () => void;
  onGenerateVanillaSparseStructure: () => void;
  onImageSelected: (file: Blob | File, name: string) => void;
  onManualAction: Dispatch<ManualSymmetryAction>;
  onSymShapeAction: Dispatch<SymShapeAction>;
  onSymSparseAction: Dispatch<SymSparseStructureAction>;
  onTextureAction: Dispatch<TextureAction>;
  onVanillaShapeAction: Dispatch<VanillaShapeAction>;
  onVanillaSparseAction: Dispatch<VanillaSparseStructureAction>;
  onThemeChange: (theme: ThemeMode) => void;
  selectedModelId: ModelId;
  symShapeState: SymShapeState;
  symShapeTuple: SymmetryTuple;
  symSparseState: SymSparseStructureState;
  symSparseTuple: SymmetryTuple;
  textureState: TextureState;
  vanillaShapeState: VanillaShapeState;
  vanillaSparseState: VanillaSparseStructureState;
  theme: ThemeMode;
};

export function AppLayout({
  currentNodeId,
  dagEdges,
  dagNodes,
  dagStatus,
  detectionState,
  imageConditionState,
  manualState,
  onDetectFinerSymmetry,
  onDetectMajorAxis,
  onDetectionAction,
  onEnterImageCondition,
  onEnterManualSymmetry,
  onGenerateCondition,
  onGenerateSymShape,
  onGenerateSymSparseStructure,
  onGenerateTexture,
  onGenerateVanillaShape,
  onGenerateVanillaSparseStructure,
  onImageSelected,
  onManualAction,
  onSymShapeAction,
  onSymSparseAction,
  onTextureAction,
  onVanillaShapeAction,
  onVanillaSparseAction,
  onThemeChange,
  selectedModelId,
  symShapeState,
  symShapeTuple,
  symSparseState,
  symSparseTuple,
  textureState,
  vanillaShapeState,
  vanillaSparseState,
  theme,
}: AppLayoutProps) {
  const showingManualSymmetry = currentNodeId === 'manual_sym';
  const showingDetection = currentNodeId === 'detect_sym';
  const showingVanillaShape = currentNodeId === 'nat_shape';
  const showingVanillaSparseStructure = currentNodeId === 'nat_ss';
  const showingSymShape = currentNodeId === 'sym_shape';
  const showingSymSparseStructure = currentNodeId === 'sym_ss';
  const showingTexture = currentNodeId === 'texture';
  const viewerSymmetryPreview = showingSymSparseStructure
    ? symSparseTuple
    : showingManualSymmetry
      ? manualState.symmetryPreview
      : showingDetection
        ? detectionState.symmetryPreview
        : null;
  let viewerContent: ViewerContent = { kind: 'empty' };

  if (showingDetection) {
    // MOCK_TEST_GLB_START
    // public/mock/test.glb stands in for the trellis2_native_shape result consumed by detect_sym.
    viewerContent = {
      kind: 'glb',
      material: 'neutral',
      orientation: 'mock_test_glb',
      url: '/mock/test.glb',
    };
    // MOCK_TEST_GLB_END
  } else if (showingSymSparseStructure && symSparseState.generatedOccUrl) {
    // MOCK_SYM_SS_OCC_START
    // public/mock/occ.glb stands in for the generated sym_ss sparse-structure artifact.
    viewerContent = { kind: 'glb', material: 'neutral', url: symSparseState.generatedOccUrl };
    // MOCK_SYM_SS_OCC_END
  } else if (showingVanillaSparseStructure && vanillaSparseState.generatedOccUrl) {
    // MOCK_VANILLA_SS_OCC_START
    // public/mock/occ.glb stands in for the generated vanilla sparse-structure artifact.
    viewerContent = { kind: 'glb', material: 'neutral', url: vanillaSparseState.generatedOccUrl };
    // MOCK_VANILLA_SS_OCC_END
  } else if (showingVanillaShape && vanillaShapeState.generatedShapeUrl) {
    // MOCK_VANILLA_SHAPE_OUTPUT_START
    // public/mock/shape.glb stands in for the generated vanilla shape mesh artifact.
    viewerContent = { kind: 'glb', material: 'neutral', url: vanillaShapeState.generatedShapeUrl };
    // MOCK_VANILLA_SHAPE_OUTPUT_END
  } else if (showingVanillaShape && vanillaShapeState.inputOccUrl) {
    // MOCK_VANILLA_SHAPE_OCC_START
    // public/mock/occ.glb stands in for the vanilla shape input occupancy artifact.
    viewerContent = { kind: 'glb', material: 'neutral', url: vanillaShapeState.inputOccUrl };
    // MOCK_VANILLA_SHAPE_OCC_END
  } else if (showingSymShape && symShapeState.generatedShapeUrl) {
    // MOCK_SYM_SHAPE_OUTPUT_START
    // public/mock/shape.glb stands in for the generated sym_shape mesh artifact.
    viewerContent = { kind: 'glb', material: 'neutral', url: symShapeState.generatedShapeUrl };
    // MOCK_SYM_SHAPE_OUTPUT_END
  } else if (showingSymShape && symShapeState.inputOccUrl) {
    // MOCK_SYM_SHAPE_OCC_START
    // public/mock/occ.glb stands in for the sym_shape input occupancy artifact.
    viewerContent = { kind: 'glb', material: 'neutral', url: symShapeState.inputOccUrl };
    // MOCK_SYM_SHAPE_OCC_END
  } else if (showingTexture && textureState.generatedTextureUrl) {
    // MOCK_TEXTURE_OUTPUT_START
    // public/mock/full.glb stands in for the generated textured mesh artifact.
    viewerContent = { kind: 'glb', material: 'source', url: textureState.generatedTextureUrl };
    // MOCK_TEXTURE_OUTPUT_END
  } else if (showingTexture && textureState.inputShapeUrl) {
    // MOCK_TEXTURE_SHAPE_START
    // public/mock/shape.glb stands in for the texture input shape mesh artifact.
    viewerContent = { kind: 'glb', material: 'neutral', url: textureState.inputShapeUrl };
    // MOCK_TEXTURE_SHAPE_END
  }

  return (
    <div className="app-shell" data-theme={theme}>
      <ThreeViewer
        onOverlayPick={
          showingDetection
            ? (overlayId) => onDetectionAction({ overlayId, type: 'overlayPicked' })
            : () => undefined
        }
        selectableOverlayIds={showingDetection ? detectionState.selectableOverlayIds : []}
        overlays={showingDetection ? detectionState.overlays : []}
        selectedOverlayId={showingDetection ? detectionState.selectedOverlayId : null}
        symmetryPreview={viewerSymmetryPreview}
        theme={theme}
        viewerContent={viewerContent}
      />

      <div className="node-panel-anchor">
        {currentNodeId === null ? (
          <ModelSelectionPanel
            onConfirm={onEnterImageCondition}
            selectedModelId={selectedModelId}
          />
        ) : currentNodeId === 'img_cond' ? (
          <NodePanel
            instruction={imageConditionInstruction(imageConditionState)}
            title="Image condition"
          >
            <ImageConditionPanel
              onEnterManualSymmetry={onEnterManualSymmetry}
              onGenerateCondition={onGenerateCondition}
              onImageSelected={onImageSelected}
              state={imageConditionState}
            />
          </NodePanel>
        ) : showingManualSymmetry ? (
          <NodePanel
            instruction={manualSymmetryInstruction(manualState)}
            title="Manually specify symmetry."
          >
            <ManualSymmetryPanel dispatch={onManualAction} state={manualState} />
          </NodePanel>
        ) : showingDetection ? (
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
        ) : showingVanillaSparseStructure ? (
          <NodePanel
            instruction={vanillaSparseStructureInstruction(vanillaSparseState)}
            title="Vanilla sparse structure generation."
          >
            <VanillaSparseStructurePanel
              dispatch={onVanillaSparseAction}
              onGenerate={onGenerateVanillaSparseStructure}
              state={vanillaSparseState}
            />
          </NodePanel>
        ) : showingVanillaShape ? (
          <NodePanel
            instruction={vanillaShapeInstruction(vanillaShapeState)}
            title="Vanilla shape generation."
          >
            <VanillaShapePanel
              dispatch={onVanillaShapeAction}
              onGenerate={onGenerateVanillaShape}
              state={vanillaShapeState}
            />
          </NodePanel>
        ) : showingSymSparseStructure ? (
          <NodePanel
            instruction={symSparseStructureInstruction(symSparseState)}
            title="Symmetry enforced sparse structure generation"
          >
            <SymSparseStructurePanel
              dispatch={onSymSparseAction}
              onGenerate={onGenerateSymSparseStructure}
              state={symSparseState}
              symmetryTuple={symSparseTuple}
            />
          </NodePanel>
        ) : showingSymShape ? (
          <NodePanel
            instruction={symShapeInstruction(symShapeState)}
            title="Symmetry enforced shape generation."
          >
            <SymShapePanel
              dispatch={onSymShapeAction}
              onGenerate={onGenerateSymShape}
              state={symShapeState}
              symmetryTuple={symShapeTuple}
            />
          </NodePanel>
        ) : showingTexture ? (
          <NodePanel instruction={textureInstruction(textureState)} title="Generate texture.">
            <TexturePanel
              dispatch={onTextureAction}
              onGenerate={onGenerateTexture}
              state={textureState}
            />
          </NodePanel>
        ) : null}
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

      {currentNodeId ? (
        <div className="dag-anchor">
          <DagPanel
            currentNodeId={currentNodeId}
            edges={dagEdges}
            nodes={dagNodes}
            statusByNode={dagStatus}
          />
        </div>
      ) : null}
    </div>
  );
}
