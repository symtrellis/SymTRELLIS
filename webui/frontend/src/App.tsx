import { useEffect, useMemo, useReducer, useState, type Dispatch } from 'react';
import { artifactByRole, uploadInputImage } from './api/artifacts';
import { submitAction, submitNodeRun } from './api/execution';
import { AppLayout } from './layout/AppLayout';
import { NodeRouter } from './layout/NodeRouter';
import { viewerContentForWorkflow } from './layout/viewerContent';
import { modelOptions, modelSpecs } from './models/registry';
import {
  trellis2ArtifactRoles,
  trellis2GenerationDefaults,
  trellis2InitialShapeMetadata,
  trellis2InitialSparseMetadata,
  trellis2InitialTextureMetadata,
  trellis2ShapeMetadata,
  trellis2SparseMetadata,
  trellis2TextureMetadata,
} from './models/trellis2';
import type { Trellis2ExportParams } from './models/trellis2';
import type { NodeInstanceId } from './models/types';
import { ExportControls, type ExportState } from './node_panels/exportControls';
import { ModelSelectionPanel } from './panels/ModelSelectionPanel';
import {
  completeWorkflowNode,
  chooseWorkflowEdge,
  currentNodeCompleted,
  dagStatusForWorkflow,
  enterModelDag,
  canGoBack,
  createClientId,
  createInitialWorkflowState,
  goBackWorkflowNode,
  recordWorkflowAction,
  successorRoutesForWorkflow,
} from './state/workflow';
import { readStoredTheme, writeStoredTheme } from './state/theme';
import {
  imageConditionReducer,
  initialImageConditionState,
} from './state/imageCondition';
import {
  type CommonGenerationParams,
  type GenerationAction,
  generationReducer,
  generationInitialState,
} from './state/generation';
import {
  detectionReducer,
  initialDetectionState,
} from './state/detection';
import {
  initialManualSymmetryState,
  manualSymmetryReducer,
} from './state/symmetry';
import type { WorkflowState } from './state/workflow';
import type {
  FinerSymmetryResult,
  NodeRunRef,
  RotationAxisCandidate,
  SymmetryTuple,
  ThemeMode,
} from './types';

const initialTrellis2VanillaSparseStructureState = generationInitialState(
  trellis2GenerationDefaults.vanillaSparseStructure,
  trellis2InitialSparseMetadata,
);
const initialTrellis2SymmetrySparseStructureState = generationInitialState(
  trellis2GenerationDefaults.symmetrySparseStructure,
  trellis2InitialSparseMetadata,
);
const initialTrellis2VanillaShapeState = generationInitialState(
  trellis2GenerationDefaults.vanillaShape,
  trellis2InitialShapeMetadata,
);
const initialTrellis2SymmetryShapeState = generationInitialState(
  trellis2GenerationDefaults.symmetryShape,
  trellis2InitialShapeMetadata,
);
const initialTrellis2TextureState = generationInitialState(
  trellis2GenerationDefaults.texture,
  trellis2InitialTextureMetadata,
);

const initialExportStates: Record<NodeInstanceId, ExportState> = {
  symmetry_shape: ExportControls.initialState,
  texture: ExportControls.initialState,
  vanilla_shape: ExportControls.initialState,
};

export default function App() {
  const [theme, setTheme] = useState<ThemeMode>(readStoredTheme);
  const [workflow, setWorkflow] = useState<WorkflowState>(createInitialWorkflowState);
  const [imageConditionState, dispatchImageCondition] = useReducer(
    imageConditionReducer,
    initialImageConditionState,
  );
  const [manualSymmetryState, dispatchManualSymmetry] = useReducer(
    manualSymmetryReducer,
    initialManualSymmetryState,
  );
  const [detectionState, dispatchDetection] = useReducer(
    detectionReducer,
    initialDetectionState,
  );
  const [trellis2VanillaSparseStructureState, dispatchTrellis2VanillaSparseStructure] =
    useReducer(generationReducer, initialTrellis2VanillaSparseStructureState);
  const [trellis2SymmetrySparseStructureState, dispatchTrellis2SymmetrySparseStructure] =
    useReducer(generationReducer, initialTrellis2SymmetrySparseStructureState);
  const [trellis2VanillaShapeState, dispatchTrellis2VanillaShape] = useReducer(
    generationReducer,
    initialTrellis2VanillaShapeState,
  );
  const [trellis2SymmetryShapeState, dispatchTrellis2SymmetryShape] = useReducer(
    generationReducer,
    initialTrellis2SymmetryShapeState,
  );
  const [trellis2TextureState, dispatchTrellis2Texture] = useReducer(
    generationReducer,
    initialTrellis2TextureState,
  );
  const [exportStates, setExportStates] = useState<Record<NodeInstanceId, ExportState>>(initialExportStates);
  const selectedModel = modelSpecs[workflow.selectedModelId];
  const currentNode = selectedModel.dag.nodes.find((node) => node.id === workflow.currentNodeId);
  const successorRoutes = useMemo(
    () => successorRoutesForWorkflow(selectedModel, workflow),
    [selectedModel, workflow],
  );
  const currentCompleted = currentNodeCompleted(workflow);
  const currentNodeIsEntry = workflow.currentNodeId === selectedModel.dag.entryNodeId;
  const confirmedSymmetryTuple: SymmetryTuple | null =
    manualSymmetryState.proposedSymmetry ?? detectionState.proposedSymmetry;
  const dagStatus = useMemo(() => dagStatusForWorkflow(selectedModel, workflow), [selectedModel, workflow]);
  const viewerContent = useMemo(
    () =>
      viewerContentForWorkflow({
        confirmedSymmetry: confirmedSymmetryTuple,
        currentNode,
        detectionState,
        manualSymmetryState,
        modelSpec: selectedModel,
        workflow,
      }),
    [
      confirmedSymmetryTuple,
      currentNode,
      detectionState,
      manualSymmetryState,
      selectedModel,
      workflow,
    ],
  );

  useEffect(() => {
    writeStoredTheme(theme);
  }, [theme]);

  useEffect(() => {
    const previewUrl = imageConditionState.previewUrl;

    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [imageConditionState.previewUrl]);

  const handleEnterModelDag = () => {
    setWorkflow((state) => enterModelDag(state, selectedModel));
  };

  const handleChooseNextNode = (nodeId: string) => {
    setWorkflow((state) => chooseWorkflowEdge(state, selectedModel, nodeId));
  };

  const handleExitModelDag = () => {
    resetAllPanelStates();
    setWorkflow(createInitialWorkflowState());
  };

  const handleGoBack = () => {
    const transition = goBackWorkflowNode(workflow);
    resetPanelStatesForNodes(transition.resetNodeIds);
    setWorkflow(transition.state);
  };

  const resetPanelStatesForNodes = (nodeIds: NodeInstanceId[]) => {
    if (nodeIds.includes('image_condition')) {
      dispatchImageCondition({ type: 'conditionResultCleared' });
    }

    if (nodeIds.includes('detect_adjust_symmetry')) {
      dispatchDetection({ type: 'reset' });
    }

    if (nodeIds.includes('manual_symmetry')) {
      dispatchManualSymmetry({ type: 'reset' });
    }

    if (nodeIds.includes('vanilla_sparse_structure')) {
      dispatchTrellis2VanillaSparseStructure({
        state: initialTrellis2VanillaSparseStructureState,
        type: 'reset',
      });
    }

    if (nodeIds.includes('symmetry_sparse_structure')) {
      dispatchTrellis2SymmetrySparseStructure({
        state: initialTrellis2SymmetrySparseStructureState,
        type: 'reset',
      });
    }

    if (nodeIds.includes('vanilla_shape')) {
      dispatchTrellis2VanillaShape({ state: initialTrellis2VanillaShapeState, type: 'reset' });
    }

    if (nodeIds.includes('symmetry_shape')) {
      dispatchTrellis2SymmetryShape({ state: initialTrellis2SymmetryShapeState, type: 'reset' });
    }

    if (nodeIds.includes('texture')) {
      dispatchTrellis2Texture({ state: initialTrellis2TextureState, type: 'reset' });
    }

    setExportStates((states) =>
      Object.fromEntries(Object.entries(states).filter(([nodeId]) => !nodeIds.includes(nodeId))),
    );
  };

  const resetAllPanelStates = () => {
    dispatchImageCondition({ type: 'reset' });
    dispatchDetection({ type: 'reset' });
    dispatchManualSymmetry({ type: 'reset' });
    dispatchTrellis2VanillaSparseStructure({
      state: initialTrellis2VanillaSparseStructureState,
      type: 'reset',
    });
    dispatchTrellis2SymmetrySparseStructure({
      state: initialTrellis2SymmetrySparseStructureState,
      type: 'reset',
    });
    dispatchTrellis2VanillaShape({ state: initialTrellis2VanillaShapeState, type: 'reset' });
    dispatchTrellis2SymmetryShape({ state: initialTrellis2SymmetryShapeState, type: 'reset' });
    dispatchTrellis2Texture({ state: initialTrellis2TextureState, type: 'reset' });
    setExportStates(initialExportStates);
  };

  const handleGenerateImageCondition = async () => {
    const node = currentNode;
    const file = imageConditionState.previewFile;

    if (!node || !file) {
      return;
    }

    dispatchImageCondition({ type: 'conditionGenerationStarted' });

    const uploadResult = await uploadInputImage(file, imageConditionState.previewName);
    if (!uploadResult.ok) {
      dispatchImageCondition({ message: uploadResult.message, type: 'conditionGenerationFailed' });
      return;
    }

    const artifact = uploadResult.value.artifact;
    dispatchImageCondition({ artifact, type: 'inputUploaded' });

    const nodeRunResult = await submitNodeRun({
      executionKind: 'node_run',
      inputArtifactKeys: [artifact.artifactKey],
      modelId: selectedModel.id,
      nodeInstanceId: node.id,
      nodeKind: node.kind,
      operationId: node.operation,
      parentRunKeys: [],
      params: { inputImageKey: artifact.artifactKey },
      requestId: createClientId(),
      sessionId: workflow.sessionId,
      type: 'execution.submit',
    });
    if (!nodeRunResult.ok) {
      dispatchImageCondition({ message: nodeRunResult.message, type: 'conditionGenerationFailed' });
      return;
    }

    const result = nodeRunResult.value;
    // BACKEND_PROTOCOL_PENDING: a successful image-condition run must return
    // artifactRefs.condition only after that artifact is persisted and its url is fetchable.
    const conditionArtifact = result.artifactRefs.condition;
    if (!conditionArtifact) {
      dispatchImageCondition({
        message: 'Image condition artifact was not returned by the backend.',
        type: 'conditionGenerationFailed',
      });
      return;
    }

    const nodeRun = {
      artifactRefs: result.artifactRefs,
      key: result.key,
      metadata: result.metadata,
      operationId: node.operation,
    };
    dispatchImageCondition({
      conditionArtifact,
      nodeRun,
      type: 'conditionGenerated',
    });
    setWorkflow((state) => completeWorkflowNode(state, node.id, nodeRun));
  };

  const handleDetectMajorAxis = async () => {
    dispatchDetection({ type: 'majorDetectionStarted' });

    const actionResult = await submitAction<RotationAxisCandidate[]>({
      actionKind: 'detect_rotation_symmetry',
      executionKind: 'action',
      operationId: 'symmetry.detect_rotation_symmetry',
      params: {},
      requestId: createClientId(),
      sessionId: workflow.sessionId,
      sourceNodeRunKey: workflow.currentNodeRunKey ?? undefined,
      type: 'execution.submit',
    });

    if (!actionResult.ok) {
      dispatchDetection({ message: actionResult.message, type: 'majorDetectionFailed' });
      return;
    }

    const action = actionResult.value;
    setWorkflow((state) =>
      recordWorkflowAction(state, {
        actionKind: 'detect_rotation_symmetry',
        artifactRefs: action.artifactRefs,
        key: action.key,
        metadata: action.metadata,
        operationId: 'symmetry.detect_rotation_symmetry',
      }),
    );
    dispatchDetection({
      actionKey: action.key,
      candidates: action.jsonResult,
      type: 'rotationAxesLoaded',
    });
  };

  const handleDetectFinerSymmetry = async () => {
    dispatchDetection({ type: 'finerDetectionStarted' });

    const actionResult = await submitAction<FinerSymmetryResult>({
      actionKind: 'detect_finer_symmetry',
      executionKind: 'action',
      operationId: 'symmetry.detect_finer_symmetry',
      params: {
        center: detectionState.center,
        fold: detectionState.fold,
        majorAxis: detectionState.majorAxis,
      },
      requestId: createClientId(),
      sessionId: workflow.sessionId,
      sourceNodeRunKey: workflow.currentNodeRunKey ?? undefined,
      type: 'execution.submit',
    });

    if (!actionResult.ok) {
      dispatchDetection({ message: actionResult.message, type: 'finerDetectionFailed' });
      return;
    }

    const action = actionResult.value;
    setWorkflow((state) =>
      recordWorkflowAction(state, {
        actionKind: 'detect_finer_symmetry',
        artifactRefs: action.artifactRefs,
        key: action.key,
        metadata: action.metadata,
        operationId: 'symmetry.detect_finer_symmetry',
      }),
    );
    dispatchDetection({ actionKey: action.key, result: action.jsonResult, type: 'finerResultLoaded' });
  };

  const handleConfirmManualSymmetry = async () => {
    const node = currentNode;
    const symmetry = manualSymmetryState.proposedSymmetry;

    if (!node || !symmetry) {
      return;
    }

    const nodeRunResult = await submitNodeRun({
      executionKind: 'node_run',
      inputArtifactKeys: [],
      modelId: selectedModel.id,
      nodeInstanceId: node.id,
      nodeKind: node.kind,
      operationId: node.operation,
      parentRunKeys: workflow.currentNodeRunKey ? [workflow.currentNodeRunKey] : [],
      params: { symmetry },
      requestId: createClientId(),
      sessionId: workflow.sessionId,
      type: 'execution.submit',
    });

    if (!nodeRunResult.ok) {
      return;
    }

    const result = nodeRunResult.value;
    setWorkflow((state) =>
      completeWorkflowNode(state, node.id, {
        artifactRefs: result.artifactRefs,
        key: result.key,
        metadata: result.metadata,
        operationId: node.operation,
      }),
    );
  };

  const handleConfirmDetectedSymmetry = async () => {
    const node = currentNode;
    const symmetry = detectionState.proposedSymmetry;

    if (!node || !symmetry) {
      return;
    }

    const nodeRunResult = await submitNodeRun({
      executionKind: 'node_run',
      inputArtifactKeys: [],
      modelId: selectedModel.id,
      nodeInstanceId: node.id,
      nodeKind: node.kind,
      operationId: node.operation,
      parentRunKeys: workflow.currentNodeRunKey ? [workflow.currentNodeRunKey] : [],
      params: { symmetry },
      requestId: createClientId(),
      sessionId: workflow.sessionId,
      type: 'execution.submit',
    });

    if (!nodeRunResult.ok) {
      return;
    }

    const result = nodeRunResult.value;
    setWorkflow((state) =>
      completeWorkflowNode(state, node.id, {
        artifactRefs: result.artifactRefs,
        key: result.key,
        metadata: result.metadata,
        operationId: node.operation,
      }),
    );
  };

  const submitGenerationNode = async <Params extends CommonGenerationParams, Metadata>(
    dispatch: Dispatch<GenerationAction<Params, Metadata>>,
    metadataFromResponse: (metadata: Record<string, unknown>) => Metadata,
    params: Record<string, unknown>,
    outputRoles: string[],
  ) => {
    const node = currentNode;

    if (!node) {
      return;
    }

    dispatch({ type: 'generationStarted' });

    const nodeRunResult = await submitNodeRun({
      executionKind: 'node_run',
      inputArtifactKeys: [],
      modelId: selectedModel.id,
      nodeInstanceId: node.id,
      nodeKind: node.kind,
      operationId: node.operation,
      parentRunKeys: workflow.currentNodeRunKey ? [workflow.currentNodeRunKey] : [],
      params,
      requestId: createClientId(),
      sessionId: workflow.sessionId,
      type: 'execution.submit',
    });

    if (!nodeRunResult.ok) {
      dispatch({ message: nodeRunResult.message, type: 'generationFailed' });
      return;
    }

    const result = nodeRunResult.value;
    const nodeRun: NodeRunRef = {
      artifactRefs: result.artifactRefs,
      key: result.key,
      metadata: result.metadata,
      operationId: node.operation,
    };
    const outputArtifact = artifactByRole(nodeRun.artifactRefs, outputRoles);
    dispatch({
      metadata: metadataFromResponse(result.metadata),
      nodeRun,
      outputArtifact,
      type: 'generationFinished',
    });
    setWorkflow((workflowState) => completeWorkflowNode(workflowState, node.id, nodeRun));
  };

  const handleGenerateTrellis2VanillaSparseStructure = () =>
    submitGenerationNode(
      dispatchTrellis2VanillaSparseStructure,
      trellis2SparseMetadata,
      trellis2VanillaSparseStructureState.params,
      trellis2ArtifactRoles.sparseStructureMesh,
    );

  const handleGenerateTrellis2SymmetrySparseStructure = () => {
    if (!confirmedSymmetryTuple) {
      dispatchTrellis2SymmetrySparseStructure({
        message: 'Confirm a symmetry tuple before symmetry enforced sparse structure generation.',
        type: 'generationFailed',
      });
      return;
    }

    return submitGenerationNode(
      dispatchTrellis2SymmetrySparseStructure,
      trellis2SparseMetadata,
      {
        ...trellis2SymmetrySparseStructureState.params,
        symmetry: confirmedSymmetryTuple,
      },
      trellis2ArtifactRoles.sparseStructureMesh,
    );
  };

  const handleGenerateTrellis2VanillaShape = () =>
    submitGenerationNode(
      dispatchTrellis2VanillaShape,
      trellis2ShapeMetadata,
      trellis2VanillaShapeState.params,
      trellis2ArtifactRoles.shapeMesh,
    );

  const handleGenerateTrellis2SymmetryShape = () => {
    if (!confirmedSymmetryTuple) {
      dispatchTrellis2SymmetryShape({
        message: 'Confirm a symmetry tuple before symmetry enforced shape generation.',
        type: 'generationFailed',
      });
      return;
    }

    return submitGenerationNode(
      dispatchTrellis2SymmetryShape,
      trellis2ShapeMetadata,
      {
        ...trellis2SymmetryShapeState.params,
        symmetry: confirmedSymmetryTuple,
      },
      trellis2ArtifactRoles.shapeMesh,
    );
  };

  const handleGenerateTrellis2Texture = () =>
    submitGenerationNode(
      dispatchTrellis2Texture,
      trellis2TextureMetadata,
      trellis2TextureState.params,
      trellis2ArtifactRoles.texturedMesh,
    );

  const handleExportParamsChange = (
    nodeId: NodeInstanceId,
    params: Partial<Trellis2ExportParams>,
  ) => {
    setExportStates((states) => ({
      ...states,
      [nodeId]: {
        ...(states[nodeId] ?? ExportControls.initialState),
        artifact: null,
        bundleArtifact: null,
        errorMessage: '',
        log: '',
        params: {
          ...(states[nodeId] ?? ExportControls.initialState).params,
          ...params,
        },
        progress: 0,
        status: 'idle',
      },
    }));
  };

  const handleExportGlb = async (nodeId: NodeInstanceId) => {
    const exportState = exportStates[nodeId] ?? ExportControls.initialState;
    const sourceRun = workflow.nodeRunsByNode[nodeId];

    if (!sourceRun) {
      return;
    }

    setExportStates((states) => ({
      ...states,
      [nodeId]: {
        ...exportState,
        artifact: null,
        bundleArtifact: null,
        errorMessage: '',
        log: 'Extracting GLB...',
        progress: 0,
        status: 'running',
      },
    }));

    // BACKEND_PROTOCOL_PENDING: action progress/log websocket events are not wired yet.
    // Until that contract is finalized, export progress is updated from the final response.
    const actionResult = await submitAction({
      actionKind: 'export',
      executionKind: 'action',
      operationId: 'trellis2.export_glb',
      params: exportState.params,
      requestId: createClientId(),
      sessionId: workflow.sessionId,
      sourceNodeRunKey: sourceRun.key,
      type: 'execution.submit',
    });

    if (!actionResult.ok) {
      setExportStates((states) => ({
        ...states,
        [nodeId]: {
          ...exportState,
          artifact: null,
          bundleArtifact: null,
          errorMessage: actionResult.message,
          log: actionResult.message,
          progress: 0,
          status: 'failed',
        },
      }));
      return;
    }

    const action = actionResult.value;
    const artifact = artifactByRole(action.artifactRefs, trellis2ArtifactRoles.exportGlb);
    const bundleArtifact = artifactByRole(action.artifactRefs, trellis2ArtifactRoles.exportBundle);
    setWorkflow((state) =>
      recordWorkflowAction(state, {
        actionKind: 'export',
        artifactRefs: action.artifactRefs,
        key: action.key,
        metadata: action.metadata,
        operationId: 'trellis2.export_glb',
      }),
    );
    setExportStates((states) => ({
      ...states,
      [nodeId]: {
        ...exportState,
        artifact,
        bundleArtifact,
        errorMessage: '',
        log: 'GLB extraction finished.',
        progress: 1,
        status: 'ready',
      },
    }));
  };

  const handleOverlayPicked = (overlayId: string) => {
    if (currentNode?.kind === 'detect_adjust_symmetry') {
      dispatchDetection({ overlayId, type: 'overlayPicked' });
    }
  };

  const nodePanel = currentNode ? (
    <NodeRouter
      currentNode={currentNode}
      currentNodeCompleted={currentCompleted}
      detectionState={detectionState}
      exportStates={exportStates}
      imageConditionState={imageConditionState}
      manualSymmetryState={manualSymmetryState}
      onConfirmDetectedSymmetry={handleConfirmDetectedSymmetry}
      onConfirmManualSymmetry={handleConfirmManualSymmetry}
      onDetectFinerSymmetry={handleDetectFinerSymmetry}
      onDetectMajorAxis={handleDetectMajorAxis}
      onDetectionAction={dispatchDetection}
      onChooseNextNode={handleChooseNextNode}
      onExportGlb={handleExportGlb}
      onExportParamsChange={handleExportParamsChange}
      onGoBack={currentNodeIsEntry ? handleExitModelDag : canGoBack(workflow) ? handleGoBack : undefined}
      onGenerateImageCondition={handleGenerateImageCondition}
      onGenerateTrellis2SymmetryShape={handleGenerateTrellis2SymmetryShape}
      onGenerateTrellis2SymmetrySparseStructure={handleGenerateTrellis2SymmetrySparseStructure}
      onGenerateTrellis2Texture={handleGenerateTrellis2Texture}
      onGenerateTrellis2VanillaShape={handleGenerateTrellis2VanillaShape}
      onGenerateTrellis2VanillaSparseStructure={handleGenerateTrellis2VanillaSparseStructure}
      onImageConditionAction={dispatchImageCondition}
      onManualSymmetryAction={dispatchManualSymmetry}
      onTrellis2SymmetryShapeAction={dispatchTrellis2SymmetryShape}
      onTrellis2SymmetrySparseStructureAction={dispatchTrellis2SymmetrySparseStructure}
      onTrellis2TextureAction={dispatchTrellis2Texture}
      onTrellis2VanillaShapeAction={dispatchTrellis2VanillaShape}
      onTrellis2VanillaSparseStructureAction={dispatchTrellis2VanillaSparseStructure}
      successorRoutes={successorRoutes}
      symmetryTuple={confirmedSymmetryTuple}
      trellis2SymmetryShapeState={trellis2SymmetryShapeState}
      trellis2SymmetrySparseStructureState={trellis2SymmetrySparseStructureState}
      trellis2TextureState={trellis2TextureState}
      trellis2VanillaShapeState={trellis2VanillaShapeState}
      trellis2VanillaSparseStructureState={trellis2VanillaSparseStructureState}
    />
  ) : (
    <ModelSelectionPanel
      modelOptions={modelOptions}
      onConfirm={handleEnterModelDag}
      selectedModelId={workflow.selectedModelId}
    />
  );

  return (
    <AppLayout
      chosenEdgeIds={workflow.chosenEdges.map((edge) => edge.id)}
      currentNodeId={workflow.currentNodeId}
      dagEdges={selectedModel.dag.edges}
      dagLayout={selectedModel.dag.layout}
      dagNodes={selectedModel.dag.nodes}
      dagStatus={dagStatus}
      nodePanel={nodePanel}
      onOverlayPicked={handleOverlayPicked}
      onThemeChange={setTheme}
      theme={theme}
      viewerContent={viewerContent}
    />
  );
}
