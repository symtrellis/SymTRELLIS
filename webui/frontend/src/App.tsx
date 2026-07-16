import {
  useCallback,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
  type Dispatch,
  type SetStateAction,
} from 'react';
import { restoreSession, submitAction, submitNodeRun } from './api/execution';
import { uploadInputImage } from './api/storage';
import { AppLayout } from './layout/AppLayout';
import { NodeRouter } from './layout/NodeRouter';
import { viewerContentForWorkflow } from './layout/viewerContent';
import { modelOptions, modelSpecs } from './models/registry';
import {
  trellis2GenerationDefaults,
  trellis2InitialShapeMetadata,
  trellis2InitialSparseMetadata,
  trellis2InitialTextureMetadata,
  trellis2OperationIds,
  trellis2ShapeMetadata,
  trellis2SparseMetadata,
  trellis2TextureMetadata,
  type Trellis2ExportParams,
  type Trellis2ShapeMetadata,
  type Trellis2SparseMetadata,
  type Trellis2SymmetryShapeParams,
  type Trellis2SymmetryShapeAction,
  type Trellis2SymmetrySparseStructureParams,
  type Trellis2SymmetrySparseStructureAction,
  type Trellis2TextureAction,
  type Trellis2TextureMetadata,
  type Trellis2TextureParams,
  type Trellis2VanillaShapeAction,
  type Trellis2VanillaShapeParams,
  type Trellis2VanillaSparseStructureAction,
  type Trellis2VanillaSparseStructureParams,
} from './models/trellis2';
import type { NodeInstanceId } from './models/types';
import { ExportControls, type ExportState } from './node_panels/exportControls';
import { ModelSelectionPanel } from './panels/ModelSelectionPanel';
import {
  chosenEdgeIdsForWorkflow,
  chooseWorkflowEdge,
  completeWorkflowNode,
  createInitialWorkflowState,
  currentNodeCompleted,
  dagStatusForWorkflow,
  enterModelDag,
  goBackWorkflowNode,
  latestRunKeyForWorkflow,
  parentRunKeysForCurrentNode,
  recordWorkflowAction,
  restoreWorkflowSession,
  successorRoutesForWorkflow,
  workflowUrl,
  type WorkflowState,
} from './state/workflow';
import { createInitialGenerationState, generationReducer } from './state/generation';
import { readStoredTheme, writeStoredTheme } from './state/theme';
import {
  createInitialImageConditionState,
  imageConditionReducer,
  type ImageConditionAction,
} from './state/imageCondition';
import {
  detectionReducer,
  initialDetectionState,
  type DetectionAction,
} from './state/detection';
import {
  initialManualSymmetryState,
  manualSymmetryReducer,
  type ManualSymmetryAction,
} from './state/symmetry';
import type {
  FinerSymmetryDetectionResult,
  NodeRunKey,
  NodeRunResult,
  ReflectionPlaneDetectionResult,
  RequestId,
  RotationAxisDetectionResult,
  SessionId,
  SymmetryTuple,
  ThemeMode,
} from './types';

const initialTrellis2VanillaSparseStructureState = createInitialGenerationState<
  Trellis2VanillaSparseStructureParams,
  Trellis2SparseMetadata
>(trellis2GenerationDefaults.vanillaSparseStructure, trellis2InitialSparseMetadata);
const initialTrellis2SymmetrySparseStructureState = createInitialGenerationState<
  Trellis2SymmetrySparseStructureParams,
  Trellis2SparseMetadata
>(trellis2GenerationDefaults.symmetrySparseStructure, trellis2InitialSparseMetadata);
const initialTrellis2VanillaShapeState = createInitialGenerationState<
  Trellis2VanillaShapeParams,
  Trellis2ShapeMetadata
>(trellis2GenerationDefaults.vanillaShape, trellis2InitialShapeMetadata);
const initialTrellis2SymmetryShapeState = createInitialGenerationState<
  Trellis2SymmetryShapeParams,
  Trellis2ShapeMetadata
>(trellis2GenerationDefaults.symmetryShape, trellis2InitialShapeMetadata);
const initialTrellis2TextureState = createInitialGenerationState<
  Trellis2TextureParams,
  Trellis2TextureMetadata
>(trellis2GenerationDefaults.texture, trellis2InitialTextureMetadata);

function newRequestId(): RequestId {
  return `request_${Date.now().toString(36)}_${Math.random().toString(36).slice(2)}` as RequestId;
}

function resetPanelStates(
  dispatchers: {
    detection: Dispatch<DetectionAction>;
    imageCondition: Dispatch<ImageConditionAction>;
    manualSymmetry: Dispatch<ManualSymmetryAction>;
    symmetryShape: Dispatch<Trellis2SymmetryShapeAction>;
    symmetrySparseStructure: Dispatch<Trellis2SymmetrySparseStructureAction>;
    texture: Dispatch<Trellis2TextureAction>;
    vanillaShape: Dispatch<Trellis2VanillaShapeAction>;
    vanillaSparseStructure: Dispatch<Trellis2VanillaSparseStructureAction>;
  },
  nodeIds: NodeInstanceId[],
  setExportStates: Dispatch<SetStateAction<Record<NodeInstanceId, ExportState>>>,
) {
  if (nodeIds.includes('image_condition')) {
    dispatchers.imageCondition({ type: 'resetToNodeStart' });
  }

  if (nodeIds.includes('manual_symmetry')) {
    dispatchers.manualSymmetry({ type: 'reset' });
  }

  if (nodeIds.includes('detect_adjust_symmetry')) {
    dispatchers.detection({ type: 'reset' });
  }

  if (nodeIds.includes('vanilla_sparse_structure')) {
    dispatchers.vanillaSparseStructure({
      metadata: trellis2InitialSparseMetadata,
      params: trellis2GenerationDefaults.vanillaSparseStructure,
      type: 'resetToNodeStart',
    });
  }

  if (nodeIds.includes('symmetry_sparse_structure')) {
    dispatchers.symmetrySparseStructure({
      metadata: trellis2InitialSparseMetadata,
      params: trellis2GenerationDefaults.symmetrySparseStructure,
      type: 'resetToNodeStart',
    });
  }

  if (nodeIds.includes('vanilla_shape')) {
    dispatchers.vanillaShape({
      metadata: trellis2InitialShapeMetadata,
      params: trellis2GenerationDefaults.vanillaShape,
      type: 'resetToNodeStart',
    });
  }

  if (nodeIds.includes('symmetry_shape')) {
    dispatchers.symmetryShape({
      metadata: trellis2InitialShapeMetadata,
      params: trellis2GenerationDefaults.symmetryShape,
      type: 'resetToNodeStart',
    });
  }

  if (nodeIds.includes('texture')) {
    dispatchers.texture({
      metadata: trellis2InitialTextureMetadata,
      params: trellis2GenerationDefaults.texture,
      type: 'resetToNodeStart',
    });
  }

  setExportStates((states) =>
    Object.fromEntries(
      Object.entries(states).filter(([nodeId]) => !nodeIds.includes(nodeId)),
    ),
  );
}

export default function App() {
  const restoredOnce = useRef(false);
  const sessionMutationInFlightRef = useRef(false);
  const [urlSyncReady, setUrlSyncReady] = useState(false);
  const [sessionMutationInFlight, setSessionMutationInFlight] = useState(false);
  const [theme, setTheme] = useState<ThemeMode>(readStoredTheme);
  const [workflow, setWorkflow] = useState<WorkflowState>(createInitialWorkflowState);
  const [imageConditionState, dispatchImageCondition] = useReducer(
    imageConditionReducer,
    createInitialImageConditionState(),
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
    useReducer(
      generationReducer<Trellis2VanillaSparseStructureParams, Trellis2SparseMetadata>,
      initialTrellis2VanillaSparseStructureState,
    );
  const [trellis2SymmetrySparseStructureState, dispatchTrellis2SymmetrySparseStructure] =
    useReducer(
      generationReducer<Trellis2SymmetrySparseStructureParams, Trellis2SparseMetadata>,
      initialTrellis2SymmetrySparseStructureState,
    );
  const [trellis2VanillaShapeState, dispatchTrellis2VanillaShape] = useReducer(
    generationReducer<Trellis2VanillaShapeParams, Trellis2ShapeMetadata>,
    initialTrellis2VanillaShapeState,
  );
  const [trellis2SymmetryShapeState, dispatchTrellis2SymmetryShape] = useReducer(
    generationReducer<Trellis2SymmetryShapeParams, Trellis2ShapeMetadata>,
    initialTrellis2SymmetryShapeState,
  );
  const [trellis2TextureState, dispatchTrellis2Texture] = useReducer(
    generationReducer<Trellis2TextureParams, Trellis2TextureMetadata>,
    initialTrellis2TextureState,
  );
  const [exportStates, setExportStates] = useState<Record<NodeInstanceId, ExportState>>({
    symmetry_shape: ExportControls.initialState,
    texture: ExportControls.initialState,
    vanilla_shape: ExportControls.initialState,
  });
  const panelStateDispatchers = {
    detection: dispatchDetection,
    imageCondition: dispatchImageCondition,
    manualSymmetry: dispatchManualSymmetry,
    symmetryShape: dispatchTrellis2SymmetryShape,
    symmetrySparseStructure: dispatchTrellis2SymmetrySparseStructure,
    texture: dispatchTrellis2Texture,
    vanillaShape: dispatchTrellis2VanillaShape,
    vanillaSparseStructure: dispatchTrellis2VanillaSparseStructure,
  };

  const selectedModel = modelSpecs[workflow.selectedModelId];
  const currentNode = selectedModel.dag.nodes.find((node) => node.id === workflow.currentNodeId);
  const successorRoutes = useMemo(
    () => successorRoutesForWorkflow(selectedModel, workflow),
    [selectedModel, workflow],
  );
  const currentCompleted = currentNodeCompleted(workflow);
  const confirmedSymmetryTuple: SymmetryTuple | null =
    (workflow.nodeRunsByNode.detect_adjust_symmetry?.jsonResult as
      | SymmetryTuple
      | null
      | undefined) ??
    (workflow.nodeRunsByNode.manual_symmetry?.jsonResult as SymmetryTuple | null | undefined) ??
    detectionState.proposedSymmetry ??
    manualSymmetryState.proposedSymmetry;
  const dagStatus = useMemo(
    () => dagStatusForWorkflow(selectedModel, workflow),
    [selectedModel, workflow],
  );
  const chosenEdgeIds = useMemo(() => chosenEdgeIdsForWorkflow(workflow), [workflow]);
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

  const handleSessionExpired = useCallback(() => {
    const params = new URLSearchParams(window.location.search);
    params.delete('session');
    params.delete('key');
    const search = params.toString();
    window.history.replaceState(null, '', search ? `/?${search}` : '/');

    dispatchImageCondition({ type: 'resetSession' });
    dispatchManualSymmetry({ type: 'reset' });
    dispatchDetection({ type: 'reset' });
    dispatchTrellis2VanillaSparseStructure({
      metadata: trellis2InitialSparseMetadata,
      params: trellis2GenerationDefaults.vanillaSparseStructure,
      type: 'resetToNodeStart',
    });
    dispatchTrellis2SymmetrySparseStructure({
      metadata: trellis2InitialSparseMetadata,
      params: trellis2GenerationDefaults.symmetrySparseStructure,
      type: 'resetToNodeStart',
    });
    dispatchTrellis2VanillaShape({
      metadata: trellis2InitialShapeMetadata,
      params: trellis2GenerationDefaults.vanillaShape,
      type: 'resetToNodeStart',
    });
    dispatchTrellis2SymmetryShape({
      metadata: trellis2InitialShapeMetadata,
      params: trellis2GenerationDefaults.symmetryShape,
      type: 'resetToNodeStart',
    });
    dispatchTrellis2Texture({
      metadata: trellis2InitialTextureMetadata,
      params: trellis2GenerationDefaults.texture,
      type: 'resetToNodeStart',
    });
    setExportStates({});
    setWorkflow(enterModelDag(createInitialWorkflowState(), selectedModel));
    sessionMutationInFlightRef.current = false;
    setSessionMutationInFlight(false);
    setUrlSyncReady(true);
  }, [selectedModel]);

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

  useEffect(() => {
    if (restoredOnce.current) {
      return;
    }

    restoredOnce.current = true;
    void (async () => {
      const params = new URLSearchParams(window.location.search);
      const sessionId = params.get('session') as SessionId | null;
      const key = params.get('key') as NodeRunKey | null;
      const nodeId = params.get('node') as NodeInstanceId | null;

      if (!sessionId) {
        if (nodeId === selectedModel.dag.entryNodeId) {
          setWorkflow((state) => enterModelDag(state, selectedModel));
        }
        setUrlSyncReady(true);
        return;
      }

      const result = await restoreSession(sessionId, key ?? undefined);

      if (!result.ok) {
        if (result.kind === 'session_expired') {
          handleSessionExpired();
          return;
        }

        if (result.kind === 'not_found') {
          params.delete('session');
          params.delete('key');
          const search = params.toString();
          window.history.replaceState(null, '', search ? `/?${search}` : '/');
          setUrlSyncReady(true);
          return;
        }

        setWorkflow((state) => enterModelDag(state, selectedModel));
        dispatchImageCondition({
          message: `Session restore failed: ${result.message}`,
          type: 'conditionGenerationFailed',
        });
        setUrlSyncReady(true);
        return;
      }

      if (result.value.modelId !== 'trellis2') {
        setUrlSyncReady(true);
        return;
      }

      const model = modelSpecs[result.value.modelId];
      const restoredWorkflow = restoreWorkflowSession(model, result.value, nodeId);
      setWorkflow(restoredWorkflow);

      const imageConditionRun = restoredWorkflow.nodeRunsByNode.image_condition;
      if (imageConditionRun) {
        const imageOutput = imageConditionRun.outputs.image_png;
        const restoredRun: NodeRunResult = {
          cached: true,
          jsonResult: imageConditionRun.jsonResult,
          key: imageConditionRun.key,
          metadata: imageConditionRun.metadata,
          outputs: imageConditionRun.outputs,
          sessionId: result.value.sessionId,
          sessionRevision: restoredWorkflow.sessionRevision,
        };
        dispatchImageCondition({
          filename: imageConditionRun.metadata.sourceFilename as string,
          previewUrl: imageOutput.url,
          run: restoredRun,
          type: 'conditionRestored',
        });
      }

      const vanillaSparseStructureRun =
        restoredWorkflow.nodeRunsByNode.vanilla_sparse_structure;
      if (vanillaSparseStructureRun) {
        dispatchTrellis2VanillaSparseStructure({
          metadata: trellis2SparseMetadata(vanillaSparseStructureRun.metadata),
          params: vanillaSparseStructureRun.params as Trellis2VanillaSparseStructureParams,
          result: {
            cached: true,
            jsonResult: vanillaSparseStructureRun.jsonResult,
            key: vanillaSparseStructureRun.key,
            metadata: vanillaSparseStructureRun.metadata,
            outputs: vanillaSparseStructureRun.outputs,
            sessionId: result.value.sessionId,
            sessionRevision: restoredWorkflow.sessionRevision,
          },
          type: 'generationRestored',
        });
      }

      const symmetrySparseStructureRun =
        restoredWorkflow.nodeRunsByNode.symmetry_sparse_structure;
      if (symmetrySparseStructureRun) {
        dispatchTrellis2SymmetrySparseStructure({
          metadata: trellis2SparseMetadata(symmetrySparseStructureRun.metadata),
          params: symmetrySparseStructureRun.params as Trellis2SymmetrySparseStructureParams,
          result: {
            cached: true,
            jsonResult: symmetrySparseStructureRun.jsonResult,
            key: symmetrySparseStructureRun.key,
            metadata: symmetrySparseStructureRun.metadata,
            outputs: symmetrySparseStructureRun.outputs,
            sessionId: result.value.sessionId,
            sessionRevision: restoredWorkflow.sessionRevision,
          },
          type: 'generationRestored',
        });
      }

      const vanillaShapeRun = restoredWorkflow.nodeRunsByNode.vanilla_shape;
      if (vanillaShapeRun) {
        dispatchTrellis2VanillaShape({
          metadata: trellis2ShapeMetadata(vanillaShapeRun.metadata),
          params: vanillaShapeRun.params as Trellis2VanillaShapeParams,
          result: {
            cached: true,
            jsonResult: vanillaShapeRun.jsonResult,
            key: vanillaShapeRun.key,
            metadata: vanillaShapeRun.metadata,
            outputs: vanillaShapeRun.outputs,
            sessionId: result.value.sessionId,
            sessionRevision: restoredWorkflow.sessionRevision,
          },
          type: 'generationRestored',
        });
      }

      const symmetryShapeRun = restoredWorkflow.nodeRunsByNode.symmetry_shape;
      if (symmetryShapeRun) {
        dispatchTrellis2SymmetryShape({
          metadata: trellis2ShapeMetadata(symmetryShapeRun.metadata),
          params: symmetryShapeRun.params as Trellis2SymmetryShapeParams,
          result: {
            cached: true,
            jsonResult: symmetryShapeRun.jsonResult,
            key: symmetryShapeRun.key,
            metadata: symmetryShapeRun.metadata,
            outputs: symmetryShapeRun.outputs,
            sessionId: result.value.sessionId,
            sessionRevision: restoredWorkflow.sessionRevision,
          },
          type: 'generationRestored',
        });
      }

      const textureRun = restoredWorkflow.nodeRunsByNode.texture;
      if (textureRun) {
        dispatchTrellis2Texture({
          metadata: trellis2TextureMetadata(textureRun.metadata),
          params: textureRun.params as Trellis2TextureParams,
          result: {
            cached: true,
            jsonResult: textureRun.jsonResult,
            key: textureRun.key,
            metadata: textureRun.metadata,
            outputs: textureRun.outputs,
            sessionId: result.value.sessionId,
            sessionRevision: restoredWorkflow.sessionRevision,
          },
          type: 'generationRestored',
        });
      }

      const manualSymmetryRun = restoredWorkflow.nodeRunsByNode.manual_symmetry;
      if (manualSymmetryRun) {
        dispatchManualSymmetry({
          symmetry: manualSymmetryRun.jsonResult as SymmetryTuple,
          type: 'manualSymmetryRestored',
        });
      }

      const restoredActions = Object.values(restoredWorkflow.actionsBySourceRunKey).flat();
      const rotationAction =
        restoredActions
          .filter(
            (action) =>
              action.operationId === trellis2OperationIds.detectRotationSymmetry,
          )
          .at(-1) ?? null;
      const reflectionAction =
        restoredActions
          .filter(
            (action) =>
              action.operationId === trellis2OperationIds.detectReflectionPlanes,
          )
          .at(-1) ?? null;
      const finerAction =
        restoredActions
          .filter(
            (action) => action.operationId === trellis2OperationIds.detectFinerSymmetry,
          )
          .at(-1) ?? null;
      const detectedSymmetryRun = restoredWorkflow.nodeRunsByNode.detect_adjust_symmetry;

      if (rotationAction || reflectionAction || finerAction || detectedSymmetryRun) {
        dispatchDetection({
          confirmedSymmetry: detectedSymmetryRun
            ? (detectedSymmetryRun.jsonResult as SymmetryTuple)
            : null,
          finerAction,
          reflectionAction,
          rotationAction,
          type: 'detectionRestored',
        });
      }

      const restoredExportStates: Record<NodeInstanceId, ExportState> = {};
      for (const exportNodeId of ['vanilla_shape', 'symmetry_shape', 'texture']) {
        const sourceRun = restoredWorkflow.nodeRunsByNode[exportNodeId];

        if (sourceRun) {
          const exportAction =
            (restoredWorkflow.actionsBySourceRunKey[sourceRun.key] ?? [])
              .filter((action) => action.operationId === trellis2OperationIds.exportGlb)
              .at(-1) ?? null;

          if (exportAction) {
            restoredExportStates[exportNodeId] = {
              errorMessage: '',
              log: 'Export ready.',
              params: exportAction.params as Trellis2ExportParams,
              progress: 1,
              requestId: null,
              result: {
                cached: true,
                jsonResult: exportAction.jsonResult,
                key: exportAction.key,
                metadata: exportAction.metadata,
                outputs: exportAction.outputs,
                sessionId: result.value.sessionId,
                sessionRevision: restoredWorkflow.sessionRevision,
              },
              status: 'ready',
            };
          }
        }
      }
      setExportStates(restoredExportStates);

      setUrlSyncReady(true);
    })();
  }, [handleSessionExpired, selectedModel]);

  useEffect(() => {
    if (!urlSyncReady) {
      return;
    }

    const nextUrl = workflowUrl(workflow);
    const currentUrl = `${window.location.pathname}${window.location.search}`;

    if (nextUrl !== currentUrl) {
      window.history.replaceState(null, '', nextUrl);
    }
  }, [urlSyncReady, workflow]);

  const handleEnterModelDag = () => {
    if (sessionMutationInFlightRef.current) {
      return;
    }

    resetPanelStates(
      panelStateDispatchers,
      selectedModel.dag.nodes.map((node) => node.id),
      setExportStates,
    );
    setWorkflow((state) => enterModelDag(state, selectedModel));
  };

  const handleGoBack = () => {
    if (sessionMutationInFlightRef.current) {
      return;
    }

    if (workflow.currentNodeId === selectedModel.dag.entryNodeId) {
      resetPanelStates(
        panelStateDispatchers,
        selectedModel.dag.nodes.map((node) => node.id),
        setExportStates,
      );
      setWorkflow(createInitialWorkflowState());
      return;
    }

    const transition = goBackWorkflowNode(workflow);
    resetPanelStates(panelStateDispatchers, transition.resetNodeIds, setExportStates);
    setWorkflow(transition.state);
  };

  const handleChooseNextNode = (nodeId: NodeInstanceId) => {
    if (sessionMutationInFlightRef.current) {
      return;
    }

    resetPanelStates(panelStateDispatchers, [nodeId], setExportStates);
    setWorkflow((state) => chooseWorkflowEdge(state, selectedModel, nodeId));
  };

  const handleGenerateImageCondition = async () => {
    if (
      !currentNode ||
      !imageConditionState.file ||
      sessionMutationInFlightRef.current
    ) {
      return;
    }

    sessionMutationInFlightRef.current = true;
    setSessionMutationInFlight(true);
    dispatchImageCondition({ type: 'conditionGenerationStarted' });
    const uploadResult = await uploadInputImage(
      imageConditionState.file,
      imageConditionState.previewName,
    );
    if (!uploadResult.ok) {
      dispatchImageCondition({
        message: uploadResult.message,
        type: 'conditionGenerationFailed',
      });
      sessionMutationInFlightRef.current = false;
      setSessionMutationInFlight(false);
      return;
    }

    const params = {};
    dispatchImageCondition({ type: 'inputUploaded', upload: uploadResult.value });
    const runResult = await submitNodeRun({
      inputUploadKeys: [uploadResult.value.uploadKey],
      modelId: selectedModel.id,
      operationId: currentNode.operation,
      parentRunKeys: [],
      params,
      sessionId: workflow.sessionId,
      sessionRevision: workflow.sessionRevision,
    });
    if (!runResult.ok) {
      if (runResult.kind === 'session_expired') {
        handleSessionExpired();
        return;
      }
      dispatchImageCondition({
        message: runResult.message,
        type: 'conditionGenerationFailed',
      });
      sessionMutationInFlightRef.current = false;
      setSessionMutationInFlight(false);
      return;
    }

    dispatchImageCondition({ run: runResult.value, type: 'conditionGenerated' });
    setWorkflow((state) =>
      completeWorkflowNode(state, currentNode.id, currentNode.operation, params, runResult.value),
    );
    sessionMutationInFlightRef.current = false;
    setSessionMutationInFlight(false);
  };

  const handleConfirmManualSymmetry = async () => {
    if (
      !currentNode ||
      !workflow.sessionId ||
      !manualSymmetryState.proposedSymmetry ||
      sessionMutationInFlightRef.current
    ) {
      return;
    }

    const params = { symmetry: manualSymmetryState.proposedSymmetry };
    sessionMutationInFlightRef.current = true;
    setSessionMutationInFlight(true);
    dispatchManualSymmetry({ type: 'confirmationStarted' });
    const result = await submitNodeRun({
      inputUploadKeys: [],
      modelId: selectedModel.id,
      operationId: currentNode.operation,
      parentRunKeys: parentRunKeysForCurrentNode(workflow),
      params,
      sessionId: workflow.sessionId,
      sessionRevision: workflow.sessionRevision,
    });
    if (!result.ok) {
      if (result.kind === 'session_expired') {
        handleSessionExpired();
        return;
      }
      dispatchManualSymmetry({ message: result.message, type: 'confirmationFailed' });
      sessionMutationInFlightRef.current = false;
      setSessionMutationInFlight(false);
      return;
    }

    dispatchManualSymmetry({ type: 'confirmationCompleted' });
    setWorkflow((state) =>
      completeWorkflowNode(state, currentNode.id, currentNode.operation, params, result.value),
    );
    sessionMutationInFlightRef.current = false;
    setSessionMutationInFlight(false);
  };

  const handleConfirmDetectedSymmetry = async () => {
    if (
      !currentNode ||
      !workflow.sessionId ||
      !detectionState.proposedSymmetry ||
      sessionMutationInFlightRef.current
    ) {
      return;
    }

    const params = { symmetry: detectionState.proposedSymmetry };
    sessionMutationInFlightRef.current = true;
    setSessionMutationInFlight(true);
    dispatchDetection({ type: 'confirmationStarted' });
    const result = await submitNodeRun({
      inputUploadKeys: [],
      modelId: selectedModel.id,
      operationId: currentNode.operation,
      parentRunKeys: parentRunKeysForCurrentNode(workflow),
      params,
      sessionId: workflow.sessionId,
      sessionRevision: workflow.sessionRevision,
    });
    if (!result.ok) {
      if (result.kind === 'session_expired') {
        handleSessionExpired();
        return;
      }
      dispatchDetection({ message: result.message, type: 'confirmationFailed' });
      sessionMutationInFlightRef.current = false;
      setSessionMutationInFlight(false);
      return;
    }

    dispatchDetection({ type: 'confirmationCompleted' });
    setWorkflow((state) =>
      completeWorkflowNode(state, currentNode.id, currentNode.operation, params, result.value),
    );
    sessionMutationInFlightRef.current = false;
    setSessionMutationInFlight(false);
  };

  const handleDetectMajorAxis = async () => {
    if (!workflow.sessionId || sessionMutationInFlightRef.current) {
      return;
    }

    const sourceRunKey = latestRunKeyForWorkflow(workflow);
    if (!sourceRunKey) {
      return;
    }

    const params = {};
    sessionMutationInFlightRef.current = true;
    setSessionMutationInFlight(true);
    dispatchDetection({ type: 'majorDetectionStarted' });
    const result = await submitAction<RotationAxisDetectionResult[]>({
      modelId: selectedModel.id,
      operationId: trellis2OperationIds.detectRotationSymmetry,
      params,
      sessionId: workflow.sessionId,
      sessionRevision: workflow.sessionRevision,
      sourceNodeRunKey: sourceRunKey,
    });
    if (!result.ok) {
      if (result.kind === 'session_expired') {
        handleSessionExpired();
        return;
      }
      dispatchDetection({ message: result.message, type: 'majorDetectionFailed' });
      sessionMutationInFlightRef.current = false;
      setSessionMutationInFlight(false);
      return;
    }

    dispatchDetection({
      actionKey: result.value.key,
      candidates: result.value.jsonResult,
      type: 'rotationAxesLoaded',
    });
    setWorkflow((state) =>
      recordWorkflowAction(
        state,
        sourceRunKey,
        trellis2OperationIds.detectRotationSymmetry,
        params,
        result.value,
      ),
    );
    sessionMutationInFlightRef.current = false;
    setSessionMutationInFlight(false);
  };

  const handleDetectFinerSymmetry = async () => {
    if (!workflow.sessionId || sessionMutationInFlightRef.current) {
      return;
    }

    const sourceRunKey = latestRunKeyForWorkflow(workflow);
    if (!sourceRunKey) {
      return;
    }

    const params = {
      center: detectionState.center,
      majorAxis: detectionState.majorAxis,
    };
    sessionMutationInFlightRef.current = true;
    setSessionMutationInFlight(true);
    dispatchDetection({ type: 'finerDetectionStarted' });
    const result = await submitAction<FinerSymmetryDetectionResult>({
      modelId: selectedModel.id,
      operationId: trellis2OperationIds.detectFinerSymmetry,
      params,
      sessionId: workflow.sessionId,
      sessionRevision: workflow.sessionRevision,
      sourceNodeRunKey: sourceRunKey,
    });
    if (!result.ok) {
      if (result.kind === 'session_expired') {
        handleSessionExpired();
        return;
      }
      dispatchDetection({ message: result.message, type: 'finerDetectionFailed' });
      sessionMutationInFlightRef.current = false;
      setSessionMutationInFlight(false);
      return;
    }

    dispatchDetection({
      actionKey: result.value.key,
      result: result.value.jsonResult,
      type: 'finerResultLoaded',
    });
    setWorkflow((state) =>
      recordWorkflowAction(
        state,
        sourceRunKey,
        trellis2OperationIds.detectFinerSymmetry,
        params,
        result.value,
      ),
    );
    sessionMutationInFlightRef.current = false;
    setSessionMutationInFlight(false);
  };

  const handleDetectReflectionPlanes = async () => {
    if (!workflow.sessionId || sessionMutationInFlightRef.current) {
      return;
    }

    const sourceRunKey = latestRunKeyForWorkflow(workflow);
    if (!sourceRunKey) {
      return;
    }

    const params = {};
    sessionMutationInFlightRef.current = true;
    setSessionMutationInFlight(true);
    dispatchDetection({ type: 'reflectionDetectionStarted' });
    const result = await submitAction<ReflectionPlaneDetectionResult[]>({
      modelId: selectedModel.id,
      operationId: trellis2OperationIds.detectReflectionPlanes,
      params,
      sessionId: workflow.sessionId,
      sessionRevision: workflow.sessionRevision,
      sourceNodeRunKey: sourceRunKey,
    });
    if (!result.ok) {
      if (result.kind === 'session_expired') {
        handleSessionExpired();
        return;
      }
      dispatchDetection({ message: result.message, type: 'reflectionDetectionFailed' });
      sessionMutationInFlightRef.current = false;
      setSessionMutationInFlight(false);
      return;
    }

    dispatchDetection({
      actionKey: result.value.key,
      candidates: result.value.jsonResult,
      type: 'reflectionPlanesLoaded',
    });
    setWorkflow((state) =>
      recordWorkflowAction(
        state,
        sourceRunKey,
        trellis2OperationIds.detectReflectionPlanes,
        params,
        result.value,
      ),
    );
    sessionMutationInFlightRef.current = false;
    setSessionMutationInFlight(false);
  };

  const handleGenerateTrellis2VanillaSparseStructure = async () => {
    if (!currentNode || !workflow.sessionId || sessionMutationInFlightRef.current) {
      return;
    }

    const requestId = newRequestId();
    const params = trellis2VanillaSparseStructureState.params;
    sessionMutationInFlightRef.current = true;
    setSessionMutationInFlight(true);
    dispatchTrellis2VanillaSparseStructure({ requestId, type: 'generationStarted' });
    const result = await submitNodeRun({
      inputUploadKeys: [],
      modelId: selectedModel.id,
      operationId: currentNode.operation,
      parentRunKeys: parentRunKeysForCurrentNode(workflow),
      params,
      sessionId: workflow.sessionId,
      sessionRevision: workflow.sessionRevision,
    }, (progress) => {
      dispatchTrellis2VanillaSparseStructure({
        progress: progress.progress,
        requestId,
        stage: progress.stage,
        type: 'generationProgressUpdated',
      });
    });
    if (!result.ok) {
      if (result.kind === 'session_expired') {
        handleSessionExpired();
        return;
      }
      dispatchTrellis2VanillaSparseStructure({
        message: result.message,
        requestId,
        type: 'generationFailed',
      });
      sessionMutationInFlightRef.current = false;
      setSessionMutationInFlight(false);
      return;
    }

    dispatchTrellis2VanillaSparseStructure({
      metadata: trellis2SparseMetadata(result.value.metadata),
      requestId,
      result: result.value,
      type: 'generationCompleted',
    });
    setWorkflow((state) =>
      completeWorkflowNode(state, currentNode.id, currentNode.operation, params, result.value),
    );
    sessionMutationInFlightRef.current = false;
    setSessionMutationInFlight(false);
  };

  const handleGenerateTrellis2SymmetrySparseStructure = async () => {
    if (
      !currentNode ||
      !workflow.sessionId ||
      !confirmedSymmetryTuple ||
      sessionMutationInFlightRef.current
    ) {
      return;
    }

    const requestId = newRequestId();
    const params = trellis2SymmetrySparseStructureState.params;
    sessionMutationInFlightRef.current = true;
    setSessionMutationInFlight(true);
    dispatchTrellis2SymmetrySparseStructure({ requestId, type: 'generationStarted' });
    const result = await submitNodeRun({
      inputUploadKeys: [],
      modelId: selectedModel.id,
      operationId: currentNode.operation,
      parentRunKeys: parentRunKeysForCurrentNode(workflow),
      params,
      sessionId: workflow.sessionId,
      sessionRevision: workflow.sessionRevision,
    }, (progress) => {
      dispatchTrellis2SymmetrySparseStructure({
        progress: progress.progress,
        requestId,
        stage: progress.stage,
        type: 'generationProgressUpdated',
      });
    });
    if (!result.ok) {
      if (result.kind === 'session_expired') {
        handleSessionExpired();
        return;
      }
      dispatchTrellis2SymmetrySparseStructure({
        message: result.message,
        requestId,
        type: 'generationFailed',
      });
      sessionMutationInFlightRef.current = false;
      setSessionMutationInFlight(false);
      return;
    }

    dispatchTrellis2SymmetrySparseStructure({
      metadata: trellis2SparseMetadata(result.value.metadata),
      requestId,
      result: result.value,
      type: 'generationCompleted',
    });
    setWorkflow((state) =>
      completeWorkflowNode(state, currentNode.id, currentNode.operation, params, result.value),
    );
    sessionMutationInFlightRef.current = false;
    setSessionMutationInFlight(false);
  };

  const handleGenerateTrellis2VanillaShape = async () => {
    if (!currentNode || !workflow.sessionId || sessionMutationInFlightRef.current) {
      return;
    }

    const requestId = newRequestId();
    const params = trellis2VanillaShapeState.params;
    sessionMutationInFlightRef.current = true;
    setSessionMutationInFlight(true);
    dispatchTrellis2VanillaShape({ requestId, type: 'generationStarted' });
    const result = await submitNodeRun({
      inputUploadKeys: [],
      modelId: selectedModel.id,
      operationId: currentNode.operation,
      parentRunKeys: parentRunKeysForCurrentNode(workflow),
      params,
      sessionId: workflow.sessionId,
      sessionRevision: workflow.sessionRevision,
    }, (progress) => {
      dispatchTrellis2VanillaShape({
        progress: progress.progress,
        requestId,
        stage: progress.stage,
        type: 'generationProgressUpdated',
      });
    });
    if (!result.ok) {
      if (result.kind === 'session_expired') {
        handleSessionExpired();
        return;
      }
      dispatchTrellis2VanillaShape({
        message: result.message,
        requestId,
        type: 'generationFailed',
      });
      sessionMutationInFlightRef.current = false;
      setSessionMutationInFlight(false);
      return;
    }

    dispatchTrellis2VanillaShape({
      metadata: trellis2ShapeMetadata(result.value.metadata),
      requestId,
      result: result.value,
      type: 'generationCompleted',
    });
    setWorkflow((state) =>
      completeWorkflowNode(state, currentNode.id, currentNode.operation, params, result.value),
    );
    sessionMutationInFlightRef.current = false;
    setSessionMutationInFlight(false);
  };

  const handleGenerateTrellis2SymmetryShape = async () => {
    if (
      !currentNode ||
      !workflow.sessionId ||
      !confirmedSymmetryTuple ||
      sessionMutationInFlightRef.current
    ) {
      return;
    }

    const requestId = newRequestId();
    const params = trellis2SymmetryShapeState.params;
    sessionMutationInFlightRef.current = true;
    setSessionMutationInFlight(true);
    dispatchTrellis2SymmetryShape({ requestId, type: 'generationStarted' });
    const result = await submitNodeRun({
      inputUploadKeys: [],
      modelId: selectedModel.id,
      operationId: currentNode.operation,
      parentRunKeys: parentRunKeysForCurrentNode(workflow),
      params,
      sessionId: workflow.sessionId,
      sessionRevision: workflow.sessionRevision,
    }, (progress) => {
      dispatchTrellis2SymmetryShape({
        progress: progress.progress,
        requestId,
        stage: progress.stage,
        type: 'generationProgressUpdated',
      });
    });
    if (!result.ok) {
      if (result.kind === 'session_expired') {
        handleSessionExpired();
        return;
      }
      dispatchTrellis2SymmetryShape({
        message: result.message,
        requestId,
        type: 'generationFailed',
      });
      sessionMutationInFlightRef.current = false;
      setSessionMutationInFlight(false);
      return;
    }

    dispatchTrellis2SymmetryShape({
      metadata: trellis2ShapeMetadata(result.value.metadata),
      requestId,
      result: result.value,
      type: 'generationCompleted',
    });
    setWorkflow((state) =>
      completeWorkflowNode(state, currentNode.id, currentNode.operation, params, result.value),
    );
    sessionMutationInFlightRef.current = false;
    setSessionMutationInFlight(false);
  };

  const handleGenerateTrellis2Texture = async () => {
    if (!currentNode || !workflow.sessionId || sessionMutationInFlightRef.current) {
      return;
    }

    const requestId = newRequestId();
    const params = trellis2TextureState.params;
    sessionMutationInFlightRef.current = true;
    setSessionMutationInFlight(true);
    dispatchTrellis2Texture({ requestId, type: 'generationStarted' });
    const result = await submitNodeRun({
      inputUploadKeys: [],
      modelId: selectedModel.id,
      operationId: currentNode.operation,
      parentRunKeys: parentRunKeysForCurrentNode(workflow),
      params,
      sessionId: workflow.sessionId,
      sessionRevision: workflow.sessionRevision,
    }, (progress) => {
      dispatchTrellis2Texture({
        progress: progress.progress,
        requestId,
        stage: progress.stage,
        type: 'generationProgressUpdated',
      });
    });
    if (!result.ok) {
      if (result.kind === 'session_expired') {
        handleSessionExpired();
        return;
      }
      dispatchTrellis2Texture({
        message: result.message,
        requestId,
        type: 'generationFailed',
      });
      sessionMutationInFlightRef.current = false;
      setSessionMutationInFlight(false);
      return;
    }

    dispatchTrellis2Texture({
      metadata: trellis2TextureMetadata(result.value.metadata),
      requestId,
      result: result.value,
      type: 'generationCompleted',
    });
    setWorkflow((state) =>
      completeWorkflowNode(state, currentNode.id, currentNode.operation, params, result.value),
    );
    sessionMutationInFlightRef.current = false;
    setSessionMutationInFlight(false);
  };

  const handleExportParamsChange = (
    nodeId: NodeInstanceId,
    params: Partial<Trellis2ExportParams>,
  ) => {
    setExportStates((states) => {
      const state = states[nodeId] ?? ExportControls.initialState;

      return {
        ...states,
        [nodeId]: {
          ...state,
          errorMessage: '',
          log: '',
          params: {
            ...state.params,
            ...params,
          },
          progress: 0,
          requestId: null,
          result: null,
          status: 'idle',
        },
      };
    });
  };

  const handleExportGlb = async (nodeId: NodeInstanceId) => {
    if (!workflow.sessionId || sessionMutationInFlightRef.current) {
      return;
    }

    const sourceRunKey = workflow.nodeRunsByNode[nodeId]?.key;
    if (!sourceRunKey) {
      return;
    }

    const requestId = newRequestId();
    const exportState = exportStates[nodeId] ?? ExportControls.initialState;
    const params = exportState.params;
    sessionMutationInFlightRef.current = true;
    setSessionMutationInFlight(true);
    setExportStates((states) => ({
      ...states,
      [nodeId]: {
        ...exportState,
        errorMessage: '',
        log: 'Exporting GLB.',
        progress: 0,
        requestId,
        result: null,
        status: 'running',
      },
    }));
    const result = await submitAction({
      modelId: selectedModel.id,
      operationId: trellis2OperationIds.exportGlb,
      params,
      sessionId: workflow.sessionId,
      sessionRevision: workflow.sessionRevision,
      sourceNodeRunKey: sourceRunKey,
    }, (progress) => {
      setExportStates((states) => {
        const state = states[nodeId] ?? exportState;

        if (state.requestId !== requestId) {
          return states;
        }

        return {
          ...states,
          [nodeId]: {
            ...state,
            log: progress.stage,
            progress: progress.progress,
          },
        };
      });
    });
    if (!result.ok) {
      if (result.kind === 'session_expired') {
        handleSessionExpired();
        return;
      }
      setExportStates((states) => {
        const state = states[nodeId] ?? exportState;

        if (state.requestId !== requestId) {
          return states;
        }

        return {
          ...states,
          [nodeId]: {
            ...state,
            errorMessage: result.message,
            log: '',
            progress: 0,
            requestId: null,
            result: null,
            status: 'failed',
          },
        };
      });
      sessionMutationInFlightRef.current = false;
      setSessionMutationInFlight(false);
      return;
    }

    setExportStates((states) => {
      const state = states[nodeId] ?? exportState;

      if (state.requestId !== requestId) {
        return states;
      }

      return {
        ...states,
        [nodeId]: {
          ...state,
          errorMessage: '',
          log: 'Export ready.',
          progress: 1,
          requestId: null,
          result: result.value,
          status: 'ready',
        },
      };
    });
    setWorkflow((state) =>
      recordWorkflowAction(
        state,
        sourceRunKey,
        trellis2OperationIds.exportGlb,
        params,
        result.value,
      ),
    );
    sessionMutationInFlightRef.current = false;
    setSessionMutationInFlight(false);
  };

  const handleOverlayPicked = (overlayId: string) => {
    if (
      !sessionMutationInFlightRef.current &&
      currentNode?.kind === 'detect_adjust_symmetry'
    ) {
      dispatchDetection({ overlayId, type: 'overlayPicked' });
    }
  };

  const nodePanel = currentNode ? (
    <div aria-busy={sessionMutationInFlight} inert={sessionMutationInFlight}>
      <NodeRouter
        currentNode={currentNode}
        currentNodeCompleted={currentCompleted}
        detectionState={detectionState}
        exportStates={exportStates}
        imageConditionState={imageConditionState}
        manualSymmetryState={manualSymmetryState}
        onChooseNextNode={handleChooseNextNode}
        onConfirmDetectedSymmetry={handleConfirmDetectedSymmetry}
        onConfirmManualSymmetry={handleConfirmManualSymmetry}
        onDetectFinerSymmetry={handleDetectFinerSymmetry}
        onDetectMajorAxis={handleDetectMajorAxis}
        onDetectReflectionPlanes={handleDetectReflectionPlanes}
        onDetectionAction={dispatchDetection}
        onExportGlb={handleExportGlb}
        onExportParamsChange={handleExportParamsChange}
        onGenerateImageCondition={handleGenerateImageCondition}
        onGenerateTrellis2SymmetryShape={handleGenerateTrellis2SymmetryShape}
        onGenerateTrellis2SymmetrySparseStructure={handleGenerateTrellis2SymmetrySparseStructure}
        onGenerateTrellis2Texture={handleGenerateTrellis2Texture}
        onGenerateTrellis2VanillaShape={handleGenerateTrellis2VanillaShape}
        onGenerateTrellis2VanillaSparseStructure={handleGenerateTrellis2VanillaSparseStructure}
        onGoBack={handleGoBack}
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
    </div>
  ) : (
    <ModelSelectionPanel
      modelOptions={modelOptions}
      onConfirm={handleEnterModelDag}
      selectedModelId={workflow.selectedModelId}
    />
  );

  return (
    <AppLayout
      chosenEdgeIds={chosenEdgeIds}
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
