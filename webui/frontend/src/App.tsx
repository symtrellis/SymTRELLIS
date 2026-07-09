import { useEffect, useMemo, useReducer, useRef, useState } from 'react';
import { createJsonWebSocket } from './api/client';
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
  trellis2ShapeMetadata,
  trellis2SparseMetadata,
  trellis2TextureMetadata,
  type Trellis2ExportParams,
  type Trellis2ShapeMetadata,
  type Trellis2SparseMetadata,
  type Trellis2SymmetryShapeParams,
  type Trellis2SymmetrySparseStructureParams,
  type Trellis2TextureMetadata,
  type Trellis2TextureParams,
  type Trellis2VanillaShapeParams,
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
  type WorkflowState,
} from './state/workflow';
import {
  createInitialGenerationState,
  generationReducer,
} from './state/generation';
import { readStoredTheme, writeStoredTheme } from './state/theme';
import {
  createInitialImageConditionState,
  imageConditionReducer,
} from './state/imageCondition';
import {
  detectionReducer,
  initialDetectionState,
} from './state/detection';
import {
  initialManualSymmetryState,
  manualSymmetryReducer,
} from './state/symmetry';
import type {
  FinerSymmetryResult,
  NodeRunKey,
  RequestId,
  RotationAxisCandidate,
  SessionId,
  SymmetryTuple,
  ThemeMode,
} from './types';

const detectRotationOperation = 'symmetry.detect_rotation_symmetry';
const detectFinerOperation = 'symmetry.detect_finer_symmetry';
const exportGlbOperation = 'trellis2.export_glb';

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

function initialExportStates(): Record<NodeInstanceId, ExportState> {
  return {
    symmetry_shape: ExportControls.initialState,
    texture: ExportControls.initialState,
    vanilla_shape: ExportControls.initialState,
  };
}

export default function App() {
  const restoredOnce = useRef(false);
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
  const [exportStates, setExportStates] = useState<Record<NodeInstanceId, ExportState>>(
    initialExportStates,
  );

  const selectedModel = modelSpecs[workflow.selectedModelId];
  const currentNode = selectedModel.dag.nodes.find((node) => node.id === workflow.currentNodeId);
  const successorRoutes = useMemo(
    () => successorRoutesForWorkflow(selectedModel, workflow),
    [selectedModel, workflow],
  );
  const currentCompleted = currentNodeCompleted(workflow);
  const confirmedSymmetryTuple: SymmetryTuple | null =
    manualSymmetryState.proposedSymmetry ?? detectionState.proposedSymmetry;
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

  useEffect(() => {
    writeStoredTheme(theme);
  }, [theme]);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let reconnectTimer = 0;
    let stopped = false;

    const handleMessage = (event: MessageEvent) => {
      const update = JSON.parse(event.data) as {
        progress?: unknown;
        request_id?: unknown;
        type?: unknown;
      };

      if (
        update.type !== 'progress' ||
        typeof update.request_id !== 'string' ||
        typeof update.progress !== 'number'
      ) {
        return;
      }

      const action = {
        progress: update.progress,
        requestId: update.request_id as RequestId,
        type: 'generationProgressUpdated' as const,
      };
      dispatchTrellis2VanillaSparseStructure(action);
      dispatchTrellis2SymmetrySparseStructure(action);
      dispatchTrellis2VanillaShape(action);
      dispatchTrellis2SymmetryShape(action);
      dispatchTrellis2Texture(action);
    };

    const connect = () => {
      const connection = createJsonWebSocket('/ws');
      const nextSocket = connection.socket;
      socket = nextSocket;

      nextSocket.addEventListener('message', handleMessage);
      nextSocket.addEventListener('error', () => {
        nextSocket.close();
      });
      nextSocket.addEventListener('close', () => {
        if (!stopped && reconnectTimer === 0) {
          reconnectTimer = window.setTimeout(() => {
            reconnectTimer = 0;
            connect();
          }, 1000);
        }
      });
    };

    connect();

    return () => {
      stopped = true;
      if (reconnectTimer !== 0) {
        window.clearTimeout(reconnectTimer);
      }
      socket?.close();
    };
  }, []);

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
    const params = new URLSearchParams(window.location.search);
    const sessionId = params.get('session') as SessionId | null;
    const key = params.get('key') as NodeRunKey | null;

    if (!sessionId) {
      return;
    }

    restoreSession(sessionId, key ?? undefined).then((result) => {
      if (result.ok && result.value.modelId === 'trellis2') {
        const model = modelSpecs[result.value.modelId];
        setWorkflow(restoreWorkflowSession(model, result.value));
        // TODO(BACKEND_CONTRACT): hydrate each node panel from restored params/json_result
        // after every operation's durable result contract is fixed.
      }
    });
  }, []);

  const resetAllPanelStates = () => {
    dispatchImageCondition({ type: 'resetToNodeStart' });
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
    setExportStates(initialExportStates());
  };

  const resetPanelStatesForNodes = (nodeIds: NodeInstanceId[]) => {
    if (nodeIds.includes('image_condition')) {
      dispatchImageCondition({ type: 'resetToNodeStart' });
    }

    if (nodeIds.includes('manual_symmetry')) {
      dispatchManualSymmetry({ type: 'reset' });
    }

    if (nodeIds.includes('detect_adjust_symmetry')) {
      dispatchDetection({ type: 'reset' });
    }

    if (nodeIds.includes('vanilla_sparse_structure')) {
      dispatchTrellis2VanillaSparseStructure({
        metadata: trellis2InitialSparseMetadata,
        params: trellis2GenerationDefaults.vanillaSparseStructure,
        type: 'resetToNodeStart',
      });
    }

    if (nodeIds.includes('symmetry_sparse_structure')) {
      dispatchTrellis2SymmetrySparseStructure({
        metadata: trellis2InitialSparseMetadata,
        params: trellis2GenerationDefaults.symmetrySparseStructure,
        type: 'resetToNodeStart',
      });
    }

    if (nodeIds.includes('vanilla_shape')) {
      dispatchTrellis2VanillaShape({
        metadata: trellis2InitialShapeMetadata,
        params: trellis2GenerationDefaults.vanillaShape,
        type: 'resetToNodeStart',
      });
    }

    if (nodeIds.includes('symmetry_shape')) {
      dispatchTrellis2SymmetryShape({
        metadata: trellis2InitialShapeMetadata,
        params: trellis2GenerationDefaults.symmetryShape,
        type: 'resetToNodeStart',
      });
    }

    if (nodeIds.includes('texture')) {
      dispatchTrellis2Texture({
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
  };

  const handleEnterModelDag = () => {
    resetAllPanelStates();
    setWorkflow((state) => enterModelDag(state, selectedModel));
  };

  const handleGoBack = () => {
    if (workflow.currentNodeId === selectedModel.dag.entryNodeId) {
      resetAllPanelStates();
      setWorkflow(createInitialWorkflowState());
      return;
    }

    const transition = goBackWorkflowNode(workflow);
    resetPanelStatesForNodes(transition.resetNodeIds);
    setWorkflow(transition.state);
  };

  const handleChooseNextNode = (nodeId: NodeInstanceId) => {
    resetPanelStatesForNodes([nodeId]);
    setWorkflow((state) => chooseWorkflowEdge(state, selectedModel, nodeId));
  };

  const handleGenerateImageCondition = () => {
    if (!currentNode || !imageConditionState.file) {
      return;
    }

    dispatchImageCondition({ type: 'conditionGenerationStarted' });
    uploadInputImage(imageConditionState.file, imageConditionState.previewName).then((uploadResult) => {
      if (!uploadResult.ok) {
        dispatchImageCondition({
          message: uploadResult.message,
          type: 'conditionGenerationFailed',
        });
        return;
      }

      dispatchImageCondition({ type: 'inputUploaded', upload: uploadResult.value });
      submitNodeRun({
        inputUploadKeys: [uploadResult.value.uploadKey],
        modelId: selectedModel.id,
        operationId: currentNode.operation,
        parentRunKeys: [],
        // TODO(BACKEND_CONTRACT): image condition operation params are finalized with backend implementation.
        params: {},
        requestId: newRequestId(),
        sessionId: workflow.sessionId,
      }).then((runResult) => {
        if (!runResult.ok) {
          dispatchImageCondition({
            message: runResult.message,
            type: 'conditionGenerationFailed',
          });
          return;
        }

        dispatchImageCondition({ run: runResult.value, type: 'conditionGenerated' });
        setWorkflow((state) => completeWorkflowNode(state, currentNode.id, runResult.value));
      });
    });
  };

  const handleConfirmManualSymmetry = () => {
    if (!currentNode || !workflow.sessionId || !manualSymmetryState.proposedSymmetry) {
      return;
    }

    submitNodeRun({
      inputUploadKeys: [],
      modelId: selectedModel.id,
      operationId: currentNode.operation,
      parentRunKeys: parentRunKeysForCurrentNode(workflow),
      // TODO(BACKEND_CONTRACT): confirm symmetry param field is finalized with symmetry operation.
      params: { symmetry: manualSymmetryState.proposedSymmetry },
      requestId: newRequestId(),
      sessionId: workflow.sessionId,
    }).then((result) => {
      if (result.ok) {
        setWorkflow((state) => completeWorkflowNode(state, currentNode.id, result.value));
      }
    });
  };

  const handleConfirmDetectedSymmetry = () => {
    if (!currentNode || !workflow.sessionId || !detectionState.proposedSymmetry) {
      return;
    }

    submitNodeRun({
      inputUploadKeys: [],
      modelId: selectedModel.id,
      operationId: currentNode.operation,
      parentRunKeys: parentRunKeysForCurrentNode(workflow),
      // TODO(BACKEND_CONTRACT): confirm symmetry param field is finalized with symmetry operation.
      params: { symmetry: detectionState.proposedSymmetry },
      requestId: newRequestId(),
      sessionId: workflow.sessionId,
    }).then((result) => {
      if (result.ok) {
        setWorkflow((state) => completeWorkflowNode(state, currentNode.id, result.value));
      }
    });
  };

  const handleDetectMajorAxis = () => {
    if (!workflow.sessionId) {
      return;
    }

    const sourceRunKey = latestRunKeyForWorkflow(workflow);
    if (!sourceRunKey) {
      return;
    }

    dispatchDetection({ type: 'majorDetectionStarted' });
    submitAction<RotationAxisCandidate[]>({
      operationId: detectRotationOperation,
      // TODO(BACKEND_CONTRACT): rotation detection params are finalized with detection operation.
      params: {},
      requestId: newRequestId(),
      sessionId: workflow.sessionId,
      sourceNodeRunKey: sourceRunKey,
    }).then((result) => {
      if (!result.ok) {
        dispatchDetection({ message: result.message, type: 'majorDetectionFailed' });
        return;
      }

      dispatchDetection({
        actionKey: result.value.key,
        candidates: result.value.jsonResult,
        type: 'rotationAxesLoaded',
      });
      setWorkflow((state) =>
        recordWorkflowAction(state, sourceRunKey, detectRotationOperation, result.value),
      );
    });
  };

  const handleDetectFinerSymmetry = () => {
    if (!workflow.sessionId) {
      return;
    }

    const sourceRunKey = latestRunKeyForWorkflow(workflow);
    if (!sourceRunKey) {
      return;
    }

    dispatchDetection({ type: 'finerDetectionStarted' });
    submitAction<FinerSymmetryResult>({
      operationId: detectFinerOperation,
      // TODO(BACKEND_CONTRACT): finer detection params are finalized with detection operation.
      params: {
        center: detectionState.center,
        fold: detectionState.fold,
        majorAxis: detectionState.majorAxis,
      },
      requestId: newRequestId(),
      sessionId: workflow.sessionId,
      sourceNodeRunKey: sourceRunKey,
    }).then((result) => {
      if (!result.ok) {
        dispatchDetection({ message: result.message, type: 'finerDetectionFailed' });
        return;
      }

      dispatchDetection({
        actionKey: result.value.key,
        result: result.value.jsonResult,
        type: 'finerResultLoaded',
      });
      setWorkflow((state) =>
        recordWorkflowAction(state, sourceRunKey, detectFinerOperation, result.value),
      );
    });
  };

  const handleGenerateTrellis2VanillaSparseStructure = () => {
    if (!currentNode || !workflow.sessionId) {
      return;
    }

    const requestId = newRequestId();
    dispatchTrellis2VanillaSparseStructure({ requestId, type: 'generationStarted' });
    submitNodeRun({
      inputUploadKeys: [],
      modelId: selectedModel.id,
      operationId: currentNode.operation,
      parentRunKeys: parentRunKeysForCurrentNode(workflow),
      params: trellis2VanillaSparseStructureState.params,
      requestId,
      sessionId: workflow.sessionId,
    }).then((result) => {
      if (!result.ok) {
        dispatchTrellis2VanillaSparseStructure({
          message: result.message,
          requestId,
          type: 'generationFailed',
        });
        return;
      }

      dispatchTrellis2VanillaSparseStructure({
        metadata: trellis2SparseMetadata(result.value.metadata),
        result: result.value,
        type: 'generationCompleted',
      });
      setWorkflow((state) => completeWorkflowNode(state, currentNode.id, result.value));
    });
  };

  const handleGenerateTrellis2SymmetrySparseStructure = () => {
    if (!currentNode || !workflow.sessionId || !confirmedSymmetryTuple) {
      return;
    }

    const requestId = newRequestId();
    dispatchTrellis2SymmetrySparseStructure({ requestId, type: 'generationStarted' });
    submitNodeRun({
      inputUploadKeys: [],
      modelId: selectedModel.id,
      operationId: currentNode.operation,
      parentRunKeys: parentRunKeysForCurrentNode(workflow),
      params: {
        ...trellis2SymmetrySparseStructureState.params,
        // TODO(BACKEND_CONTRACT): symmetry tuple field name is finalized with TRELLIS.2 operation.
        symmetry: confirmedSymmetryTuple,
      },
      requestId,
      sessionId: workflow.sessionId,
    }).then((result) => {
      if (!result.ok) {
        dispatchTrellis2SymmetrySparseStructure({
          message: result.message,
          requestId,
          type: 'generationFailed',
        });
        return;
      }

      dispatchTrellis2SymmetrySparseStructure({
        metadata: trellis2SparseMetadata(result.value.metadata),
        result: result.value,
        type: 'generationCompleted',
      });
      setWorkflow((state) => completeWorkflowNode(state, currentNode.id, result.value));
    });
  };

  const handleGenerateTrellis2VanillaShape = () => {
    if (!currentNode || !workflow.sessionId) {
      return;
    }

    const requestId = newRequestId();
    dispatchTrellis2VanillaShape({ requestId, type: 'generationStarted' });
    submitNodeRun({
      inputUploadKeys: [],
      modelId: selectedModel.id,
      operationId: currentNode.operation,
      parentRunKeys: parentRunKeysForCurrentNode(workflow),
      params: trellis2VanillaShapeState.params,
      requestId,
      sessionId: workflow.sessionId,
    }).then((result) => {
      if (!result.ok) {
        dispatchTrellis2VanillaShape({
          message: result.message,
          requestId,
          type: 'generationFailed',
        });
        return;
      }

      dispatchTrellis2VanillaShape({
        metadata: trellis2ShapeMetadata(result.value.metadata),
        result: result.value,
        type: 'generationCompleted',
      });
      setWorkflow((state) => completeWorkflowNode(state, currentNode.id, result.value));
    });
  };

  const handleGenerateTrellis2SymmetryShape = () => {
    if (!currentNode || !workflow.sessionId || !confirmedSymmetryTuple) {
      return;
    }

    const requestId = newRequestId();
    dispatchTrellis2SymmetryShape({ requestId, type: 'generationStarted' });
    submitNodeRun({
      inputUploadKeys: [],
      modelId: selectedModel.id,
      operationId: currentNode.operation,
      parentRunKeys: parentRunKeysForCurrentNode(workflow),
      params: {
        ...trellis2SymmetryShapeState.params,
        // TODO(BACKEND_CONTRACT): symmetry tuple field name is finalized with TRELLIS.2 operation.
        symmetry: confirmedSymmetryTuple,
      },
      requestId,
      sessionId: workflow.sessionId,
    }).then((result) => {
      if (!result.ok) {
        dispatchTrellis2SymmetryShape({
          message: result.message,
          requestId,
          type: 'generationFailed',
        });
        return;
      }

      dispatchTrellis2SymmetryShape({
        metadata: trellis2ShapeMetadata(result.value.metadata),
        result: result.value,
        type: 'generationCompleted',
      });
      setWorkflow((state) => completeWorkflowNode(state, currentNode.id, result.value));
    });
  };

  const handleGenerateTrellis2Texture = () => {
    if (!currentNode || !workflow.sessionId) {
      return;
    }

    const requestId = newRequestId();
    dispatchTrellis2Texture({ requestId, type: 'generationStarted' });
    submitNodeRun({
      inputUploadKeys: [],
      modelId: selectedModel.id,
      operationId: currentNode.operation,
      parentRunKeys: parentRunKeysForCurrentNode(workflow),
      params: trellis2TextureState.params,
      requestId,
      sessionId: workflow.sessionId,
    }).then((result) => {
      if (!result.ok) {
        dispatchTrellis2Texture({
          message: result.message,
          requestId,
          type: 'generationFailed',
        });
        return;
      }

      dispatchTrellis2Texture({
        metadata: trellis2TextureMetadata(),
        result: result.value,
        type: 'generationCompleted',
      });
      setWorkflow((state) => completeWorkflowNode(state, currentNode.id, result.value));
    });
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

  const handleExportGlb = (nodeId: NodeInstanceId) => {
    if (!workflow.sessionId) {
      return;
    }

    const sourceRunKey = workflow.nodeRunsByNode[nodeId]?.key;
    if (!sourceRunKey) {
      return;
    }

    const requestId = newRequestId();
    const exportState = exportStates[nodeId] ?? ExportControls.initialState;
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
    submitAction({
      operationId: exportGlbOperation,
      params: exportState.params,
      requestId,
      sessionId: workflow.sessionId,
      sourceNodeRunKey: sourceRunKey,
    }).then((result) => {
      if (!result.ok) {
        setExportStates((states) => ({
          ...states,
          [nodeId]: {
            ...(states[nodeId] ?? exportState),
            errorMessage: result.message,
            log: '',
            progress: 0,
            requestId: null,
            result: null,
            status: 'failed',
          },
        }));
        return;
      }

      setExportStates((states) => ({
        ...states,
        [nodeId]: {
          ...(states[nodeId] ?? exportState),
          errorMessage: '',
          log: 'Export ready.',
          progress: 1,
          requestId: null,
          result: result.value,
          status: 'ready',
        },
      }));
      setWorkflow((state) => recordWorkflowAction(state, sourceRunKey, exportGlbOperation, result.value));
    });
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
      onChooseNextNode={handleChooseNextNode}
      onConfirmDetectedSymmetry={handleConfirmDetectedSymmetry}
      onConfirmManualSymmetry={handleConfirmManualSymmetry}
      onDetectFinerSymmetry={handleDetectFinerSymmetry}
      onDetectMajorAxis={handleDetectMajorAxis}
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
