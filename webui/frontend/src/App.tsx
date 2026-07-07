import { useEffect, useReducer, useRef, useState } from 'react';
import { detectFinerSymmetry, detectRotationAxes } from './api';
import { AppLayout } from './layout/AppLayout';
import { dagEdges, dagNodes } from './dag';
import {
  cascadeShapeLatentGridSize,
  dagStatusForCurrentNode,
  detectionReducer,
  initialDetectionState,
  initialImageConditionState,
  initialManualSymmetryState,
  initialSymShapeState,
  initialSymSparseStructureState,
  initialTextureState,
  initialVanillaShapeState,
  initialVanillaSparseStructureState,
  imageConditionReducer,
  manualSymmetryReducer,
  readStoredTheme,
  symShapeReducer,
  symSparseStructureReducer,
  textureReducer,
  vanillaShapeReducer,
  vanillaSparseStructureReducer,
  writeStoredTheme,
} from './state';
import type { AppStage, ModelId, NodeId, SymmetryTuple, ThemeMode } from './types';

export default function App() {
  const [theme, setTheme] = useState<ThemeMode>(readStoredTheme);
  const [appStage, setAppStage] = useState<AppStage>('dag');
  const [currentNodeId, setCurrentNodeId] = useState<NodeId>('nat_shape');
  const [imageConditionState, dispatchImageCondition] = useReducer(
    imageConditionReducer,
    initialImageConditionState,
  );
  const [detectionState, dispatchDetection] = useReducer(detectionReducer, initialDetectionState);
  const [manualState, dispatchManual] = useReducer(manualSymmetryReducer, initialManualSymmetryState);
  const [vanillaSparseState, dispatchVanillaSparse] = useReducer(
    vanillaSparseStructureReducer,
    initialVanillaSparseStructureState,
  );
  const [symSparseState, dispatchSymSparse] = useReducer(
    symSparseStructureReducer,
    initialSymSparseStructureState,
  );
  const [vanillaShapeState, dispatchVanillaShape] = useReducer(
    vanillaShapeReducer,
    initialVanillaShapeState,
  );
  const [symShapeState, dispatchSymShape] = useReducer(symShapeReducer, initialSymShapeState);
  const [textureState, dispatchTexture] = useReducer(textureReducer, initialTextureState);
  const vanillaSparseMockTimerRef = useRef<number | null>(null);
  const vanillaShapeMockTimerRef = useRef<number | null>(null);
  const symSparseMockTimerRef = useRef<number | null>(null);
  const symShapeMockTimerRef = useRef<number | null>(null);
  const textureMockTimerRef = useRef<number | null>(null);
  const selectedModelId: ModelId = 'trellis2';
  const confirmedSymmetryTuple: SymmetryTuple = manualState.proposedSymmetry ??
    detectionState.proposedSymmetry ?? {
      center: [0, 0, 0],
      label: 'C2',
      majorAxis: [0, 0, 1],
      minorAxis: [1, 0, 0],
    };

  useEffect(() => {
    writeStoredTheme(theme);
  }, [theme]);

  useEffect(() => {
    const imageUrl = imageConditionState.uploadedImageUrl;

    return () => {
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }
    };
  }, [imageConditionState.uploadedImageUrl]);

  useEffect(() => {
    return () => {
      if (vanillaSparseMockTimerRef.current !== null) {
        window.clearInterval(vanillaSparseMockTimerRef.current);
      }

      if (symSparseMockTimerRef.current !== null) {
        window.clearInterval(symSparseMockTimerRef.current);
      }

      if (symShapeMockTimerRef.current !== null) {
        window.clearInterval(symShapeMockTimerRef.current);
      }

      if (vanillaShapeMockTimerRef.current !== null) {
        window.clearInterval(vanillaShapeMockTimerRef.current);
      }

      if (textureMockTimerRef.current !== null) {
        window.clearInterval(textureMockTimerRef.current);
      }
    };
  }, []);

  const handleDetectMajorAxis = async () => {
    dispatchDetection({ type: 'majorDetectionStarted' });
    dispatchDetection({ candidates: await detectRotationAxes(), type: 'rotationAxesLoaded' });
  };

  const handleDetectFinerSymmetry = async () => {
    dispatchDetection({ type: 'finerDetectionStarted' });
    dispatchDetection({ result: await detectFinerSymmetry(), type: 'finerResultLoaded' });
  };

  const handleEnterImageCondition = () => {
    setAppStage('dag');
    setCurrentNodeId('img_cond');
  };

  const handleEnterManualSymmetry = () => {
    setCurrentNodeId('manual_sym');
  };

  const handleImageSelected = (file: Blob | File, name: string) => {
    dispatchImageCondition({
      file,
      name,
      type: 'imageUploaded',
      url: URL.createObjectURL(file),
    });
  };

  const handleGenerateCondition = () => {
    dispatchImageCondition({ type: 'conditionGenerated' });
  };

  const handleGenerateVanillaSparseStructure = () => {
    if (vanillaSparseMockTimerRef.current !== null) {
      window.clearInterval(vanillaSparseMockTimerRef.current);
    }

    dispatchVanillaSparse({ type: 'generationStarted' });

    // MOCK_VANILLA_SS_GENERATION_START
    // Replace this interval with the backend vanilla nat_ss job/WebSocket progress stream.
    const totalSteps = vanillaSparseState.steps;
    let completedSteps = 0;
    vanillaSparseMockTimerRef.current = window.setInterval(() => {
      completedSteps += 1;
      dispatchVanillaSparse({
        progress: Math.min(completedSteps / totalSteps, 1),
        type: 'generationProgressed',
      });

      if (completedSteps >= totalSteps) {
        if (vanillaSparseMockTimerRef.current !== null) {
          window.clearInterval(vanillaSparseMockTimerRef.current);
          vanillaSparseMockTimerRef.current = null;
        }

        dispatchVanillaSparse({
          generatedOccUrl: '/mock/occ.glb',
          type: 'generationFinished',
          voxelCount: 5521,
        });
      }
    }, 200);
    // MOCK_VANILLA_SS_GENERATION_END
  };

  const handleGenerateSymSparseStructure = () => {
    if (symSparseMockTimerRef.current !== null) {
      window.clearInterval(symSparseMockTimerRef.current);
    }

    dispatchSymSparse({ type: 'generationStarted' });

    // MOCK_SYM_SS_GENERATION_START
    // Replace this interval with the backend sym_ss job/WebSocket progress stream.
    const totalSteps = symSparseState.steps;
    let completedSteps = 0;
    symSparseMockTimerRef.current = window.setInterval(() => {
      completedSteps += 1;
      dispatchSymSparse({
        progress: Math.min(completedSteps / totalSteps, 1),
        type: 'generationProgressed',
      });

      if (completedSteps >= totalSteps) {
        if (symSparseMockTimerRef.current !== null) {
          window.clearInterval(symSparseMockTimerRef.current);
          symSparseMockTimerRef.current = null;
        }

        dispatchSymSparse({
          generatedOccUrl: '/mock/occ.glb',
          type: 'generationFinished',
          voxelCount: 5521,
        });
      }
    }, 200);
    // MOCK_SYM_SS_GENERATION_END
  };

  const handleGenerateVanillaShape = () => {
    if (vanillaShapeMockTimerRef.current !== null) {
      window.clearInterval(vanillaShapeMockTimerRef.current);
    }

    const mode = vanillaShapeState.mode;
    const maxTokens = vanillaShapeState.maxTokens;
    const totalSteps = vanillaShapeState.steps;
    dispatchVanillaShape({ inputOccUrl: '/mock/occ.glb', type: 'generationStarted' });

    // MOCK_VANILLA_SHAPE_GENERATION_START
    // Replace this interval with the backend vanilla nat_shape job/WebSocket progress stream.
    let completedSteps = 0;
    vanillaShapeMockTimerRef.current = window.setInterval(() => {
      completedSteps += 1;
      dispatchVanillaShape({
        progress: Math.min(completedSteps / totalSteps, 1),
        type: 'generationProgressed',
      });

      if (completedSteps >= totalSteps) {
        if (vanillaShapeMockTimerRef.current !== null) {
          window.clearInterval(vanillaShapeMockTimerRef.current);
          vanillaShapeMockTimerRef.current = null;
        }

        dispatchVanillaShape({
          generatedShapeUrl: '/mock/shape.glb',
          oVoxelGridSize: mode === '512' ? 512 : 1536,
          shapeLatentGridSize: mode === '512' ? 32 : cascadeShapeLatentGridSize(maxTokens),
          type: 'generationFinished',
          voxelCount: 489880,
        });
      }
    }, 200);
    // MOCK_VANILLA_SHAPE_GENERATION_END
  };

  const handleGenerateSymShape = () => {
    if (symShapeMockTimerRef.current !== null) {
      window.clearInterval(symShapeMockTimerRef.current);
    }

    const mode = symShapeState.mode;
    const maxTokens = symShapeState.maxTokens;
    const totalSteps = symShapeState.steps;
    dispatchSymShape({ inputOccUrl: '/mock/occ.glb', type: 'generationStarted' });

    // MOCK_SYM_SHAPE_GENERATION_START
    // Replace this interval with the backend sym_shape job/WebSocket progress stream.
    let completedSteps = 0;
    symShapeMockTimerRef.current = window.setInterval(() => {
      completedSteps += 1;
      dispatchSymShape({
        progress: Math.min(completedSteps / totalSteps, 1),
        type: 'generationProgressed',
      });

      if (completedSteps >= totalSteps) {
        if (symShapeMockTimerRef.current !== null) {
          window.clearInterval(symShapeMockTimerRef.current);
          symShapeMockTimerRef.current = null;
        }

        dispatchSymShape({
          generatedShapeUrl: '/mock/shape.glb',
          oVoxelGridSize: mode === '512' ? 512 : 1536,
          shapeLatentGridSize: mode === '512' ? 32 : cascadeShapeLatentGridSize(maxTokens),
          type: 'generationFinished',
          voxelCount: 489880,
        });
      }
    }, 200);
    // MOCK_SYM_SHAPE_GENERATION_END
  };

  const handleGenerateTexture = () => {
    if (textureMockTimerRef.current !== null) {
      window.clearInterval(textureMockTimerRef.current);
    }

    const totalSteps = textureState.steps;
    dispatchTexture({ inputShapeUrl: '/mock/shape.glb', type: 'generationStarted' });

    // MOCK_TEXTURE_GENERATION_START
    // Replace this interval with the backend texture job/WebSocket progress stream.
    let completedSteps = 0;
    textureMockTimerRef.current = window.setInterval(() => {
      completedSteps += 1;
      dispatchTexture({
        progress: Math.min(completedSteps / totalSteps, 1),
        type: 'generationProgressed',
      });

      if (completedSteps >= totalSteps) {
        if (textureMockTimerRef.current !== null) {
          window.clearInterval(textureMockTimerRef.current);
          textureMockTimerRef.current = null;
        }

        dispatchTexture({
          generatedTextureUrl: '/mock/full.glb',
          type: 'generationFinished',
        });
      }
    }, 200);
    // MOCK_TEXTURE_GENERATION_END
  };

  return (
    <AppLayout
      currentNodeId={appStage === 'dag' ? currentNodeId : null}
      dagEdges={dagEdges}
      dagNodes={dagNodes}
      dagStatus={dagStatusForCurrentNode(currentNodeId)}
      detectionState={detectionState}
      imageConditionState={imageConditionState}
      manualState={manualState}
      onDetectFinerSymmetry={handleDetectFinerSymmetry}
      onDetectMajorAxis={handleDetectMajorAxis}
      onDetectionAction={dispatchDetection}
      onEnterImageCondition={handleEnterImageCondition}
      onEnterManualSymmetry={handleEnterManualSymmetry}
      onGenerateCondition={handleGenerateCondition}
      onGenerateSymShape={handleGenerateSymShape}
      onGenerateSymSparseStructure={handleGenerateSymSparseStructure}
      onGenerateTexture={handleGenerateTexture}
      onGenerateVanillaShape={handleGenerateVanillaShape}
      onGenerateVanillaSparseStructure={handleGenerateVanillaSparseStructure}
      onImageSelected={handleImageSelected}
      onManualAction={dispatchManual}
      onSymShapeAction={dispatchSymShape}
      onSymSparseAction={dispatchSymSparse}
      onTextureAction={dispatchTexture}
      onVanillaShapeAction={dispatchVanillaShape}
      onVanillaSparseAction={dispatchVanillaSparse}
      onThemeChange={setTheme}
      selectedModelId={selectedModelId}
      symShapeState={symShapeState}
      symShapeTuple={confirmedSymmetryTuple}
      symSparseState={symSparseState}
      symSparseTuple={confirmedSymmetryTuple}
      textureState={textureState}
      vanillaShapeState={vanillaShapeState}
      vanillaSparseState={vanillaSparseState}
      theme={theme}
    />
  );
}
