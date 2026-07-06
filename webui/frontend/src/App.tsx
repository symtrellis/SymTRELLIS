import { useEffect, useReducer, useState } from 'react';
import { detectFinerSymmetry, detectRotationAxes } from './api';
import { AppLayout } from './layout/AppLayout';
import { dagEdges, dagNodes } from './dag';
import {
  detectionReducer,
  initialCurrentNodeId,
  initialDagStatus,
  initialDetectionState,
  readStoredTheme,
  writeStoredTheme,
} from './state';
import type { ThemeMode } from './types';

export default function App() {
  const [theme, setTheme] = useState<ThemeMode>(readStoredTheme);
  const [detectionState, dispatchDetection] = useReducer(detectionReducer, initialDetectionState);

  useEffect(() => {
    writeStoredTheme(theme);
  }, [theme]);

  const handleDetectMajorAxis = async () => {
    dispatchDetection({ type: 'majorDetectionStarted' });
    dispatchDetection({ candidates: await detectRotationAxes(), type: 'rotationAxesLoaded' });
  };

  const handleDetectFinerSymmetry = async () => {
    dispatchDetection({ type: 'finerDetectionStarted' });
    dispatchDetection({ result: await detectFinerSymmetry(), type: 'finerResultLoaded' });
  };

  return (
    <AppLayout
      currentNodeId={initialCurrentNodeId}
      dagEdges={dagEdges}
      dagNodes={dagNodes}
      dagStatus={initialDagStatus}
      detectionState={detectionState}
      onDetectFinerSymmetry={handleDetectFinerSymmetry}
      onDetectMajorAxis={handleDetectMajorAxis}
      onDetectionAction={dispatchDetection}
      onThemeChange={setTheme}
      theme={theme}
    />
  );
}
