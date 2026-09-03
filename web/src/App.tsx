/**
 * Main application shell for bpTorch v0.1.
 */

import React, { useEffect, useState } from 'react';
import { ReactFlowProvider } from '@xyflow/react';
import { ApiClient } from './api/client';
import type { NodeDefinitionSummary } from './api/contracts';
import { Canvas } from './canvas/Canvas';
import { BottomDrawer } from './components/BottomDrawer';
import { BreadcrumbBar } from './components/BreadcrumbBar';
import { BentoShell } from './components/layout/BentoShell';
import { TopBar } from './components/TopBar';
import { PropertyInspector } from './inspector/PropertyInspector';
import { NodePalette } from './palette/NodePalette';
import {
  isProject,
  readPersistedProject,
  useProjectStore,
} from './stores/projectStore';
import { useTraceStore } from './stores/traceStore';

export const App: React.FC = () => {
  const { loadProject } = useProjectStore();
  const { addLog } = useTraceStore();
  const [catalog, setCatalog] = useState<NodeDefinitionSummary[]>([]);

  useEffect(() => {
    ApiClient.getNodeCatalog()
      .then((cat) => {
        setCatalog(cat);
        addLog('info', `Loaded ${cat.length} node definitions from backend registry.`);
      })
      .catch((error) => {
        addLog(
          'error',
          `Failed to fetch node registry: ${
            error instanceof Error ? error.message : String(error)
          }`
        );
      });

    const persisted = readPersistedProject();
    if (persisted.status === 'loaded') {
      loadProject(persisted.project);
      addLog('info', `Restored saved project "${persisted.project.project.name}".`);
      return;
    }

    if (persisted.status === 'invalid') {
      addLog('warn', `Saved project could not be restored: ${persisted.error}`);
    }

    const fallbackProject = useProjectStore.getState().project;
    fetch('/examples/arch_1_nanogpt_tiny/arch_1_nanogpt_tiny.nbp.json')
      .then((res) => {
        if (res.ok) return res.json() as Promise<unknown>;
        throw new Error('Template file not reachable');
      })
      .then((data) => {
        if (!isProject(data)) {
          throw new Error('Bundled template has an invalid project shape');
        }

        if (useProjectStore.getState().project !== fallbackProject) {
          return;
        }

        loadProject(data);
        addLog('info', 'Loaded Arch 1: nanoGPT Tiny. Open Templates → Architecture Samples to browse all 26 architectures.');
      })
      .catch((error) => {
        addLog(
          'error',
          `Failed to load bundled nanoGPT template: ${
            error instanceof Error ? error.message : String(error)
          }`
        );
      });
  }, [loadProject, addLog]);

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        width: '100vw',
        background: '#0c0d12',
        overflow: 'hidden',
        fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
      }}
    >
      <div style={{ flexShrink: 0, position: 'relative', zIndex: 100, overflow: 'visible' }}>
        <TopBar />
        <BreadcrumbBar />
      </div>

      <BentoShell
        palette={<NodePalette catalog={catalog} />}
        canvas={
          <ReactFlowProvider>
            <Canvas catalog={catalog} />
          </ReactFlowProvider>
        }
        inspector={<PropertyInspector catalog={catalog} />}
        drawer={<BottomDrawer />}
      />
    </div>
  );
};
