/**
 * Hierarchical Breadcrumb navigation bar with Event Graph and Function tabs and repeat-instance switcher.
 */

import React from 'react';
import {
  Activity,
  ArrowLeft,
  ArrowRight,
  ChevronRight,
  Flame,
  Layers,
} from 'lucide-react';
import { useProjectStore } from '../stores/projectStore';
import { useUIStore } from '../stores/uiStore';

export const BreadcrumbBar: React.FC = () => {
  const project = useProjectStore((s) => s.project);
  const openGraphId = useProjectStore((s) => s.openGraphId);
  const openGraph = useProjectStore((s) => s.openGraph);
  const navigateBack = useProjectStore((s) => s.navigateBack);
  const navigateForward = useProjectStore((s) => s.navigateForward);
  const historyIndex = useProjectStore((s) => s.historyIndex);
  const graphHistory = useProjectStore((s) => s.graphHistory);

  const { repeatInstanceIndex, setRepeatInstanceIndex } = useUIStore();

  const rootGraphId = project?.model?.root_graph_id || 'graph_gpt';
  const currentGraph = project?.model?.graphs?.[openGraphId];

  // Build breadcrumb segments by analyzing graph relationships
  const rootName = project?.model?.graphs?.[rootGraphId]?.name || 'nanoGPT';
  const breadcrumbSegments: Array<{ id: string; name: string }> = [
    { id: rootGraphId, name: rootName },
  ];

  if (openGraphId !== rootGraphId && currentGraph) {
    if (openGraphId.includes('attention') || openGraphId.includes('mlp')) {
      if (project?.model?.graphs?.['graph_stack']) {
        breadcrumbSegments.push({ id: 'graph_stack', name: 'Transformer Stack' });
      }
      if (project?.model?.graphs?.['graph_block']) {
        breadcrumbSegments.push({ id: 'graph_block', name: 'Transformer Block' });
      }
    } else if (openGraphId === 'graph_block') {
      if (project?.model?.graphs?.['graph_stack']) {
        breadcrumbSegments.push({ id: 'graph_stack', name: 'Transformer Stack' });
      }
    }
    breadcrumbSegments.push({ id: openGraphId, name: currentGraph.name });
  }

  const n_layer = Number(project?.model?.config?.['n_layer'] || 2);
  const isInsideOrNearRepeat =
    openGraphId.includes('block') ||
    openGraphId.includes('attention') ||
    openGraphId.includes('mlp') ||
    openGraphId.includes('stack');

  // Available graph kinds
  const isEventGraph = currentGraph?.kind === 'training_event' || openGraphId.includes('event');

  return (
    <div
      style={{
        height: 36,
        background: '#141620',
        borderBottom: '1px solid #1f2430',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 16px',
        color: '#94a3b8',
        fontSize: 11,
        userSelect: 'none',
      }}
    >
      {/* Left: Navigation Buttons & Breadcrumbs */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        {/* Back / Forward History */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 2, marginRight: 6 }}>
          <button
            title="Back"
            disabled={historyIndex <= 0}
            onClick={navigateBack}
            style={{
              background: 'transparent',
              border: 'none',
              color: historyIndex > 0 ? '#e2e8f0' : '#475569',
              cursor: historyIndex > 0 ? 'pointer' : 'default',
              padding: 3,
              display: 'flex',
              alignItems: 'center',
            }}
          >
            <ArrowLeft size={13} />
          </button>
          <button
            title="Forward"
            disabled={historyIndex >= graphHistory.length - 1}
            onClick={navigateForward}
            style={{
              background: 'transparent',
              border: 'none',
              color: historyIndex < graphHistory.length - 1 ? '#e2e8f0' : '#475569',
              cursor: historyIndex < graphHistory.length - 1 ? 'pointer' : 'default',
              padding: 3,
              display: 'flex',
              alignItems: 'center',
            }}
          >
            <ArrowRight size={13} />
          </button>
        </div>

        {/* Graph Kind Chips */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginRight: 8 }}>
          <button
            onClick={() => openGraph(rootGraphId)}
            style={{
              background: !isEventGraph ? '#1e3a8a' : '#181b24',
              color: !isEventGraph ? '#93c5fd' : '#64748b',
              border: '1px solid',
              borderColor: !isEventGraph ? '#3b82f6' : '#272c3b',
              borderRadius: 4,
              padding: '2px 8px',
              fontSize: 10,
              fontWeight: 600,
              cursor: 'not-allowed',
              opacity: 0.6,
              display: 'flex',
              alignItems: 'center',
              gap: 4,
            }}
          >
            <Activity size={11} />
            Architecture
          </button>
          <button
            title="Visual event graph preview; structured loop and branch execution is not yet enabled"
            disabled={!project?.model?.graphs?.['graph_training_event']}
            onClick={() => {
              if (project?.model?.graphs?.['graph_training_event']) {
                openGraph('graph_training_event');
              }
            }}
            style={{
              background: isEventGraph ? '#701a75' : '#181b24',
              color: isEventGraph ? '#f5d0fe' : '#64748b',
              border: '1px solid',
              borderColor: isEventGraph ? '#d946ef' : '#272c3b',
              borderRadius: 4,
              padding: '2px 8px',
              fontSize: 10,
              fontWeight: 600,
              cursor: project?.model?.graphs?.['graph_training_event'] ? 'pointer' : 'not-allowed',
              display: 'flex',
              alignItems: 'center',
              gap: 4,
            }}
          >
            <Flame size={11} />
            Event Graph (Preview)
          </button>
        </div>

        {/* Breadcrumb Segments */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          {breadcrumbSegments.map((seg, idx) => {
            const isLast = idx === breadcrumbSegments.length - 1;

            return (
              <React.Fragment key={seg.id}>
                {idx > 0 && <ChevronRight size={12} color="#475569" />}
                <button
                  onClick={() => openGraph(seg.id)}
                  style={{
                    background: 'transparent',
                    border: 'none',
                    color: isLast ? '#38bdf8' : '#94a3b8',
                    fontWeight: isLast ? 600 : 400,
                    cursor: isLast ? 'default' : 'pointer',
                    fontSize: 11,
                    padding: '2px 4px',
                    borderRadius: 3,
                  }}
                >
                  {seg.name}
                </button>
              </React.Fragment>
            );
          })}
        </div>
      </div>

      {/* Right: Repeat Instance Switcher */}
      {isInsideOrNearRepeat && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Layers size={13} color="#94a3b8" />
          <span style={{ fontSize: 11 }}>Block Instance:</span>
          <select
            disabled
            title="Block instance switching is not yet implemented"
            value={repeatInstanceIndex}
            onChange={(e) => setRepeatInstanceIndex(parseInt(e.target.value, 10))}
            style={{
              background: '#1e2330',
              border: '1px solid #2d3446',
              color: '#e2e8f0',
              padding: '2px 8px',
              borderRadius: 4,
              fontSize: 11,
              outline: 'none',
              cursor: 'not-allowed',
              opacity: 0.6,
            }}
          >
            {Array.from({ length: n_layer }).map((_, i) => (
              <option key={i} value={i}>
                Instance {i + 1} of {n_layer} (blocks[{i}])
              </option>
            ))}
          </select>
        </div>
      )}
    </div>
  );
};
