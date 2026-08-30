/**
 * Blueprint-style visual custom node component for React Flow.
 * Renders Exec (chevron/triangle) and Data (typed colored circles) ports,
 * along with latency and gradient telemetry badges.
 */

import React, { memo } from 'react';
import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  Box,
  CircleDot,
  Columns,
  Combine,
  Crosshair,
  Divide,
  Flame,
  GitCommit,
  GitFork,
  Key,
  Layers,
  ListOrdered,
  LogIn,
  LogOut,
  Maximize2,
  Minimize2,
  Percent,
  PieChart,
  Plus,
  RefreshCw,
  Scissors,
  Settings,
  Shield,
  Sliders,
  Target,
  Terminal,
  X,
  Zap,
} from 'lucide-react';
import { PortDefinition, TensorSpec } from '../api/contracts';
import { useProjectStore } from '../stores/projectStore';
import { useTraceStore } from '../stores/traceStore';
import { useUIStore } from '../stores/uiStore';
import { useValidationStore } from '../stores/validationStore';
import { BlueprintPortHandle } from './BlueprintPortHandle';

const ICON_MAP: Record<string, React.FC<{ size?: number; className?: string }>> = {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  Box,
  Columns,
  Combine,
  Crosshair,
  Divide,
  Flame,
  GitCommit,
  GitFork,
  Key,
  Layers,
  ListOrdered,
  LogIn,
  LogOut,
  Maximize2,
  Minimize2,
  Percent,
  PieChart,
  Plus,
  RefreshCw,
  Scissors,
  Settings,
  Shield,
  Sliders,
  Target,
  Terminal,
  X,
  Zap,
};

export interface BlueprintNodeData {
  id: string;
  definitionId: string;
  displayName: string;
  category: string;
  icon?: string | null;
  isComposite?: boolean;
  targetGraphId?: string | null;
  properties: Record<string, unknown>;
  inputs: PortDefinition[];
  outputs: PortDefinition[];
  breakpoint?: boolean;
}

const CATEGORY_COLORS: Record<string, { bg: string; border: string; badge: string }> = {
  'Flow Control': { bg: '#14532d', border: '#22c55e', badge: '#16a34a' },
  Events: { bg: '#701a75', border: '#d946ef', badge: '#c026d3' },
  Variables: { bg: '#831843', border: '#f43f5e', badge: '#e11d48' },
  'Data Pipelines': { bg: '#1e3a8a', border: '#3b82f6', badge: '#1d4ed8' },
  Optimization: { bg: '#7c2d12', border: '#f97316', badge: '#ea580c' },
  'LR Schedulers': { bg: '#134e4a', border: '#14b8a6', badge: '#0d9488' },
  'Metrics & Evaluation': { bg: '#3b0764', border: '#a855f7', badge: '#9333ea' },
  Persistence: { bg: '#374151', border: '#9ca3af', badge: '#6b7280' },
  Inputs: { bg: '#064e3b', border: '#10b981', badge: '#047857' },
  Layers: { bg: '#1e3a8a', border: '#3b82f6', badge: '#1d4ed8' },
  'Tensor Operations': { bg: '#7c2d12', border: '#f97316', badge: '#c2410c' },
  Attention: { bg: '#581c87', border: '#a855f7', badge: '#7e22ce' },
  'Composite Modules': { bg: '#312e81', border: '#6366f1', badge: '#4338ca' },
  'Loss & Outputs': { bg: '#881337', border: '#f43f5e', badge: '#be123c' },
};

function getPortHandleStyle(port: PortDefinition, isOutput: boolean): React.CSSProperties {
  const isExec =
    port.kind === 'exec' ||
    port.id.includes('exec') ||
    port.id.includes('then_') ||
    port.id === 'loop_body';

  if (isExec) {
    return {
      background: '#ffffff',
      width: 10,
      height: 10,
      borderRadius: 2,
      transform: 'rotate(45deg)',
      left: isOutput ? undefined : -15,
      right: isOutput ? -15 : undefined,
      border: '2px solid #0f172a',
      zIndex: 10,
    };
  }

  const dtypeFamily = port.tensor_type?.dtype_family || 'floating';
  let portColor = '#38bdf8';
  if (dtypeFamily === 'integer') {
    portColor = '#22c55e';
  } else if (dtypeFamily === 'boolean') {
    portColor = '#eab308';
  } else if (dtypeFamily === 'any' || !port.tensor_type) {
    portColor = '#f97316';
  }

  return {
    background: portColor,
    width: 10,
    height: 10,
    borderRadius: '50%',
    left: isOutput ? undefined : -15,
    right: isOutput ? -15 : undefined,
    border: '2px solid #0f172a',
    zIndex: 10,
  };
}

export const BlueprintNode = memo(({ data, selected }: { data: BlueprintNodeData; selected?: boolean }) => {
  const { openGraph, setNodeBreakpoint } = useProjectStore();
  const { selectedNodeId, selectNode } = useUIStore();
  const { resolvedShapes, diagnostics } = useValidationStore();
  const { nodeStates, nodeGradientNorms } = useTraceStore();

  const isSelected = selected || selectedNodeId === data.id;
  const executionState = nodeStates[data.id] || 'pending';

  const categoryColor = CATEGORY_COLORS[data.category] || {
    bg: '#1e293b',
    border: '#64748b',
    badge: '#334155',
  };

  const IconComponent = data.icon ? ICON_MAP[data.icon] || Box : Box;
  const nodeDiagnostics = diagnostics.filter((d) => d.node_id === data.id);
  const hasError = nodeDiagnostics.some((d) => d.severity === 'error');

  let borderColor = categoryColor.border;
  if (hasError || executionState === 'failed') {
    borderColor = '#ef4444';
  } else if (executionState === 'running') {
    borderColor = '#60a5fa';
  } else if (executionState === 'paused') {
    borderColor = '#f59e0b';
  } else if (executionState === 'completed') {
    borderColor = '#22c55e';
  } else if (isSelected) {
    borderColor = '#38bdf8';
  }

  const getShapeString = (spec?: TensorSpec): string => {
    if (!spec || !spec.shape) return '';
    const dims = spec.shape.map((d) => {
      if (typeof d === 'object') {
        if ('name' in d) return d.name;
        if ('key' in d) return d.key;
        if ('value' in d) return String(d.value);
      }
      return '?';
    });
    return `[${dims.join(', ')}]`;
  };

  const gNorm = nodeGradientNorms[data.id] ?? nodeGradientNorms[`${data.id}.weight`];
  let gradStatusColor: string | null = null;
  if (gNorm !== undefined) {
    if (gNorm > 10.0) gradStatusColor = '#ef4444';
    else if (gNorm < 1e-4) gradStatusColor = '#3b82f6';
    else gradStatusColor = '#22c55e';
  }

  return (
    <div
      onClick={(e) => {
        e.stopPropagation();
        selectNode(data.id);
      }}
      style={{
        minWidth: 200,
        background: '#13151b',
        borderRadius: 8,
        border: `2px solid ${borderColor}`,
        boxShadow: isSelected
          ? '0 0 16px rgba(56, 189, 248, 0.4)'
          : executionState === 'running'
            ? '0 0 16px rgba(96, 165, 250, 0.5)'
            : '0 4px 12px rgba(0, 0, 0, 0.5)',
        color: '#f1f5f9',
        fontSize: 12,
        overflow: 'hidden',
        transition: 'all 0.15s ease',
      }}
    >
      <div
        style={{
          background: categoryColor.bg,
          padding: '6px 10px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <IconComponent size={14} className="text-white opacity-90" />
          <span style={{ fontWeight: 600, fontSize: 12, letterSpacing: '0.02em' }}>
            {data.displayName}
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {gradStatusColor && (
            <span
              title={`Grad Norm: ${gNorm?.toFixed(4)}`}
              style={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                background: gradStatusColor,
                display: 'inline-block',
                boxShadow: `0 0 6px ${gradStatusColor}`,
              }}
            />
          )}

          <button
            title={data.breakpoint ? 'Disable Breakpoint' : 'Set Breakpoint'}
            onClick={(e) => {
              e.stopPropagation();
              setNodeBreakpoint(data.id, !data.breakpoint);
            }}
            style={{
              background: 'transparent',
              border: 'none',
              cursor: 'pointer',
              padding: 2,
              display: 'flex',
              alignItems: 'center',
            }}
          >
            <CircleDot
              size={14}
              style={{
                color: data.breakpoint ? '#ef4444' : '#64748b',
                fill: data.breakpoint ? '#ef4444' : 'transparent',
              }}
            />
          </button>
        </div>
      </div>

      <div style={{ padding: '8px 10px', display: 'flex', flexDirection: 'column', gap: 6 }}>
        {data.inputs.map((port) => (
          <div
            key={port.id}
            style={{
              display: 'flex',
              alignItems: 'center',
              position: 'relative',
              paddingLeft: 4,
              minHeight: 18,
            }}
          >
            <BlueprintPortHandle
              nodeId={data.id}
              port={port}
              isOutput={false}
              baseStyle={getPortHandleStyle(port, false)}
            />
            <span
              style={{
                color: port.kind === 'exec' ? '#ffffff' : '#94a3b8',
                fontSize: 11,
                fontWeight: port.kind === 'exec' ? 600 : 400,
              }}
            >
              {port.display_name}
            </span>
          </div>
        ))}

        {data.outputs.map((port) => {
          const spec = resolvedShapes[data.id]?.[port.id];
          const shapeStr = getShapeString(spec);

          return (
            <div
              key={port.id}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                position: 'relative',
                paddingRight: 4,
                minHeight: 18,
              }}
            >
              <span
                style={{
                  color: port.kind === 'exec' ? '#ffffff' : '#e2e8f0',
                  fontSize: 11,
                  fontWeight: port.kind === 'exec' ? 600 : 500,
                }}
              >
                {port.display_name}
              </span>
              {shapeStr && (
                <span
                  style={{
                    fontSize: 10,
                    color: '#67e8f9',
                    background: '#164e63',
                    padding: '1px 5px',
                    borderRadius: 3,
                    marginLeft: 8,
                    fontFamily: 'monospace',
                  }}
                >
                  {shapeStr}
                </span>
              )}
              <BlueprintPortHandle
                nodeId={data.id}
                port={port}
                isOutput
                baseStyle={getPortHandleStyle(port, true)}
              />
            </div>
          );
        })}

        {hasError && (
          <div
            style={{
              fontSize: 10,
              background: 'rgba(239, 68, 68, 0.2)',
              color: '#f87171',
              padding: '3px 6px',
              borderRadius: 4,
              border: '1px solid #ef4444',
              marginTop: 4,
            }}
          >
            {nodeDiagnostics[0].message}
          </div>
        )}

        {data.isComposite && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              let targetId = data.targetGraphId;
              if (!targetId) {
                if (data.definitionId.includes('input_embeddings')) {
                  targetId = 'graph_input_embeddings';
                } else if (data.definitionId.includes('attention')) {
                  targetId = 'graph_attention';
                } else if (data.definitionId.includes('mlp')) {
                  targetId = 'graph_mlp';
                } else if (data.definitionId.includes('stack')) {
                  targetId = 'graph_block';
                } else if (data.definitionId.includes('block')) {
                  targetId = 'graph_block';
                }
              }
              if (targetId) {
                openGraph(targetId);
              }
            }}
            style={{
              marginTop: 6,
              width: '100%',
              background: '#1e293b',
              border: '1px solid #475569',
              color: '#94a3b8',
              padding: '4px 8px',
              borderRadius: 4,
              fontSize: 10,
              cursor: 'pointer',
            }}
          >
            Open Subgraph
          </button>
        )}
      </div>
    </div>
  );
});
