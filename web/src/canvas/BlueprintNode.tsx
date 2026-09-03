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
  ChevronDown,
  ChevronRight,
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
  disabled?: boolean;
  collapsed?: boolean;
}

export function compactPropertySummary(properties?: Record<string, unknown>): string | null {
  if (!properties || typeof properties !== 'object') return null;

  const keyMap: Array<{ key: string; label: string }> = [
    { key: 'n_embd', label: 'C' },
    { key: 'n_head', label: 'heads' },
    { key: 'n_kv_head', label: 'kv' },
    { key: 'attention_implementation', label: 'attention_implementation' },
    { key: 'in_features', label: 'in_features' },
    { key: 'out_features', label: 'out_features' },
    { key: 'hidden_dim', label: 'hidden_dim' },
    { key: 'dropout', label: 'dropout' },
  ];

  const parts: string[] = [];
  for (const { key, label } of keyMap) {
    if (key in properties) {
      const val = properties[key];
      if (val === undefined || val === null || val === '') continue;
      let displayVal: string;
      if (val && typeof val === 'object' && 'kind' in val && val.kind === 'config_ref' && 'key' in val) {
        displayVal = String(val.key ?? '');
      } else {
        displayVal = String(val);
      }
      if (!displayVal) continue;
      parts.push(`${label}: ${displayVal}`);
      if (parts.length === 3) break;
    }
  }

  return parts.length > 0 ? parts.join(' | ') : null;
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
      transform: 'translateY(-50%) rotate(45deg)',
      left: isOutput ? undefined : -16,
      right: isOutput ? -16 : undefined,
      top: '50%',
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
    transform: 'translateY(-50%)',
    left: isOutput ? undefined : -16,
    right: isOutput ? -16 : undefined,
    top: '50%',
    border: '2px solid #0f172a',
    zIndex: 10,
  };
}

export const BlueprintNode = memo(({ data, selected }: { data: BlueprintNodeData; selected?: boolean }) => {
  const { openGraph, setNodeBreakpoint, toggleNodeCollapsed } = useProjectStore();
  const { selectedNodeId, selectedNodeIds, selectNode, repeatInstanceIndex } = useUIStore();
  const { resolvedShapes, diagnostics } = useValidationStore();
  const { nodeStates, nodeGradientNorms, parameterNorms } = useTraceStore();

  const isSelected =
    Boolean(selected) ||
    selectedNodeId === data.id ||
    (Array.isArray(selectedNodeIds) && selectedNodeIds.includes(data.id));

  let executionState = nodeStates[data.id] || 'pending';
  if (typeof repeatInstanceIndex === 'number') {
    const keys = Object.keys(nodeStates);
    const hasIndexedKey = keys.some((k) => k.includes('[') && (k.includes(data.id) || k.endsWith(`/${data.id}`)));
    if (hasIndexedKey) {
      const match = keys.find(
        (k) => (k.includes(data.id) || k.endsWith(`/${data.id}`)) && k.includes(`[${repeatInstanceIndex}]`)
      );
      if (match && nodeStates[match]) {
        executionState = nodeStates[match];
      }
    }
  }
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
  const pNorm = parameterNorms?.[data.id] ?? parameterNorms?.[`${data.id}.weight`];
  let gradStatusColor: string | null = null;
  if (gNorm !== undefined) {
    if (gNorm > 10.0) gradStatusColor = '#ef4444';
    else if (gNorm < 1e-4) gradStatusColor = '#3b82f6';
    else gradStatusColor = '#22c55e';
  }

  if (data.definitionId === 'builtin.comment@1') {
    return (
      <div
        onClick={(e) => {
          e.stopPropagation();
          selectNode(data.id);
        }}
        style={{
          minWidth: 180,
          background: '#3f3a1d',
          border: '1px solid #a16207',
          borderRadius: 8,
          boxShadow: isSelected
            ? '0 0 16px rgba(245, 158, 11, 0.4)'
            : '0 4px 12px rgba(0, 0, 0, 0.5)',
          color: '#fef08a',
          fontSize: 12,
          padding: '8px 12px',
          opacity: data.disabled ? 0.45 : 1,
          cursor: 'pointer',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontWeight: 600, marginBottom: 4, color: '#fde047' }}>
          <IconComponent size={14} />
          <span>{data.displayName}</span>
        </div>
        <div style={{ fontSize: 11, color: '#fef9c3', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
          {String(data.properties.text ?? '')}
        </div>
      </div>
    );
  }

  return (
    <div
      onClick={(e) => {
        e.stopPropagation();
        selectNode(data.id);
      }}
      style={{
        minWidth: 220,
        position: 'relative',
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
        overflow: 'visible',
        transition: 'all 0.15s ease',
        opacity: data.disabled ? 0.45 : 1,
      }}
    >
      <div
        style={{
          background: categoryColor.bg,
          padding: '6px 10px',
          borderTopLeftRadius: 6,
          borderTopRightRadius: 6,
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
            {data.definitionId.startsWith('custom.') && !data.displayName.endsWith('*') && (
              <span title="Custom module" style={{ color: '#fbbf24', marginLeft: 2 }}>*</span>
            )}
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {data.disabled && (
            <span
              style={{
                fontSize: 10,
                fontWeight: 600,
                background: '#ef4444',
                color: '#fff',
                padding: '1px 5px',
                borderRadius: 4,
                letterSpacing: '0.04em',
              }}
            >
              Disabled
            </span>
          )}
          {pNorm !== undefined && (
            <span
              title={`Param L2: ${pNorm.toFixed(4)}`}
              style={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                background: '#f59e0b',
                display: 'inline-block',
                boxShadow: '0 0 6px #f59e0b',
              }}
            />
          )}
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
            aria-label={data.collapsed ? 'Expand node' : 'Collapse node'}
            title={data.collapsed ? 'Expand node' : 'Collapse node'}
            onClick={(e) => {
              e.stopPropagation();
              toggleNodeCollapsed(data.id);
            }}
            style={{
              background: 'transparent',
              border: 'none',
              cursor: 'pointer',
              padding: 2,
              display: 'flex',
              alignItems: 'center',
              color: '#94a3b8',
            }}
          >
            {data.collapsed ? <ChevronRight size={14} /> : <ChevronDown size={14} />}
          </button>

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
      {!data.collapsed && compactPropertySummary(data.properties) && (
        <div
          style={{
            fontSize: 10,
            color: '#94a3b8',
            padding: '4px 10px 0',
          }}
        >
          {compactPropertySummary(data.properties)}
        </div>
      )}
      <div style={{ padding: data.collapsed ? '2px 10px' : '8px 10px', display: 'flex', flexDirection: 'column', gap: data.collapsed ? 0 : 6 }}>
        {(() => {
          const rowCount = Math.max(data.inputs.length, data.outputs.length);
          if (rowCount === 0) return null;
          return Array.from({ length: rowCount }).map((_, idx) => {
            const inPort = data.inputs[idx] || null;
            const outPort = data.outputs[idx] || null;

            return (
              <div
                key={idx}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  minHeight: data.collapsed ? 4 : 20,
                  height: data.collapsed ? 4 : undefined,
                  position: 'relative',
                  gap: 16,
                }}
              >
                {/* Left Side: Input Pin & Name */}
                {inPort ? (
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 6,
                      position: 'relative',
                      paddingLeft: 2,
                    }}
                  >
                    <BlueprintPortHandle
                      nodeId={data.id}
                      port={inPort}
                      isOutput={false}
                      baseStyle={getPortHandleStyle(inPort, false)}
                    />
                    {!data.collapsed && (
                      <>
                        {(() => {
                          const spec = resolvedShapes[data.id]?.[inPort.id];
                          const shapeStr = getShapeString(spec);
                          return shapeStr ? (
                            <span
                              style={{
                                fontSize: 10,
                                color: '#67e8f9',
                                background: '#164e63',
                                padding: '1px 5px',
                                borderRadius: 3,
                                fontFamily: 'monospace',
                                marginRight: 8,
                              }}
                            >
                              {shapeStr}
                            </span>
                          ) : null;
                        })()}
                        <span
                          style={{
                            color: inPort.kind === 'exec' ? '#ffffff' : '#94a3b8',
                            fontSize: 11,
                            fontWeight: inPort.kind === 'exec' ? 600 : 400,
                          }}
                        >
                          {inPort.display_name}
                        </span>
                      </>
                    )}
                  </div>
                ) : (
                  <div style={{ flex: 1 }} />
                )}

                {/* Right Side: Output Name, Shape & Pin */}
                {outPort ? (
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 6,
                      position: 'relative',
                      paddingRight: 2,
                      marginLeft: 'auto',
                      justifyContent: 'flex-end',
                    }}
                  >
                    {!data.collapsed && (
                      <>
                        <span
                          style={{
                            color: outPort.kind === 'exec' ? '#ffffff' : '#e2e8f0',
                            fontSize: 11,
                            fontWeight: outPort.kind === 'exec' ? 600 : 500,
                          }}
                        >
                          {outPort.display_name}
                        </span>
                        {(() => {
                          const spec = resolvedShapes[data.id]?.[outPort.id];
                          const shapeStr = getShapeString(spec);
                          return shapeStr ? (
                            <span
                              style={{
                                fontSize: 10,
                                color: '#67e8f9',
                                background: '#164e63',
                                padding: '1px 5px',
                                borderRadius: 3,
                                fontFamily: 'monospace',
                              }}
                            >
                              {shapeStr}
                            </span>
                          ) : null;
                        })()}
                      </>
                    )}
                    <BlueprintPortHandle
                      nodeId={data.id}
                      port={outPort}
                      isOutput={true}
                      baseStyle={getPortHandleStyle(outPort, true)}
                    />
                  </div>
                ) : (
                  <div style={{ flex: 1 }} />
                )}
              </div>
            );
          });
        })()}
        {!data.collapsed && hasError && (
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

        {!data.collapsed && data.isComposite && (
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
