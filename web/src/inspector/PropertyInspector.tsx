/**
 * Right-side Property Inspector supporting constrained controls, dynamic dropdown providers,
 * binding selectors, parameter accounting, and tensor statistics.
 */

import React, { useState } from 'react';
import {
  Box,
  ChevronRight,
  Copy,
  PanelRight,
} from 'lucide-react';
import { ConfigRefValue, NodeDefinitionSummary } from '../api/contracts';
import { useProjectStore } from '../stores/projectStore';
import { useTraceStore } from '../stores/traceStore';
import { useUIStore } from '../stores/uiStore';
import { useValidationStore } from '../stores/validationStore';

function isConfigRef(val: unknown): val is ConfigRefValue {
  return typeof val === 'object' && val !== null && 'kind' in val && (val as ConfigRefValue).kind === 'config_ref';
}

function getNodeParamTrainable(summary: unknown): string {
  if (typeof summary === 'object' && summary !== null && 'trainable' in summary) {
    const candidate = (summary as { trainable?: unknown }).trainable;
    return typeof candidate === 'number' ? candidate.toLocaleString() : '0';
  }
  return '0';
}

function getNodeParamShapes(summary: unknown): Record<string, number[]> {
  if (typeof summary === 'object' && summary !== null && 'shapes' in summary) {
    const candidate = (summary as { shapes?: Record<string, number[]> }).shapes;
    if (typeof candidate === 'object' && candidate !== null) {
      return candidate;
    }
  }
  return {};
}

export const PropertyInspector: React.FC<{ catalog: NodeDefinitionSummary[] }> = ({ catalog }) => {
  const { project, openGraphId, updateNodeProperty, updateModelConfig, createEditableModuleCopy, updateNodeMetadata } =
    useProjectStore();
  const { selectedNodeId, selectedEdgeId, selectNode, selectEdge, isInspectorOpen, toggleInspector, repeatInstanceIndex } = useUIStore();
  const { resolvedShapes, diagnostics, parameterSummary } = useValidationStore();
  const { retainedSummaries, trainingMetrics, updateHyperparameters } = useTraceStore();

  const [activeTab, setActiveTab] = useState<'properties' | 'parameters' | 'tensors' | 'docs'>('properties');

  if (!isInspectorOpen) {
    return (
      <div
        style={{
          width: '100%',
          height: '100%',
          background: '#12141c',
          borderLeft: '1px solid #1f2430',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          paddingTop: 10,
        }}
      >
        <button
          onClick={toggleInspector}
          title="Expand Inspector"
          style={{
            background: 'transparent',
            border: 'none',
            color: '#94a3b8',
            cursor: 'pointer',
            padding: 6,
          }}
        >
          <PanelRight size={16} />
        </button>
      </div>
    );
  }

  const currentGraph = project.model.graphs[openGraphId];
  const selectedNode = currentGraph?.nodes.find((n) => n.id === selectedNodeId);
  const nodeDefn = selectedNode ? catalog.find((c) => c.type_id === selectedNode.definition_id) : null;

  // Selected edge lookup
  const selectedEdge = selectedEdgeId ? currentGraph?.edges.find((e) => e.id === selectedEdgeId) : null;
  const edgeTensorSummary = selectedEdge ? retainedSummaries[`${selectedEdge.source.node_id}:${selectedEdge.source.port_id}`] : null;

  if (selectedEdge) {
    const srcSpec = resolvedShapes[selectedEdge.source.node_id]?.[selectedEdge.source.port_id];

    return (
      <aside
        style={{
          width: '100%',
          height: '100%',
          background: 'transparent',
          display: 'flex',
          flexDirection: 'column',
          color: '#e2e8f0',
          fontSize: 12,
        }}
      >
        <div style={{ padding: '12px 14px', borderBottom: '1px solid #1f2430', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontWeight: 600, fontSize: 12, color: '#38bdf8' }}>Edge Connection</span>
          <div style={{ display: 'flex', gap: 6 }}>
            <button
              onClick={toggleInspector}
              style={{ background: 'transparent', border: 'none', color: '#64748b', cursor: 'pointer' }}
            >
              <ChevronRight size={15} />
            </button>
            <button
              onClick={() => selectEdge(null)}
              style={{ background: 'transparent', border: 'none', color: '#94a3b8', cursor: 'pointer' }}
            >
              ✕
            </button>
          </div>
        </div>
        <div style={{ padding: 14, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div>
            <span style={{ color: '#94a3b8', fontSize: 11 }}>Producer:</span>
            <p style={{ fontFamily: 'monospace', color: '#e2e8f0' }}>{selectedEdge.source.node_id}.{selectedEdge.source.port_id}</p>
          </div>
          <div>
            <span style={{ color: '#94a3b8', fontSize: 11 }}>Consumer:</span>
            <p style={{ fontFamily: 'monospace', color: '#e2e8f0' }}>{selectedEdge.target.node_id}.{selectedEdge.target.port_id}</p>
          </div>
          {srcSpec && (
            <div>
              <span style={{ color: '#94a3b8', fontSize: 11 }}>Data Contract:</span>
              <div style={{ background: '#181b24', padding: '6px 8px', borderRadius: 4, marginTop: 4, fontFamily: 'monospace' }}>
                <span style={{ color: '#22c55e' }}>{srcSpec.dtype}</span> [{srcSpec.shape?.map((d) => (typeof d === 'object' && 'name' in d ? d.name : typeof d === 'object' && 'key' in d ? d.key : typeof d === 'object' && 'value' in d ? String(d.value) : '?')).join(', ')}]
              </div>
            </div>
          )}
          {edgeTensorSummary && (
            <div>
              <span style={{ color: '#94a3b8', fontSize: 11 }}>Execution Tensor Statistics:</span>
              <div style={{ background: '#181b24', padding: 8, borderRadius: 4, marginTop: 4, display: 'flex', flexDirection: 'column', gap: 4, fontFamily: 'monospace', fontSize: 11 }}>
                <div>Shape: [{edgeTensorSummary.shape.join(', ')}]</div>
                <div>Elements: {edgeTensorSummary.numel.toLocaleString()}</div>
                {edgeTensorSummary.min !== undefined && <div>Min: {edgeTensorSummary.min}</div>}
                {edgeTensorSummary.max !== undefined && <div>Max: {edgeTensorSummary.max}</div>}
                {edgeTensorSummary.mean !== undefined && <div>Mean: {edgeTensorSummary.mean}</div>}
                {edgeTensorSummary.std !== undefined && <div>Std: {edgeTensorSummary.std}</div>}
                {edgeTensorSummary.l2_norm !== undefined && <div>L2 Norm: {edgeTensorSummary.l2_norm}</div>}
                {edgeTensorSummary.nan_count !== undefined && <div>NaN count: {edgeTensorSummary.nan_count}</div>}
                {edgeTensorSummary.sample_values && (
                  <div style={{ marginTop: 6 }}>
                    <div style={{ color: '#94a3b8', marginBottom: 2 }}>Sample Values:</div>
                    <div style={{ background: '#0f172a', padding: 4, borderRadius: 2, overflowX: 'auto' }}>
                      [{edgeTensorSummary.sample_values.map(String).join(', ')}]
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </aside>
    );
  }

  if (!selectedNode) {
    return (
      <aside
        style={{
          width: '100%',
          height: '100%',
          background: 'transparent',
          display: 'flex',
          flexDirection: 'column',
          color: '#64748b',
          fontSize: 12,
        }}
      >
        <div style={{ padding: '12px 14px', borderBottom: '1px solid #1f2430', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontWeight: 600, fontSize: 12, color: '#e2e8f0' }}>Model Configuration</span>
          <button
            onClick={toggleInspector}
            style={{ background: 'transparent', border: 'none', color: '#64748b', cursor: 'pointer' }}
          >
            <ChevronRight size={15} />
          </button>
        </div>
        <div style={{ padding: 14, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 12 }}>
          {/* Global Model Configuration Controls */}
          {Object.entries(project.model.config).map(([key, val]) => (
            <div key={key} style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <label style={{ fontSize: 11, fontWeight: 500, color: '#94a3b8' }}>
                {key.replace('_', ' ')}
              </label>
              {typeof val === 'boolean' ? (
                <select
                  value={val ? 'true' : 'false'}
                  onChange={(e) => updateModelConfig(key, e.target.value === 'true')}
                  style={{
                    background: '#181b24',
                    border: '1px solid #272c3b',
                    color: '#e2e8f0',
                    padding: '5px 8px',
                    borderRadius: 4,
                    fontSize: 11,
                  }}
                >
                  <option value="true">Enabled</option>
                  <option value="false">Disabled</option>
                </select>
              ) : key === 'attention_implementation' ? (
                <select
                  value={String(val)}
                  onChange={(e) => updateModelConfig(key, e.target.value)}
                  style={{
                    background: '#181b24',
                    border: '1px solid #272c3b',
                    color: '#e2e8f0',
                    padding: '5px 8px',
                    borderRadius: 4,
                    fontSize: 11,
                  }}
                >
                  <option value="sdpa">PyTorch SDPA</option>
                  <option value="manual">Manual Causal Attention</option>
                </select>
              ) : key === 'n_head' ? (
                // Constrained valid divisor dropdown for n_head
                <select
                  value={Number(val)}
                  onChange={(e) => updateModelConfig(key, parseInt(e.target.value, 10))}
                  style={{
                    background: '#181b24',
                    border: '1px solid #272c3b',
                    color: '#e2e8f0',
                    padding: '5px 8px',
                    borderRadius: 4,
                    fontSize: 11,
                  }}
                >
                  {(() => {
                    const n_embd = Number(project.model.config['n_embd'] || 64);
                    const divisors: number[] = [];
                    for (let d = 1; d <= n_embd; d++) {
                      if (n_embd % d === 0) divisors.push(d);
                    }
                    return divisors.map((d) => (
                      <option key={d} value={d}>
                        {d} heads (head_dim = {n_embd / d})
                      </option>
                    ));
                  })()}
                </select>
              ) : (
                <input
                  type={typeof val === 'number' ? 'number' : 'text'}
                  value={val as string | number}
                  onChange={(e) => {
                    const nextVal = typeof val === 'number' ? parseFloat(e.target.value) || 0 : e.target.value;
                    updateModelConfig(key, nextVal);
                  }}
                  style={{
                    background: '#181b24',
                    border: '1px solid #272c3b',
                    color: '#e2e8f0',
                    padding: '5px 8px',
                    borderRadius: 4,
                    fontSize: 11,
                  }}
                />
              )}
            </div>
          ))}

          {/* Live Hyperparameter Tweakers */}
          <div style={{ marginTop: 14, paddingTop: 12, borderTop: '1px solid #1f2430', display: 'flex', flexDirection: 'column', gap: 10 }}>
            <span style={{ fontWeight: 600, fontSize: 11, color: '#38bdf8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Live Hyperparameter Tweakers
            </span>

            {/* Learning Rate Tweaker */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11 }}>
                <span style={{ color: '#94a3b8' }}>Learning Rate:</span>
                <span style={{ color: '#22c55e', fontFamily: 'monospace' }}>
                  {trainingMetrics?.learning_rate ? trainingMetrics.learning_rate.toExponential(2) : '6.00e-4'}
                </span>
              </div>
              <input
                type="range"
                min="-5"
                max="-2"
                step="0.1"
                defaultValue="-3.22"
                onChange={(e) => {
                  const lr = Math.pow(10, parseFloat(e.target.value));
                  updateHyperparameters({ learning_rate: lr });
                }}
                style={{ width: '100%', cursor: 'pointer' }}
              />
            </div>

            {/* Weight Decay Tweaker */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11 }}>
                <span style={{ color: '#94a3b8' }}>Weight Decay:</span>
                <span style={{ color: '#38bdf8', fontFamily: 'monospace' }}>0.10</span>
              </div>
              <input
                type="range"
                min="0"
                max="0.5"
                step="0.01"
                defaultValue="0.1"
                onChange={(e) => {
                  const wd = parseFloat(e.target.value);
                  updateHyperparameters({ weight_decay: wd });
                }}
                style={{ width: '100%', cursor: 'pointer' }}
              />
            </div>

            {/* Gradient Clip Tweaker */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11 }}>
                <span style={{ color: '#94a3b8' }}>Grad Clip:</span>
                <span style={{ color: '#f59e0b', fontFamily: 'monospace' }}>1.0</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="5.0"
                step="0.1"
                defaultValue="1.0"
                onChange={(e) => {
                  const gc = parseFloat(e.target.value);
                  updateHyperparameters({ grad_clip: gc });
                }}
                style={{ width: '100%', cursor: 'pointer' }}
              />
            </div>
          </div>
        </div>
      </aside>
    );
  }

  const configKeys = Object.keys(project.model.config);
  const nodeDiags = diagnostics.filter((d) => d.node_id === selectedNode.id);
  const repeatIndex = typeof repeatInstanceIndex === 'number' ? repeatInstanceIndex : useUIStore.getState().repeatInstanceIndex;
  let nodeTensorSummary = null;
  if (typeof repeatIndex === 'number') {
    const indexedKey = `${selectedNode.id}[${repeatIndex}]`;
    const summaryKeys = Object.keys(retainedSummaries);
    const matchedKey = summaryKeys.find(
      (k) => (k.includes(selectedNode.id) || k.startsWith(selectedNode.id)) && k.includes(`[${repeatIndex}]`)
    );
    nodeTensorSummary =
      (matchedKey ? retainedSummaries[matchedKey] : null) ||
      retainedSummaries[indexedKey] ||
      retainedSummaries[`${indexedKey}:output`];
  }
  if (!nodeTensorSummary) {
    nodeTensorSummary = retainedSummaries[selectedNode.id] || retainedSummaries[`${selectedNode.id}:output`];
  }
  const nodeParamSummary = parameterSummary.breakdown_by_node?.[selectedNode.id];
  const paramShapes = getNodeParamShapes(nodeParamSummary);

  return (
    <aside
      style={{
        width: '100%',
        height: '100%',
        background: 'transparent',
        display: 'flex',
        flexDirection: 'column',
        color: '#e2e8f0',
        fontSize: 12,
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: '10px 14px',
          borderBottom: '1px solid #1f2430',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <Box size={15} color="#38bdf8" />
          <span style={{ fontWeight: 600, fontSize: 12 }}>{selectedNode.display_name}</span>
        </div>
        <div style={{ display: 'flex', gap: 4 }}>
          <button
            onClick={toggleInspector}
            style={{ background: 'transparent', border: 'none', color: '#64748b', cursor: 'pointer' }}
          >
            <ChevronRight size={15} />
          </button>
          <button
            onClick={() => selectNode(null)}
            style={{ background: 'transparent', border: 'none', color: '#94a3b8', cursor: 'pointer' }}
          >
            ✕
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div
        style={{
          display: 'flex',
          borderBottom: '1px solid #1f2430',
          background: '#151822',
        }}
      >
        {(['properties', 'parameters', 'tensors', 'docs'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            style={{
              flex: 1,
              padding: '7px 2px',
              fontSize: 10,
              fontWeight: 500,
              background: activeTab === tab ? '#1e2330' : 'transparent',
              color: activeTab === tab ? '#38bdf8' : '#94a3b8',
              border: 'none',
              borderBottom: activeTab === tab ? '2px solid #38bdf8' : 'none',
              cursor: 'pointer',
              textTransform: 'capitalize',
            }}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflowY: 'auto', padding: 14, display: 'flex', flexDirection: 'column', gap: 12 }}>
        {activeTab === 'properties' && (
          <>
            {/* Identity */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <label style={{ fontSize: 11, fontWeight: 500, color: '#94a3b8' }}>Display Name</label>
              <input
                type="text"
                value={selectedNode.display_name}
                onChange={(e) => {
                  useProjectStore.getState().updateNodeDisplayName(selectedNode.id, e.target.value);
                }}
                style={{
                  background: '#181b24',
                  border: '1px solid #272c3b',
                  color: '#e2e8f0',
                  padding: '5px 8px',
                  borderRadius: 4,
                  fontSize: 11,
                }}
              />
            </div>

            {/* Properties from schema */}
            {nodeDefn &&
              Object.entries(nodeDefn.property_schema.properties || {}).map(([propKey]) => {
                const currentVal = selectedNode.properties[propKey];
                const configRefActive = isConfigRef(currentVal);

                return (
                  <div key={propKey} style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <label style={{ fontSize: 11, fontWeight: 500, color: '#94a3b8' }}>
                        {propKey.replace('_', ' ')}
                      </label>
                      {/* Value source selector */}
                      <select
                        value={configRefActive ? 'config_ref' : 'literal'}
                        onChange={(e) => {
                          if (e.target.value === 'config_ref') {
                            updateNodeProperty(selectedNode.id, propKey, {
                              kind: 'config_ref',
                              key: configKeys[0] || 'n_embd',
                            });
                          } else {
                            updateNodeProperty(selectedNode.id, propKey, 64);
                          }
                        }}
                        style={{
                          background: 'transparent',
                          border: 'none',
                          color: '#60a5fa',
                          fontSize: 10,
                          cursor: 'pointer',
                        }}
                      >
                        <option value="literal">Literal</option>
                        <option value="config_ref">Config Ref</option>
                      </select>
                    </div>

                    {configRefActive ? (
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <select
                          value={currentVal.key}
                          onChange={(e) =>
                            updateNodeProperty(selectedNode.id, propKey, {
                              kind: 'config_ref',
                              key: e.target.value,
                            })
                          }
                          style={{
                            flex: 1,
                            background: '#181b24',
                            border: '1px solid #272c3b',
                            color: '#93c5fd',
                            padding: '5px 8px',
                            borderRadius: 4,
                            fontSize: 11,
                          }}
                        >
                          {configKeys.map((k) => (
                            <option key={k} value={k}>
                              {k} = {String(project.model.config[k])}
                            </option>
                          ))}
                        </select>
                      </div>
                    ) : typeof currentVal === 'boolean' ? (
                      <select
                        value={currentVal ? 'true' : 'false'}
                        onChange={(e) => updateNodeProperty(selectedNode.id, propKey, e.target.value === 'true')}
                        style={{
                          background: '#181b24',
                          border: '1px solid #272c3b',
                          color: '#e2e8f0',
                          padding: '5px 8px',
                          borderRadius: 4,
                          fontSize: 11,
                        }}
                      >
                        <option value="true">True</option>
                        <option value="false">False</option>
                      </select>
                    ) : (
                      <input
                        type={typeof currentVal === 'number' ? 'number' : 'text'}
                        value={currentVal !== undefined ? String(currentVal) : ''}
                        onChange={(e) => {
                          const val = typeof currentVal === 'number' ? parseFloat(e.target.value) || 0 : e.target.value;
                          updateNodeProperty(selectedNode.id, propKey, val);
                        }}
                        style={{
                          background: '#181b24',
                          border: '1px solid #272c3b',
                          color: '#e2e8f0',
                          padding: '5px 8px',
                          borderRadius: 4,
                          fontSize: 11,
                        }}
                      />
                    )}
                  </div>
                );
              })}

            {/* Custom Module Fork Button */}
            {nodeDefn?.is_composite && !selectedNode.definition_id.startsWith('custom.') && (
              <div style={{ marginTop: 6 }}>
                <button
                  onClick={() => createEditableModuleCopy(selectedNode.id)}
                  style={{
                    width: '100%',
                    background: '#1e293b',
                    border: '1px solid #334155',
                    color: '#38bdf8',
                    padding: '6px 10px',
                    borderRadius: 4,
                    fontSize: 11,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: 5,
                    fontWeight: 500,
                  }}
                >
                  <Copy size={12} />
                  Create Editable Copy
                </button>
              </div>
            )}
            {/* Metadata: Disabled and Notes */}
            <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 6, borderTop: '1px solid #1e293b', paddingTop: 8 }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, cursor: 'pointer', color: '#e2e8f0' }}>
                <input
                  type="checkbox"
                  aria-label="Disabled"
                  checked={Boolean(selectedNode.metadata?.disabled)}
                  onChange={(e) => updateNodeMetadata(selectedNode.id, { disabled: e.target.checked })}
                />
                Disabled
              </label>
              <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 11, color: '#94a3b8' }}>
                <span>Notes</span>
                <textarea
                  aria-label="Notes"
                  value={selectedNode.metadata?.notes ?? ''}
                  onChange={(e) => updateNodeMetadata(selectedNode.id, { notes: e.target.value })}
                  placeholder="Node notes / description..."
                  rows={2}
                  style={{
                    background: '#181b24',
                    border: '1px solid #272c3b',
                    color: '#e2e8f0',
                    padding: '4px 6px',
                    borderRadius: 4,
                    fontSize: 11,
                    resize: 'vertical',
                    fontFamily: 'inherit',
                  }}
                />
              </label>
            </div>

            {/* Diagnostics */}
            {nodeDiags.length > 0 && (
              <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 4 }}>
                <span style={{ fontSize: 10, fontWeight: 600, color: '#f87171' }}>Diagnostics:</span>
                {nodeDiags.map((d, i) => (
                  <div
                    key={i}
                    style={{
                      background: 'rgba(239, 68, 68, 0.15)',
                      border: '1px solid #ef4444',
                      padding: 6,
                      borderRadius: 4,
                      fontSize: 10,
                      color: '#fca5a5',
                    }}
                  >
                    <div>{d.message}</div>
                    {d.suggestions?.length > 0 && (
                      <div style={{ marginTop: 3, color: '#93c5fd', fontSize: 9 }}>
                        Suggestion: {d.suggestions[0]}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </>
        )}

        {activeTab === 'parameters' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {nodeParamSummary ? (
              <>
                <div style={{ background: '#181b24', padding: 8, borderRadius: 4 }}>
                  <span style={{ color: '#94a3b8', fontSize: 10 }}>Trainable Parameters:</span>
                  <p style={{ fontSize: 15, fontWeight: 700, color: '#22c55e' }}>
                    {getNodeParamTrainable(nodeParamSummary)}
                  </p>
                </div>
                {Object.keys(paramShapes).length > 0 && (
                  <div>
                    <span style={{ color: '#94a3b8', fontSize: 10 }}>Parameter Shapes:</span>
                    <div style={{ background: '#181b24', padding: 6, borderRadius: 4, marginTop: 4, fontFamily: 'monospace', fontSize: 10 }}>
                      {Object.entries(paramShapes).map(([pName, pShape]) => (
                        <div key={pName} style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span>{pName}:</span>
                          <span style={{ color: '#38bdf8' }}>[{pShape.join(', ')}]</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div style={{ color: '#64748b' }}>No learnable parameters for this node.</div>
            )}
          </div>
        )}

        {activeTab === 'tensors' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {nodeTensorSummary ? (
              <div style={{ background: '#181b24', padding: 8, borderRadius: 4, display: 'flex', flexDirection: 'column', gap: 3, fontFamily: 'monospace', fontSize: 10 }}>
                <div>Shape: [{nodeTensorSummary.shape.join(', ')}]</div>
                <div>DType: {nodeTensorSummary.dtype}</div>
                <div>Elements: {nodeTensorSummary.numel.toLocaleString()}</div>
                {nodeTensorSummary.min !== undefined && <div>Min: {nodeTensorSummary.min}</div>}
                {nodeTensorSummary.max !== undefined && <div>Max: {nodeTensorSummary.max}</div>}
                {nodeTensorSummary.mean !== undefined && <div>Mean: {nodeTensorSummary.mean}</div>}
                {nodeTensorSummary.std !== undefined && <div>Std: {nodeTensorSummary.std}</div>}
                {nodeTensorSummary.l2_norm !== undefined && <div>Norm: {nodeTensorSummary.l2_norm}</div>}
                {nodeTensorSummary.sample_values && (
                  <div style={{ marginTop: 4 }}>
                    <div style={{ color: '#94a3b8', marginBottom: 2 }}>Sample Values:</div>
                    <div style={{ background: '#0f172a', padding: 4, borderRadius: 2, overflowX: 'auto' }}>
                      [{nodeTensorSummary.sample_values.map(String).join(', ')}]
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div style={{ color: '#64748b' }}>Execute a batch to inspect live activations.</div>
            )}
          </div>
        )}

        {activeTab === 'docs' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <p style={{ color: '#94a3b8', lineHeight: '1.4em', fontSize: 11 }}>{nodeDefn?.description || 'No documentation available.'}</p>
            <div style={{ background: '#181b24', padding: 6, borderRadius: 4, fontSize: 10, color: '#64748b' }}>
              Type: {nodeDefn?.type_id}
            </div>
          </div>
        )}
      </div>
    </aside>
  );
};
