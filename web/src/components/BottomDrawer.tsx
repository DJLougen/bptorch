/**
 * Expandable bottom drawer for Live Loss Plotter, Training Metrics Dashboard,
 * Tensor Inspection, Diagnostics, Parameter Accounting, Reference Parity, and Execution Logs.
 */

import React from 'react';
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Cpu,
  Layers,
  LineChart,
  Play,
  ShieldCheck,
  Square,
  Terminal,
  Zap,
} from 'lucide-react';
import { ApiClient } from '../api/client';
import { TestSuiteResult } from '../api/contracts';
import { DrawerTab, useUIStore } from '../stores/uiStore';
import { useProjectStore } from '../stores/projectStore';
import { useTraceStore } from '../stores/traceStore';
import { useValidationStore } from '../stores/validationStore';
export const BottomDrawer: React.FC = () => {
  const { activeDrawerTab, isDrawerOpen, openDrawerTab, closeDrawer, toggleDrawer, selectedNodeId, selectedEdgeId } =
    useUIStore();
  const { project } = useProjectStore();
  const isNanoGPTBaseline =
    project.project.id === 'nanogpt_default' &&
    project.model.config['attention_implementation'] !== undefined;
  const { diagnostics, parameterSummary } = useValidationStore();
  const {
    logs,
    retainedSummaries,
    trainingMetrics,
    lossHistory,
    isTraining,
    startTraining,
    pauseTraining,
    stepBatch,
    stop,
    addLog,
  } = useTraceStore();

  const [testSuiteResult, setTestSuiteResult] = React.useState<TestSuiteResult | null>(null);
  const [testSuiteError, setTestSuiteError] = React.useState<string | null>(null);
  const [isRunningSuite, setIsRunningSuite] = React.useState<boolean>(false);
  if (!isDrawerOpen || !activeDrawerTab) {
    return (
      <div
        style={{
          height: '100%',
          background: 'transparent',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 16px',
          fontSize: 11,
          color: '#94a3b8',
          userSelect: 'none',
        }}
      >
        <div style={{ display: 'flex', gap: 12 }}>
          <button
            onClick={() => openDrawerTab('loss')}
            style={{
              background: 'transparent',
              border: 'none',
              color: isTraining ? '#22c55e' : '#38bdf8',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 4,
              fontSize: 11,
              fontWeight: isTraining ? 600 : 400,
            }}
          >
            <LineChart size={12} />
            Live Loss {lossHistory.length > 0 && `(${lossHistory[lossHistory.length - 1].loss.toFixed(4)})`}
          </button>
          <button
            onClick={() => openDrawerTab('tester')}
            style={{
              background: 'transparent',
              border: 'none',
              color: testSuiteResult?.failed === 0 ? '#22c55e' : '#38bdf8',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 4,
              fontSize: 11,
            }}
          >
            <ShieldCheck size={12} />
            Tester {testSuiteResult && `(${testSuiteResult.passed}/${testSuiteResult.total})`}
          </button>
          <button
            onClick={() => openDrawerTab('metrics')}
            style={{
              background: 'transparent',
              border: 'none',
              color: '#94a3b8',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 4,
              fontSize: 11,
            }}
          >
            <Activity size={12} />
            Metrics {trainingMetrics && `[Step ${trainingMetrics.step}]`}
          </button>
          <button
            onClick={() => openDrawerTab('diagnostics')}
            style={{
              background: 'transparent',
              border: 'none',
              color: diagnostics.length > 0 ? '#f87171' : '#94a3b8',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 4,
              fontSize: 11,
            }}
          >
            {diagnostics.length > 0 ? <AlertTriangle size={12} /> : <CheckCircle2 size={12} />}
            Diagnostics ({diagnostics.length})
          </button>
          <button
            onClick={() => openDrawerTab('logs')}
            style={{
              background: 'transparent',
              border: 'none',
              color: '#94a3b8',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 4,
              fontSize: 11,
            }}
          >
            <Terminal size={12} />
            Logs ({logs.length})
          </button>
        </div>

        <button
          onClick={toggleDrawer}
          style={{ background: 'transparent', border: 'none', color: '#94a3b8', cursor: 'pointer' }}
        >
          <ChevronUp size={14} />
        </button>
      </div>
    );
  }

  // Selected item tensor summary
  const targetId = selectedNodeId || selectedEdgeId;
  const tensorSummary = targetId ? retainedSummaries[targetId] || retainedSummaries[`${targetId}:output`] : null;

  // Render SVG Sparkline / Loss Curve
  const renderLossPlot = () => {
    if (lossHistory.length < 2) {
      return (
        <div style={{ color: '#64748b', display: 'flex', alignItems: 'center', justifyContent: 'center', height: 140 }}>
          Launch training to visualize live loss trajectory and learning rate schedule.
        </div>
      );
    }

    const width = 600;
    const height = 130;
    const padding = 20;

    const maxLoss = Math.max(...lossHistory.map((p) => p.loss), 0.1);
    const minLoss = Math.min(...lossHistory.map((p) => p.loss), 0.0);

    const points = lossHistory.map((p, idx) => {
      const x = padding + (idx / (lossHistory.length - 1)) * (width - 2 * padding);
      const y = height - padding - ((p.loss - minLoss) / Math.max(maxLoss - minLoss, 1e-4)) * (height - 2 * padding);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });

    const pathData = `M ${points.join(' L ')}`;

    return (
      <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
        <svg width={width} height={height} style={{ background: '#0a0c12', borderRadius: 6, border: '1px solid #1f2430' }}>
          <path d={pathData} fill="none" stroke="#22c55e" strokeWidth="2" />
          {lossHistory.map((p, idx) => {
            if (idx % Math.ceil(lossHistory.length / 10) === 0 || idx === lossHistory.length - 1) {
              const x = padding + (idx / (lossHistory.length - 1)) * (width - 2 * padding);
              const y = height - padding - ((p.loss - minLoss) / Math.max(maxLoss - minLoss, 1e-4)) * (height - 2 * padding);
              return <circle key={idx} cx={x} cy={y} r="3" fill="#38bdf8" />;
            }
            return null;
          })}
        </svg>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 11 }}>
          <div style={{ color: '#94a3b8' }}>Live Loss Stats:</div>
          <div>Current Loss: <span style={{ color: '#22c55e', fontWeight: 600 }}>{lossHistory[lossHistory.length - 1].loss.toFixed(5)}</span></div>
          <div>Min Loss: <span style={{ color: '#38bdf8' }}>{minLoss.toFixed(5)}</span></div>
          <div>Max Loss: <span style={{ color: '#f87171' }}>{maxLoss.toFixed(5)}</span></div>
          <div>Throughput: <span style={{ color: '#f59e0b' }}>{lossHistory[lossHistory.length - 1].tokens_per_sec.toFixed(0)} tok/s</span></div>
        </div>
      </div>
    );
  };

  return (
    <div
      style={{
        height: '100%',
        background: 'transparent',
        display: 'flex',
        flexDirection: 'column',
        fontSize: 12,
        color: '#e2e8f0',
        zIndex: 15,
      }}
    >
      {/* Drawer Header & Tabs */}
      <div
        style={{
          height: 34,
          background: '#141622',
          borderBottom: '1px solid #1f2430',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 12px',
        }}
      >
        <div style={{ display: 'flex', gap: 4 }}>
          {(
            [
              { id: 'loss', label: 'Live Loss Plotter', icon: LineChart },
              { id: 'metrics', label: 'Metrics Dashboard', icon: Activity },
              { id: 'tester', label: 'Architecture Tester', icon: ShieldCheck },
              { id: 'tensor', label: 'Tensor Inspector', icon: Zap },
              { id: 'diagnostics', label: `Diagnostics (${diagnostics.length})`, icon: AlertTriangle },
              { id: 'parameters', label: 'Parameters', icon: Layers },
              { id: 'parity', label: 'Reference Parity', icon: Cpu },
              { id: 'logs', label: `Logs (${logs.length})`, icon: Terminal },
            ] as const
          ).map((tab) => {
            const Icon = tab.icon;
            const isActive = activeDrawerTab === tab.id;

            return (
              <button
                key={tab.id}
                onClick={() => openDrawerTab(tab.id as DrawerTab)}
                style={{
                  background: isActive ? '#1e2330' : 'transparent',
                  border: 'none',
                  borderBottom: isActive ? '2px solid #38bdf8' : 'none',
                  color: isActive ? '#38bdf8' : '#94a3b8',
                  padding: '6px 10px',
                  fontSize: 11,
                  fontWeight: 500,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 5,
                }}
              >
                <Icon size={12} />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Quick Training Controls in Header */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <button
            onClick={() => {
              if (project) {
                if (isTraining) {
                  pauseTraining();
                } else {
                  startTraining(project, 100);
                }
              }
            }}
            style={{
              background: isTraining ? '#b45309' : '#15803d',
              border: 'none',
              color: '#ffffff',
              borderRadius: 4,
              padding: '3px 8px',
              fontSize: 11,
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 4,
            }}
          >
            <Play size={11} />
            {isTraining ? 'Pause' : 'Train (100 Steps)'}
          </button>
          <button
            onClick={() => stepBatch(project)}
            style={{
              background: '#1e293b',
              border: '1px solid #334155',
              color: '#e2e8f0',
              borderRadius: 4,
              padding: '3px 8px',
              fontSize: 11,
              cursor: 'pointer',
            }}
          >
            Step Batch
          </button>
          <button
            onClick={stop}
            style={{
              background: '#374151',
              border: 'none',
              color: '#f87171',
              borderRadius: 4,
              padding: '3px 6px',
              cursor: 'pointer',
            }}
            title="Stop Run"
          >
            <Square size={11} />
          </button>
          <button
            onClick={closeDrawer}
            style={{ background: 'transparent', border: 'none', color: '#94a3b8', cursor: 'pointer', padding: 2 }}
          >
            <ChevronDown size={14} />
          </button>
        </div>
      </div>

      {/* Drawer Body Content */}
      <div style={{ flex: 1, overflowY: 'auto', padding: 14 }}>
        {activeDrawerTab === 'loss' && (
          <div>
            {renderLossPlot()}
          </div>
        )}

        {activeDrawerTab === 'metrics' && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 12 }}>
            <div style={{ background: '#181b24', padding: '10px 14px', borderRadius: 6, border: '1px solid #272c3b' }}>
              <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 4 }}>Training Step</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: '#38bdf8' }}>
                {trainingMetrics?.step ?? 0}
              </div>
              <div style={{ fontSize: 10, color: '#64748b' }}>Epoch {trainingMetrics?.epoch ?? 0}</div>
            </div>

            <div style={{ background: '#181b24', padding: '10px 14px', borderRadius: 6, border: '1px solid #272c3b' }}>
              <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 4 }}>Current Loss</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: '#22c55e' }}>
                {trainingMetrics ? trainingMetrics.loss.toFixed(4) : '—'}
              </div>
              <div style={{ fontSize: 10, color: '#64748b' }}>Avg: {trainingMetrics?.avg_loss ? trainingMetrics.avg_loss.toFixed(4) : '—'}</div>
            </div>

            <div style={{ background: '#181b24', padding: '10px 14px', borderRadius: 6, border: '1px solid #272c3b' }}>
              <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 4 }}>Learning Rate</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: '#f59e0b' }}>
                {trainingMetrics ? trainingMetrics.learning_rate.toExponential(2) : '—'}
              </div>
              <div style={{ fontSize: 10, color: '#64748b' }}>Cosine Warmup</div>
            </div>

            <div style={{ background: '#181b24', padding: '10px 14px', borderRadius: 6, border: '1px solid #272c3b' }}>
              <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 4 }}>Gradient Norm</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: trainingMetrics?.grad_status === 'healthy' ? '#22c55e' : '#ef4444' }}>
                {trainingMetrics ? trainingMetrics.grad_norm.toFixed(3) : '—'}
              </div>
              <div style={{ fontSize: 10, color: '#64748b', textTransform: 'capitalize' }}>Status: {trainingMetrics?.grad_status ?? 'N/A'}</div>
            </div>

            <div style={{ background: '#181b24', padding: '10px 14px', borderRadius: 6, border: '1px solid #272c3b' }}>
              <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 4 }}>Throughput</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: '#a855f7' }}>
                {trainingMetrics ? `${trainingMetrics.tokens_per_sec.toFixed(0)} tok/s` : '—'}
              </div>
              <div style={{ fontSize: 10, color: '#64748b' }}>Step: {trainingMetrics?.step_time_ms.toFixed(1) ?? '—'} ms</div>
            </div>
          </div>
        )}

        {activeDrawerTab === 'tensor' && (
          <div>
            {tensorSummary ? (
              <div style={{ display: 'flex', gap: 24, fontFamily: 'monospace', fontSize: 11 }}>
                <div>
                  <div style={{ color: '#94a3b8', marginBottom: 4 }}>Tensor Properties:</div>
                  <div>Shape: [{tensorSummary.shape.join(', ')}]</div>
                  <div>DType: {tensorSummary.dtype}</div>
                  <div>Device: {tensorSummary.device}</div>
                  <div>Elements: {tensorSummary.numel.toLocaleString()}</div>
                </div>
                <div>
                  <div style={{ color: '#94a3b8', marginBottom: 4 }}>Summary Statistics:</div>
                  {tensorSummary.min !== undefined && <div>Min: {tensorSummary.min}</div>}
                  {tensorSummary.max !== undefined && <div>Max: {tensorSummary.max}</div>}
                  {tensorSummary.mean !== undefined && <div>Mean: {tensorSummary.mean}</div>}
                  {tensorSummary.std !== undefined && <div>Std: {tensorSummary.std}</div>}
                  {tensorSummary.l2_norm !== undefined && <div>L2 Norm: {tensorSummary.l2_norm}</div>}
                </div>
                {tensorSummary.sample_values && (
                  <div style={{ flex: 1 }}>
                    <div style={{ color: '#94a3b8', marginBottom: 4 }}>Deterministic Sample Values:</div>
                    <div style={{ background: '#0a0c12', padding: 8, borderRadius: 4, overflowX: 'auto' }}>
                      [{tensorSummary.sample_values.map(String).join(', ')}]
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div style={{ color: '#64748b' }}>
                Select an edge or node after running a batch to inspect its intermediate activations.
              </div>
            )}
          </div>
        )}

        {activeDrawerTab === 'diagnostics' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {diagnostics.length === 0 ? (
              <div style={{ color: '#22c55e', display: 'flex', alignItems: 'center', gap: 6 }}>
                <CheckCircle2 size={14} />
                No validation errors. Architecture graph is structurally and numerically valid.
              </div>
            ) : (
              diagnostics.map((diag, idx) => (
                <div
                  key={idx}
                  style={{
                    background: diag.severity === 'error' ? 'rgba(239, 68, 68, 0.1)' : 'rgba(245, 158, 11, 0.1)',
                    border: `1px solid ${diag.severity === 'error' ? '#ef4444' : '#f59e0b'}`,
                    padding: '8px 12px',
                    borderRadius: 4,
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 3,
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ fontWeight: 600, color: diag.severity === 'error' ? '#f87171' : '#fbbf24' }}>
                      [{diag.code}]
                    </span>
                    <span>{diag.message}</span>
                  </div>
                  {diag.suggestions?.length > 0 && (
                    <div style={{ fontSize: 11, color: '#93c5fd', marginLeft: 16 }}>
                      Suggestion: {diag.suggestions[0]}
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        )}

        {activeDrawerTab === 'parameters' && (
          <div style={{ display: 'flex', gap: 32 }}>
            <div>
              <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 4 }}>Model Totals:</div>
              <div style={{ fontSize: 13, fontWeight: 600 }}>
                Unique Parameters: <span style={{ color: '#38bdf8' }}>{parameterSummary.total_unique.toLocaleString()}</span>
              </div>
              <div style={{ fontSize: 12, color: '#22c55e' }}>
                Trainable: {parameterSummary.trainable.toLocaleString()}
              </div>
              <div style={{ fontSize: 12, color: '#94a3b8' }}>
                Frozen: {parameterSummary.frozen.toLocaleString()}
              </div>
              <div style={{ fontSize: 12, color: '#f59e0b' }}>
                Shared References: {parameterSummary.shared_references} (Tied Weights)
              </div>
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 4 }}>Per-Node Breakdown:</div>
              <div style={{ maxHeight: 120, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 4, fontFamily: 'monospace', fontSize: 11 }}>
                {Object.entries(parameterSummary.breakdown_by_node || {}).map(([nodePath, info]) => {
                  let totalStr = '0';
                  if (info && typeof info === 'object' && 'total' in info) {
                    totalStr = String(info.total);
                  }
                  return (
                    <div key={nodePath} style={{ display: 'flex', justifyContent: 'space-between', background: '#181b24', padding: '3px 8px', borderRadius: 3 }}>
                      <span>{nodePath}</span>
                      <span style={{ color: '#38bdf8' }}>{totalStr} params</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {activeDrawerTab === 'parity' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {isNanoGPTBaseline ? (
              <>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#38bdf8', fontWeight: 600 }}>
                  <CheckCircle2 size={16} color="#22c55e" />
                  Bundled nanoGPT baseline passed the pinned reference parity suite
                </div>
                <div style={{ color: '#94a3b8', fontSize: 11 }}>
                  This evidence applies to the bundled nanoGPT baseline, not arbitrary edited or imported projects.
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8, fontSize: 11 }}>
                  {[
                    'State-dict mapping completeness',
                    'Parameter-count equality',
                    'Forward logits and loss tolerances',
                    'Intermediate activation tolerances',
                    'Autograd gradient tolerances',
                    'One-step AdamW update tolerances',
                    'Inference output contract',
                  ].map((item) => (
                    <div key={item} style={{ background: '#161922', padding: '6px 8px', borderRadius: 4, border: '1px solid #232838', color: '#94a3b8' }}>
                      ✓ {item}
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div style={{ color: '#f59e0b' }}>
                No reference parity suite is attached to this project. Run Architecture Tester for project-specific checks.
              </div>
            )}
          </div>
        )}

        {activeDrawerTab === 'logs' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4, fontFamily: 'monospace', fontSize: 11 }}>
            {logs.length === 0 ? (
              <div style={{ color: '#64748b' }}>No execution logs captured yet.</div>
            ) : (
              logs.map((log) => (
                <div key={log.id} style={{ display: 'flex', gap: 8 }}>
                  <span style={{ color: '#64748b' }}>[{log.timestamp}]</span>
                  <span
                    style={{
                      color: log.level === 'error' ? '#ef4444' : log.level === 'warn' ? '#f59e0b' : '#38bdf8',
                      fontWeight: 600,
                    }}
                  >
                    [{log.level.toUpperCase()}]
                  </span>
                  <span>{log.message}</span>
                </div>
              ))
            )}
          </div>
        )}

        {activeDrawerTab === 'tester' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <button
                  onClick={async () => {
                    if (project) {
                      setIsRunningSuite(true);
                      setTestSuiteError(null);
                      try {
                        const res = await ApiClient.runTestSuite(project);
                        setTestSuiteResult(res);
                      } catch (err) {
                        const message = err instanceof Error ? err.message : String(err);
                        setTestSuiteError(message);
                        addLog('error', `Test suite failed: ${message}`);
                      } finally {
                        setIsRunningSuite(false);
                      }
                    }
                  }}
                  disabled={isRunningSuite}
                  style={{
                    background: isRunningSuite ? '#334155' : '#0284c7',
                    border: 'none',
                    color: '#ffffff',
                    borderRadius: 4,
                    padding: '5px 12px',
                    fontSize: 11,
                    fontWeight: 600,
                    cursor: isRunningSuite ? 'not-allowed' : 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6,
                  }}
                >
                  <ShieldCheck size={13} />
                  {isRunningSuite ? 'Running Tests...' : 'Run Test Suite (6 Tests)'}
                </button>

                {testSuiteResult && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11 }}>
                    <span style={{ color: testSuiteResult.failed === 0 ? '#22c55e' : '#ef4444', fontWeight: 600 }}>
                      {testSuiteResult.passed}/{testSuiteResult.total} Passed
                    </span>
                    <span style={{ color: '#64748b' }}>({testSuiteResult.duration_ms.toFixed(0)} ms)</span>
                  </div>
                )}
              </div>
            </div>

            {testSuiteError && (
              <div role="alert" style={{ color: '#ef4444', fontSize: 11 }}>
                {testSuiteError}
              </div>
            )}

            {/* Test Cards List */}
            {testSuiteResult ? (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
                {testSuiteResult.cases.map((tc) => {
                  const isPassed = tc.status === 'passed';
                  return (
                    <div
                      key={tc.id}
                      style={{
                        background: '#161922',
                        border: `1px solid ${isPassed ? '#15803d' : '#991b1b'}`,
                        borderRadius: 4,
                        padding: '8px 10px',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 3,
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          <span
                            style={{
                              fontSize: 10,
                              fontWeight: 700,
                              padding: '1px 5px',
                              borderRadius: 3,
                              background: isPassed ? 'rgba(34, 197, 94, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                              color: isPassed ? '#22c55e' : '#ef4444',
                            }}
                          >
                            {tc.status.toUpperCase()}
                          </span>
                          <span style={{ fontWeight: 600, fontSize: 11 }}>{tc.name}</span>
                        </div>
                        <span style={{ fontSize: 10, color: '#64748b' }}>{tc.duration_ms.toFixed(1)} ms</span>
                      </div>
                      <div style={{ fontSize: 10, color: '#94a3b8' }}>{tc.description}</div>
                      {tc.error && (
                        <div style={{ fontSize: 10, color: '#fca5a5', background: 'rgba(239, 68, 68, 0.1)', padding: 4, borderRadius: 2 }}>
                          Error: {tc.error}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            ) : (
              <div style={{ color: '#64748b', fontSize: 11 }}>
                Click "Run Test Suite" to execute the 6-pillar validation battery (Dynamic Shapes, Autograd Health, Single-Batch Overfit, Checkpointing, Standalone Cooking, and Numerical Stability).
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
