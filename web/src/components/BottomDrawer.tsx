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
  Code2,
  Cpu,
  Layers,
  LineChart,
  MessageSquare,
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
  const { project, updateTrainingConfig, updateModelConfig } = useProjectStore();
  const isNanoGPTBaseline =
    project.project.id === 'nanogpt_default' &&
    project.model.config['attention_implementation'] !== undefined;
  const { diagnostics, parameterSummary } = useValidationStore();
  const {
    logs,
    retainedSummaries,
    trainingMetrics,
    lossHistory,
    valLossHistory,
    parameterNorms,
    isTraining,
    startTraining,
    pauseTraining,
    stepBatch,
    stop,
    addLog,
    updateHyperparameters,
  } = useTraceStore();

  const [valFraction, setValFraction] = React.useState<number>(0.1);

  const [testSuiteResult, setTestSuiteResult] = React.useState<TestSuiteResult | null>(null);
  const [testSuiteError, setTestSuiteError] = React.useState<string | null>(null);
  const [isRunningSuite, setIsRunningSuite] = React.useState<boolean>(false);
  const [promptText, setPromptText] = React.useState<string>('Hello');
  const [promptTemplate, setPromptTemplate] = React.useState<'raw' | 'chatml' | 'alpaca' | 'llama3'>('raw');
  const [useKVCache, setUseKVCache] = React.useState<boolean>(true);
  const [maxNewTokens, setMaxNewTokens] = React.useState<number>(32);
  const [temperature, setTemperature] = React.useState<number>(1.0);
  const [topK, setTopK] = React.useState<number>(0);
  const [topP, setTopP] = React.useState<number>(1);
  const [isGenerating, setIsGenerating] = React.useState<boolean>(false);
  const generatedText = useTraceStore((s) => s.generatedText);
  const clearGenerated = useTraceStore((s) => s.clearGenerated);
  const [cookedCode, setCookedCode] = React.useState<string>('');
  const [cookError, setCookError] = React.useState<string | null>(null);
  const [isCooking, setIsCooking] = React.useState<boolean>(false);
  const [selectedDataset, setSelectedDataset] = React.useState<'synthetic' | 'tiny_shakespeare'>('synthetic');
  const [availableCheckpoints, setAvailableCheckpoints] = React.useState<Array<{ name: string; path: string }>>([]);
  const [learningRate, setLearningRate] = React.useState<number>(6e-4);
  const [weightDecay, setWeightDecay] = React.useState<number>(0.1);
  const [gradClip, setGradClip] = React.useState<number>(1.0);
  const [valLoss, setValLoss] = React.useState<number | null>(null);
  const [isValidatingSession, setIsValidatingSession] = React.useState<boolean>(false);
  const datasetFileInputRef = React.useRef<HTMLInputElement>(null);

  const refreshCheckpoints = async () => {
    const sId = useTraceStore.getState().sessionId;
    if (sId) {
      try {
        const res = await ApiClient.listCheckpoints(sId);
        setAvailableCheckpoints(res.checkpoints || []);
      } catch {
        // ignore
      }
    }
  };

  const handleUploadDataset = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const text = await file.text();
    try {
      let currentSessionId = useTraceStore.getState().sessionId;
      if (!currentSessionId) {
        const comp = await ApiClient.compileModel(project, 'cpu', 'training');
        currentSessionId = comp.session_id;
        useTraceStore.setState({ sessionId: currentSessionId, graphHash: comp.graph_hash });
        const newWs = ApiClient.connectTraceWebSocket(currentSessionId, (e) => {
          useTraceStore.getState().handleTraceEvent(e);
        });
        useTraceStore.setState({ ws: newWs });
      }
      const res = await ApiClient.uploadDataset(currentSessionId, text);
      addLog('info', `Custom dataset '${file.name}' uploaded (${res.num_samples} samples).`);
    } catch (err) {
      addLog('error', `Dataset upload failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const handleApplyHyperparameters = async () => {
    const sId = useTraceStore.getState().sessionId;
    if (!sId) {
      addLog('warn', 'Cannot update hyperparameters without an active session.');
      return;
    }
    try {
      await ApiClient.updateHyperparameters(sId, {
        learning_rate: learningRate,
        weight_decay: weightDecay,
        grad_clip: gradClip,
      });
      addLog('info', `Updated hyperparameters: lr=${learningRate}, wd=${weightDecay}, clip=${gradClip}`);
    } catch (err) {
      addLog('error', `Hyperparameter update failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const handleRunValidation = async () => {
    const sId = useTraceStore.getState().sessionId;
    if (!sId) {
      addLog('warn', 'Cannot run validation without an active session.');
      return;
    }
    setIsValidatingSession(true);
    try {
      const res = await ApiClient.runValidation(sId);
      setValLoss(res.val_loss);
      addLog('info', `Validation completed: val_loss=${res.val_loss.toFixed(4)} on ${res.val_samples} samples.`);
    } catch (err) {
      addLog('error', `Validation failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setIsValidatingSession(false);
    }
  };

  const [checkpointPath, setCheckpointPath] = React.useState<string>('checkpoints/manual.pt');

  const handleDatasetChange = async (name: 'synthetic' | 'tiny_shakespeare', fraction: number = valFraction) => {
    setSelectedDataset(name);
    try {
      let currentSessionId = useTraceStore.getState().sessionId;
      if (!currentSessionId) {
        const comp = await ApiClient.compileModel(project, 'cpu', 'training');
        currentSessionId = comp.session_id;
        useTraceStore.setState({
          sessionId: currentSessionId,
          graphHash: comp.graph_hash,
        });
        const newWs = ApiClient.connectTraceWebSocket(currentSessionId, (event) => {
          useTraceStore.getState().handleTraceEvent(event);
        });
        useTraceStore.setState({ ws: newWs });
      }
      const res = await ApiClient.setDataset(currentSessionId, name, fraction);
      addLog('info', `Dataset switched to '${res.name}' (${res.num_samples} samples).`);
    } catch (err) {
      addLog('error', `Failed to set dataset: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const handleSaveCheckpoint = async () => {
    try {
      let currentSessionId = useTraceStore.getState().sessionId;
      if (!currentSessionId) {
        addLog('warn', 'Cannot save checkpoint without an active session.');
        return;
      }
      const res = await ApiClient.saveCheckpoint(currentSessionId);
      addLog('info', `Checkpoint saved: ${res.path}`);
      refreshCheckpoints();
    } catch (err) {
      addLog('error', `Save checkpoint failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const handleLoadCheckpoint = async () => {
    try {
      let currentSessionId = useTraceStore.getState().sessionId;
      if (!currentSessionId) {
        addLog('warn', 'Cannot load checkpoint without an active session.');
        return;
      }
      const res = await ApiClient.loadCheckpoint(currentSessionId, checkpointPath);
      addLog('info', `Checkpoint loaded: step ${res.step}, epoch ${res.epoch}`);
    } catch (err) {
      addLog('error', `Load checkpoint failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  };


  const handleCookExport = async () => {
    setIsCooking(true);
    setCookError(null);
    try {
      const res = await ApiClient.cookExport(project);
      setCookedCode(res.code);
      setCookError(null);
      addLog('info', 'PyTorch code exported successfully.');
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setCookError(msg);
      setCookedCode('');
      addLog('error', msg);
    } finally {
      setIsCooking(false);
    }
  };


  const handleGenerate = async () => {
    clearGenerated();
    setIsGenerating(true);
    try {
      let currentSessionId = useTraceStore.getState().sessionId;
      if (!currentSessionId) {
        const comp = await ApiClient.compileModel(project, 'cpu', 'training');
        currentSessionId = comp.session_id;
        useTraceStore.setState({
          sessionId: currentSessionId,
          graphHash: comp.graph_hash,
        });
        const newWs = ApiClient.connectTraceWebSocket(currentSessionId, (event) => {
          useTraceStore.getState().handleTraceEvent(event);
        });
        useTraceStore.setState({ ws: newWs });
      }
      await ApiClient.generate(currentSessionId, {
        prompt: promptText,
        max_new_tokens: maxNewTokens,
        temperature,
        top_k: topK,
        top_p: topP,
        template: promptTemplate,
        use_cache: useKVCache,
        stream: true,
      });
    } catch (err) {
      addLog('error', `Generation failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setIsGenerating(false);
    }
  };
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
            onClick={() => openDrawerTab('playground')}
            style={{
              background: 'transparent',
              border: 'none',
              color: '#38bdf8',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 4,
              fontSize: 11,
            }}
          >
            <MessageSquare size={12} />
            Playground
          </button>
          <button
            onClick={() => openDrawerTab('code')}
            style={{
              background: 'transparent',
              border: 'none',
              color: '#38bdf8',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 4,
              fontSize: 11,
            }}
          >
            <Code2 size={12} />
            PyTorch Code
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
    const minStep = lossHistory[0].step;
    const maxStep = Math.max(lossHistory[lossHistory.length - 1].step, minStep + 1);

    const allLosses = [
      ...lossHistory.map((p) => p.loss),
      ...valLossHistory.map((p) => p.val_loss),
    ];
    const maxLoss = Math.max(...allLosses, 0.1);
    const minLoss = Math.min(...allLosses, 0.0);

    const trainPoints = lossHistory.map((p) => {
      const x = padding + ((p.step - minStep) / (maxStep - minStep)) * (width - 2 * padding);
      const y = height - padding - ((p.loss - minLoss) / Math.max(maxLoss - minLoss, 1e-4)) * (height - 2 * padding);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });
    const trainPath = `M ${trainPoints.join(' L ')}`;

    let valPath: string | null = null;
    if (valLossHistory.length >= 2) {
      const valPoints = valLossHistory.map((p) => {
        const x = padding + Math.max(0, Math.min(1, (p.step - minStep) / (maxStep - minStep))) * (width - 2 * padding);
        const y = height - padding - ((p.val_loss - minLoss) / Math.max(maxLoss - minLoss, 1e-4)) * (height - 2 * padding);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      });
      valPath = `M ${valPoints.join(' L ')}`;
    }

    return (
      <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
        <svg width={width} height={height} style={{ background: '#0a0c12', borderRadius: 6, border: '1px solid #1f2430' }}>
          <path d={trainPath} fill="none" stroke="#38bdf8" strokeWidth="2" />
          {valPath && <path d={valPath} fill="none" stroke="#f59e0b" strokeWidth="2" />}
          {lossHistory.map((p, idx) => {
            if (idx % Math.ceil(lossHistory.length / 10) === 0 || idx === lossHistory.length - 1) {
              const x = padding + ((p.step - minStep) / (maxStep - minStep)) * (width - 2 * padding);
              const y = height - padding - ((p.loss - minLoss) / Math.max(maxLoss - minLoss, 1e-4)) * (height - 2 * padding);
              return <circle key={idx} cx={x} cy={y} r="3" fill="#38bdf8" />;
            }
            return null;
          })}
          {valPath &&
            valLossHistory.map((p, idx) => {
              const x = padding + Math.max(0, Math.min(1, (p.step - minStep) / (maxStep - minStep))) * (width - 2 * padding);
              const y = height - padding - ((p.val_loss - minLoss) / Math.max(maxLoss - minLoss, 1e-4)) * (height - 2 * padding);
              return <circle key={`val_${idx}`} cx={x} cy={y} r="3" fill="#f59e0b" />;
            })}
        </svg>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 11 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ color: '#94a3b8' }}>Legend:</span>
            <span style={{ color: '#38bdf8', display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ width: 10, height: 2, background: '#38bdf8', display: 'inline-block' }} /> Train
            </span>
            <span style={{ color: '#f59e0b', display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ width: 10, height: 2, background: '#f59e0b', display: 'inline-block' }} /> Val
            </span>
          </div>
          <div style={{ color: '#94a3b8' }}>Live Loss Stats:</div>
          <div>Current Loss: <span style={{ color: '#38bdf8', fontWeight: 600 }}>{lossHistory[lossHistory.length - 1].loss.toFixed(5)}</span></div>
          {valLossHistory.length > 0 && (
            <div>Val Loss: <span style={{ color: '#f59e0b', fontWeight: 600 }}>{valLossHistory[valLossHistory.length - 1].val_loss.toFixed(5)}</span></div>
          )}
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
              { id: 'playground', label: 'Playground', icon: MessageSquare },
              { id: 'code', label: 'PyTorch Code', icon: Code2 },
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
      <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: 14, display: 'flex', flexDirection: 'column' }}>
        {activeDrawerTab === 'loss' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                flexWrap: 'wrap',
                gap: 10,
                paddingBottom: 8,
                borderBottom: '1px solid #1f2430',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: 11, color: '#94a3b8' }}>Dataset:</span>
                <select
                  aria-label="Training dataset"
                  value={selectedDataset}
                  onChange={(e) =>
                    handleDatasetChange(e.target.value as 'synthetic' | 'tiny_shakespeare')
                  }
                  style={{
                    background: '#181b24',
                    border: '1px solid #272c3b',
                    color: '#e2e8f0',
                    padding: '4px 8px',
                    borderRadius: 4,
                    fontSize: 11,
                    outline: 'none',
                  }}
                >
                  <option value="synthetic">Synthetic</option>
                  <option value="tiny_shakespeare">Tiny Shakespeare</option>
                </select>
                <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, color: '#94a3b8' }}>
                  Val Split:
                  <input
                    type="number"
                    aria-label="Validation fraction"
                    min="0.05"
                    max="0.5"
                    step="0.05"
                    value={valFraction}
                    onChange={(e) => {
                      const v = parseFloat(e.target.value) || 0.1;
                      setValFraction(v);
                      handleDatasetChange(selectedDataset, v);
                    }}
                    style={{
                      width: 55,
                      background: '#181b24',
                      border: '1px solid #272c3b',
                      color: '#e2e8f0',
                      padding: '2px 6px',
                      borderRadius: 4,
                      fontSize: 11,
                    }}
                  />
                </label>
                <input
                  ref={datasetFileInputRef}
                  type="file"
                  accept=".txt,.csv"
                  onChange={handleUploadDataset}
                  style={{ display: 'none' }}
                />
                <button
                  type="button"
                  onClick={() => datasetFileInputRef.current?.click()}
                  style={{
                    background: '#181b24',
                    border: '1px solid #272c3b',
                    color: '#cbd5e1',
                    padding: '4px 8px',
                    borderRadius: 4,
                    fontSize: 11,
                    cursor: 'pointer',
                  }}
                >
                  Upload .txt
                </button>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <button
                  onClick={handleSaveCheckpoint}
                  style={{
                    background: '#181b24',
                    border: '1px solid #272c3b',
                    color: '#cbd5e1',
                    padding: '4px 8px',
                    borderRadius: 4,
                    fontSize: 11,
                    cursor: 'pointer',
                  }}
                >
                  Save checkpoint
                </button>
                {availableCheckpoints.length > 0 && (
                  <select
                    aria-label="Select saved checkpoint"
                    value={checkpointPath}
                    onChange={(e) => setCheckpointPath(e.target.value)}
                    style={{
                      background: '#0a0c12',
                      border: '1px solid #272c3b',
                      color: '#e2e8f0',
                      padding: '4px 8px',
                      borderRadius: 4,
                      fontSize: 11,
                    }}
                  >
                    {availableCheckpoints.map((c) => (
                      <option key={c.path} value={c.path}>
                        {c.name}
                      </option>
                    ))}
                  </select>
                )}
                <input
                  type="text"
                  value={checkpointPath}
                  onChange={(e) => setCheckpointPath(e.target.value)}
                  style={{
                    background: '#0a0c12',
                    border: '1px solid #272c3b',
                    color: '#e2e8f0',
                    padding: '4px 8px',
                    borderRadius: 4,
                    fontSize: 11,
                    width: 170,
                  }}
                />
                <button
                  onClick={handleLoadCheckpoint}
                  style={{
                    background: '#181b24',
                    border: '1px solid #272c3b',
                    color: '#cbd5e1',
                    padding: '4px 8px',
                    borderRadius: 4,
                    fontSize: 11,
                    cursor: 'pointer',
                  }}
                >
                  Load checkpoint
                </button>
              </div>
            </div>

            {/* Live Hyperparameters & Validation Controls */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                flexWrap: 'wrap',
                gap: 12,
                paddingBottom: 8,
                borderBottom: '1px solid #1f2430',
                fontSize: 11,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span style={{ color: '#94a3b8' }}>Live Tuning:</span>
                <label style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  Batch:
                  <input
                    type="number"
                    aria-label="Batch size"
                    min="1"
                    step="1"
                    value={project.model.training?.batch_size ?? 8}
                    onChange={(e) => {
                      const val = Math.max(1, parseInt(e.target.value, 10) || 1);
                      updateTrainingConfig('batch_size', val);
                      const sId = useTraceStore.getState().sessionId;
                      if (sId) {
                        updateHyperparameters({ batch_size: val });
                      }
                    }}
                    style={{
                      width: 55,
                      background: '#0a0c12',
                      border: '1px solid #272c3b',
                      color: '#e2e8f0',
                      padding: '2px 4px',
                      borderRadius: 4,
                      fontSize: 11,
                    }}
                  />
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  Seq:
                  <input
                    type="number"
                    aria-label="Sequence length"
                    min="1"
                    step="1"
                    value={typeof project.model.config.block_size === 'number' ? project.model.config.block_size : 8}
                    onChange={(e) => {
                      const val = Math.max(1, parseInt(e.target.value, 10) || 1);
                      updateModelConfig('block_size', val);
                      addLog('warn', 'Sequence length change requires recompile.');
                    }}
                    style={{
                      width: 55,
                      background: '#0a0c12',
                      border: '1px solid #272c3b',
                      color: '#e2e8f0',
                      padding: '2px 4px',
                      borderRadius: 4,
                      fontSize: 11,
                    }}
                  />
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  LR:
                  <input
                    type="number"
                    step={0.0001}
                    value={learningRate}
                    onChange={(e) => setLearningRate(Number(e.target.value))}
                    style={{
                      width: 70,
                      background: '#0a0c12',
                      border: '1px solid #272c3b',
                      color: '#e2e8f0',
                      padding: '2px 4px',
                      borderRadius: 4,
                      fontSize: 11,
                    }}
                  />
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  WD:
                  <input
                    type="number"
                    step={0.01}
                    value={weightDecay}
                    onChange={(e) => setWeightDecay(Number(e.target.value))}
                    style={{
                      width: 60,
                      background: '#0a0c12',
                      border: '1px solid #272c3b',
                      color: '#e2e8f0',
                      padding: '2px 4px',
                      borderRadius: 4,
                      fontSize: 11,
                    }}
                  />
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  Clip:
                  <input
                    type="number"
                    step={0.1}
                    value={gradClip}
                    onChange={(e) => setGradClip(Number(e.target.value))}
                    style={{
                      width: 50,
                      background: '#0a0c12',
                      border: '1px solid #272c3b',
                      color: '#e2e8f0',
                      padding: '2px 4px',
                      borderRadius: 4,
                      fontSize: 11,
                    }}
                  />
                </label>
                <button
                  type="button"
                  onClick={handleApplyHyperparameters}
                  style={{
                    background: '#1e293b',
                    border: '1px solid #334155',
                    color: '#38bdf8',
                    padding: '2px 8px',
                    borderRadius: 4,
                    fontSize: 11,
                    cursor: 'pointer',
                  }}
                >
                  Apply
                </button>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <button
                  type="button"
                  onClick={handleRunValidation}
                  disabled={isValidatingSession}
                  style={{
                    background: '#1e293b',
                    border: '1px solid #334155',
                    color: '#a855f7',
                    padding: '3px 10px',
                    borderRadius: 4,
                    fontSize: 11,
                    cursor: isValidatingSession ? 'default' : 'pointer',
                    fontWeight: 600,
                  }}
                >
                  {isValidatingSession ? 'Validating...' : 'Run Validation'}
                </button>
                {valLoss !== null && (
                  <span style={{ color: '#22c55e', fontWeight: 600 }}>
                    Val Loss: {valLoss.toFixed(4)}
                  </span>
                )}
              </div>
            </div>

            {renderLossPlot()}
          </div>
        )}

        {activeDrawerTab === 'metrics' && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 12 }}>
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

            <div style={{ background: '#181b24', padding: '10px 14px', borderRadius: 6, border: '1px solid #272c3b' }}>
              <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 4 }}>Param L2</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: '#f59e0b' }}>
                {Object.keys(parameterNorms).length > 0
                  ? Object.values(parameterNorms).reduce((a, b) => a + b, 0).toFixed(3)
                  : '—'}
              </div>
              <div style={{ fontSize: 10, color: '#64748b' }}>
                {Object.keys(parameterNorms).length > 0 ? `${Object.keys(parameterNorms).length} params` : 'N/A'}
              </div>
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
          <div style={{ flex: 1, minHeight: 0, display: 'flex', gap: 32, overflow: 'hidden' }}>
            <div style={{ flexShrink: 0 }}>
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
            <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
              <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 6, flexShrink: 0 }}>Per-Node Breakdown:</div>
              <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 4, fontFamily: 'monospace', fontSize: 11, paddingRight: 4 }}>
                {Object.entries(parameterSummary.breakdown_by_node || {}).map(([nodePath, info]) => {
                  let totalStr = '0';
                  if (info && typeof info === 'object' && 'total' in info) {
                    totalStr = String(info.total);
                  }
                  return (
                    <div key={nodePath} style={{ display: 'flex', justifyContent: 'space-between', background: '#181b24', padding: '5px 10px', borderRadius: 4, border: '1px solid #232838' }}>
                      <span>{nodePath}</span>
                      <span style={{ color: '#38bdf8', fontWeight: 600 }}>{totalStr} params</span>
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
        {activeDrawerTab === 'playground' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 6 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <label style={{ fontSize: 11, color: '#94a3b8' }}>Prompt</label>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <span style={{ fontSize: 11, color: '#94a3b8' }}>Format:</span>
                    <select
                      aria-label="Prompt template"
                      value={promptTemplate}
                      onChange={(e) =>
                        setPromptTemplate(e.target.value as 'raw' | 'chatml' | 'alpaca' | 'llama3')
                      }
                      style={{
                        background: '#0a0c12',
                        border: '1px solid #1f2430',
                        borderRadius: 4,
                        color: '#e2e8f0',
                        padding: '2px 6px',
                        fontSize: 11,
                      }}
                    >
                      <option value="raw">Raw Text</option>
                      <option value="chatml">ChatML</option>
                      <option value="alpaca">Alpaca / Instruct</option>
                      <option value="llama3">Llama-3</option>
                    </select>
                    <label
                      style={{
                        fontSize: 11,
                        color: '#94a3b8',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 4,
                        cursor: 'pointer',
                      }}
                    >
                      <input
                        type="checkbox"
                        checked={useKVCache}
                        onChange={(e) => setUseKVCache(e.target.checked)}
                      />
                      KV Cache
                    </label>
                  </div>
                </div>
                <textarea
                  value={promptText}
                  onChange={(e) => setPromptText(e.target.value)}
                  placeholder="Enter prompt..."
                  rows={3}
                  style={{
                    width: '100%',
                    background: '#0a0c12',
                    border: '1px solid #1f2430',
                    borderRadius: 4,
                    color: '#e2e8f0',
                    padding: '6px 8px',
                    fontSize: 12,
                    fontFamily: 'monospace',
                    resize: 'vertical',
                    boxSizing: 'border-box',
                  }}
                />
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 6, minWidth: 140 }}>
                <label style={{ fontSize: 11, color: '#94a3b8' }}>Max New Tokens</label>
                <input
                  type="number"
                  min={1}
                  max={256}
                  value={maxNewTokens}
                  onChange={(e) => setMaxNewTokens(Number(e.target.value))}
                  style={{
                    background: '#0a0c12',
                    border: '1px solid #1f2430',
                    borderRadius: 4,
                    color: '#e2e8f0',
                    padding: '4px 8px',
                    fontSize: 11,
                  }}
                />

                <label style={{ fontSize: 11, color: '#94a3b8' }}>Temperature</label>
                <input
                  type="number"
                  step={0.1}
                  min={0}
                  max={2}
                  value={temperature}
                  onChange={(e) => setTemperature(Number(e.target.value))}
                  style={{
                    background: '#0a0c12',
                    border: '1px solid #1f2430',
                    borderRadius: 4,
                    color: '#e2e8f0',
                    padding: '4px 8px',
                    fontSize: 11,
                  }}
                />

                <label style={{ fontSize: 11, color: '#94a3b8' }}>Top-k</label>
                <input
                  type="number"
                  min={0}
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  aria-label="Top-k"
                  style={{
                    background: '#0a0c12',
                    border: '1px solid #1f2430',
                    borderRadius: 4,
                    color: '#e2e8f0',
                    padding: '4px 8px',
                    fontSize: 11,
                  }}
                />

                <label style={{ fontSize: 11, color: '#94a3b8' }}>Top-p</label>
                <input
                  type="number"
                  step={0.05}
                  min={0}
                  max={1}
                  value={topP}
                  onChange={(e) => setTopP(Number(e.target.value))}
                  aria-label="Top-p"
                  style={{
                    background: '#0a0c12',
                    border: '1px solid #1f2430',
                    borderRadius: 4,
                    color: '#e2e8f0',
                    padding: '4px 8px',
                    fontSize: 11,
                  }}
                />

                <button
                  onClick={handleGenerate}
                  disabled={isGenerating}
                  style={{
                    marginTop: 6,
                    background: isGenerating ? '#1e293b' : '#0284c7',
                    border: 'none',
                    borderRadius: 4,
                    color: '#ffffff',
                    padding: '6px 12px',
                    fontSize: 11,
                    fontWeight: 600,
                    cursor: isGenerating ? 'default' : 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: 6,
                  }}
                >
                  <Play size={12} />
                  {isGenerating ? 'Generating...' : 'Generate'}
                </button>
              </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 11, color: '#94a3b8' }}>Generated Output:</span>
                {generatedText && (
                  <button
                    onClick={clearGenerated}
                    style={{
                      background: 'transparent',
                      border: 'none',
                      color: '#64748b',
                      fontSize: 10,
                      cursor: 'pointer',
                    }}
                  >
                    Clear
                  </button>
                )}
              </div>
              <pre
                style={{
                  background: '#0a0c12',
                  border: '1px solid #1f2430',
                  borderRadius: 4,
                  padding: 10,
                  fontSize: 12,
                  fontFamily: 'monospace',
                  color: '#38bdf8',
                  minHeight: 60,
                  maxHeight: 200,
                  overflowY: 'auto',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                  margin: 0,
                }}
              >
                {generatedText || <span style={{ color: '#475569' }}>Generated text will stream here...</span>}
              </pre>
            </div>
          </div>
        )}
        {activeDrawerTab === 'code' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: 11, color: '#94a3b8' }}>
                Export standalone, zero-dependency PyTorch training script from active blueprint.
              </span>
              <button
                onClick={handleCookExport}
                disabled={isCooking}
                style={{
                  background: isCooking ? '#1e293b' : '#0284c7',
                  border: 'none',
                  borderRadius: 4,
                  color: '#ffffff',
                  padding: '5px 12px',
                  fontSize: 11,
                  fontWeight: 600,
                  cursor: isCooking ? 'default' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                }}
              >
                <Code2 size={12} />
                {isCooking ? 'Exporting...' : 'Export PyTorch Code'}
              </button>
            </div>

            {cookError ? (
              <div
                role="alert"
                style={{
                  background: '#1a0f0f',
                  border: '1px solid #7f1d1d',
                  borderRadius: 4,
                  padding: 12,
                  fontSize: 11,
                  color: '#fca5a5',
                }}
              >
                {cookError}
              </div>
            ) : cookedCode ? (
              <pre
                style={{
                  background: '#0a0c12',
                  border: '1px solid #1f2430',
                  borderRadius: 4,
                  padding: 12,
                  fontFamily: 'monospace',
                  fontSize: 11,
                  color: '#e2e8f0',
                  maxHeight: 400,
                  overflowY: 'auto',
                  whiteSpace: 'pre',
                  margin: 0,
                }}
              >
                {cookedCode}
              </pre>
            ) : (
              <div style={{ color: '#64748b', fontSize: 11, padding: 20, textAlign: 'center' }}>
                Click "Export PyTorch Code" to compile visual blueprint into standalone train.py script.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
