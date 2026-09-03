/**
 * Live execution tracing, stepping, training telemetry, and debugging Zustand store.
 */

import { create } from 'zustand';
import { ApiClient } from '../api/client';
import { Project, TensorSummary, TraceEvent, TrainingMetrics } from '../api/contracts';
import { useProjectStore } from './projectStore';
import { useUIStore } from './uiStore';
export type ExecutionState = 'idle' | 'running' | 'paused' | 'completed' | 'error';
export type NodeRunState = 'pending' | 'running' | 'completed' | 'paused' | 'failed';

export interface LossPoint {
  step: number;
  loss: number;
  lr: number;
  grad_norm: number;
  tokens_per_sec: number;
}

interface LogEntry {
  id: string;
  timestamp: string;
  level: 'info' | 'warn' | 'error';
  message: string;
}

interface TraceStoreState {
  sessionId: string | null;
  graphHash: string | null;
  status: ExecutionState;
  nodeStates: Record<string, NodeRunState>;
  retainedSummaries: Record<string, TensorSummary>;
  activeNodePath: string | null;
  logs: LogEntry[];
  ws: WebSocket | null;
  generatedText: string;

  // Training Telemetry
  trainingMetrics: TrainingMetrics | null;
  lossHistory: LossPoint[];
  valLossHistory: Array<{ step: number; val_loss: number }>;
  nodeGradientNorms: Record<string, number>;
  parameterNorms: Record<string, number>;
  isTraining: boolean;

  // Actions
  compileOnly: (project: Project) => Promise<void>;
  compileAndRun: (project: Project, speed?: string) => Promise<void>;
  startTraining: (project: Project, maxSteps?: number, speedDelay?: number) => Promise<void>;
  pauseTraining: () => Promise<void>;
  resumeTraining: (speedDelay?: number) => Promise<void>;
  stepBatch: (project?: Project) => Promise<void>;
  stepEpoch: () => Promise<void>;
  updateHyperparameters: (params: { learning_rate?: number; weight_decay?: number; grad_clip?: number; batch_size?: number }) => Promise<void>;
  step: () => Promise<void>;
  continueRun: () => Promise<void>;
  stop: () => Promise<void>;
  handleTraceEvent: (event: TraceEvent) => void;
  addLog: (level: 'info' | 'warn' | 'error', message: string) => void;
  saveCheckpoint: (path?: string) => Promise<void>;
  loadCheckpoint: (path: string) => Promise<void>;
  clearGenerated: () => void;
  clearTrace: () => void;
}

export const useTraceStore = create<TraceStoreState>((set, get) => ({
  sessionId: null,
  graphHash: null,
  status: 'idle',
  nodeStates: {},
  retainedSummaries: {},
  activeNodePath: null,
  logs: [],
  ws: null,
  generatedText: '',

  trainingMetrics: null,
  lossHistory: [],
  valLossHistory: [],
  nodeGradientNorms: {},
  parameterNorms: {},
  isTraining: false,

  addLog: (level: 'info' | 'warn' | 'error', message: string) => {
    const entry: LogEntry = {
      id: `log_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`,
      timestamp: new Date().toLocaleTimeString(),
      level,
      message,
    };
    set((state) => ({ logs: [...state.logs.slice(-200), entry] }));
  },
  clearGenerated: () => {
    set({ generatedText: '' });
  },
  saveCheckpoint: async (path?: string) => {
    const { sessionId, addLog } = get();
    if (!sessionId) return;
    try {
      const res = await ApiClient.saveCheckpoint(sessionId, path);
      addLog('info', `Checkpoint saved: ${res.path} (step ${res.step})`);
    } catch (err) {
      addLog('error', `Save checkpoint failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  },

  loadCheckpoint: async (path: string) => {
    const { sessionId, addLog } = get();
    if (!sessionId) return;
    try {
      const res = await ApiClient.loadCheckpoint(sessionId, path);
      addLog('info', `Checkpoint loaded from ${path} (step ${res.step})`);
    } catch (err) {
      addLog('error', `Load checkpoint failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  },



  clearTrace: () => {
    set({
      status: 'idle',
      nodeStates: {},
      retainedSummaries: {},
      activeNodePath: null,
      trainingMetrics: null,
      lossHistory: [],
      valLossHistory: [],
      nodeGradientNorms: {},
      parameterNorms: {},
      isTraining: false,
      generatedText: '',
    });
  },

  compileOnly: async (project: Project) => {
    const { ws, addLog } = get();
    if (ws) {
      ws.close();
    }
    try {
      const comp = await ApiClient.compileModel(project, 'cpu', 'inference');
      set({
        sessionId: comp.session_id,
        graphHash: comp.graph_hash,
      });
      addLog('info', 'Compiled.');
    } catch (err) {
      set({ status: 'error' });
      addLog('error', `Compilation failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  },

  compileAndRun: async (project: Project, speed: string = 'normal') => {
    const { ws, addLog } = get();
    if (ws) {
      ws.close();
    }

    set({
      status: 'running',
      nodeStates: {},
      retainedSummaries: {},
      activeNodePath: null,
      isTraining: false,
    });

    addLog('info', 'Compiling architecture graph into PyTorch CompiledGraphModule...');

    try {
      const comp = await ApiClient.compileModel(project, 'cpu', 'inference');
      const sessionId = comp.session_id;

      set({
        sessionId,
        graphHash: comp.graph_hash,
      });

      addLog('info', `Session compiled successfully [ID: ${sessionId.substring(0, 8)}]. Connecting trace stream...`);

      const newWs = ApiClient.connectTraceWebSocket(sessionId, (event) => {
        get().handleTraceEvent(event);
      });
      set({ ws: newWs });

      const sampleTokens = [1, 5, 12, 18, 3, 7, 22, 30];
      const sampleTargets = [5, 12, 18, 3, 7, 22, 30, 2];

      await ApiClient.runSession(
        sessionId,
        { token_ids: [sampleTokens], targets: [sampleTargets] },
        'inspection',
        speed
      );
      addLog('info', `Execution run started with speed '${speed}'.`);
    } catch (err) {
      set({ status: 'error' });
      addLog('error', `Compilation or run failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  },

  startTraining: async (project: Project, maxSteps: number = 100, speedDelay: number = 0.0) => {
    let { sessionId, ws, addLog } = get();
    const speed = useUIStore.getState().traceSpeed;
    const computedDelay =
      speedDelay > 0
        ? speedDelay
        : speed === 'instant'
        ? 0.005
        : speed === 'fast'
        ? 0.03
        : 0.08;

    if (!sessionId) {
      if (ws) {
        ws.close();
      }

      set({
        status: 'running',
        isTraining: true,
        nodeStates: {},
        retainedSummaries: {},
        lossHistory: [],
        valLossHistory: [],
        nodeGradientNorms: {},
        parameterNorms: {},
      });

      addLog('info', `Compiling and launching Blueprint Training Session (max_steps=${maxSteps})...`);

      try {
        const comp = await ApiClient.compileModel(project, 'cpu', 'training');
        sessionId = comp.session_id;

        set({
          sessionId,
          graphHash: comp.graph_hash,
        });

        const newWs = ApiClient.connectTraceWebSocket(sessionId, (event) => {
          get().handleTraceEvent(event);
        });
        set({ ws: newWs });
      } catch (err) {
        set({ status: 'error', isTraining: false });
        addLog('error', `Training launch failed: ${err instanceof Error ? err.message : String(err)}`);
        return;
      }
    } else {
      set({ status: 'running', isTraining: true });
      if (!ws) {
        const newWs = ApiClient.connectTraceWebSocket(sessionId, (event) => {
          get().handleTraceEvent(event);
        });
        set({ ws: newWs });
      }
    }

    try {
      await ApiClient.startTraining(sessionId, maxSteps, computedDelay);
      addLog('info', `Training loop started.`);
    } catch (err) {
      set({ status: 'error', isTraining: false });
      addLog('error', `Training launch failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  },
  pauseTraining: async () => {
    const { sessionId, addLog } = get();
    if (!sessionId) return;
    try {
      await ApiClient.pauseTraining(sessionId);
      set({ status: 'paused' });
      addLog('info', 'Training paused.');
    } catch (err) {
      addLog('error', `Pause failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  },

  resumeTraining: async () => {
    const { sessionId, addLog } = get();
    if (!sessionId) return;
    try {
      set({ status: 'running' });
      await ApiClient.resumeTraining(sessionId);
      addLog('info', 'Training resumed.');
    } catch (err) {
      addLog('error', `Resume failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  },
  stepBatch: async (projectArg?: Project) => {
    const project = projectArg || useProjectStore.getState().project;
    let { sessionId, ws } = get();
    if (!sessionId && project) {
      try {
        const comp = await ApiClient.compileModel(project, 'cpu', 'training');
        sessionId = comp.session_id;
        set({ sessionId, graphHash: comp.graph_hash, isTraining: true });
        if (!ws) {
          const newWs = ApiClient.connectTraceWebSocket(sessionId, (event) => {
            get().handleTraceEvent(event);
          });
          set({ ws: newWs });
        }
      } catch (err) {
        get().addLog('error', `Compilation failed: ${err instanceof Error ? err.message : String(err)}`);
        return;
      }
    }
    if (!sessionId) return;
    try {
      const res = await ApiClient.stepBatch(sessionId);
      if (res.metrics) {
        const { lossHistory } = get();
        const m = res.metrics;
        const newHistory = [
          ...lossHistory,
          {
            step: m.step,
            loss: m.loss,
            lr: m.learning_rate,
            grad_norm: m.grad_norm,
            tokens_per_sec: m.tokens_per_sec,
          },
        ];
        set({
          trainingMetrics: m,
          lossHistory: newHistory.slice(-500),
        });
      }
      if (res.event) {
        get().handleTraceEvent(res.event);
      }
      const curSessionId = get().sessionId;
      if (curSessionId) {
        ApiClient.getMetrics(curSessionId)
          .then((mRes) =>
            set({
              nodeGradientNorms: mRes.node_gradient_norms,
              parameterNorms: mRes.parameter_norms || {},
            })
          )
          .catch(() => {});
      }
    } catch (err) {
      get().addLog('error', `Batch step failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  },

  stepEpoch: async () => {
    const { sessionId, addLog } = get();
    if (!sessionId) return;
    try {
      const res = await ApiClient.stepEpoch(sessionId);
      addLog('info', `Epoch step finished: ${res.events_count} batches processed.`);
    } catch (err) {
      addLog('error', `Epoch step failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  },

  updateHyperparameters: async (params) => {
    const { sessionId, addLog } = get();
    if (!sessionId) return;
    try {
      await ApiClient.updateHyperparameters(sessionId, params);
      addLog('info', `Hyperparameters dynamically updated.`);
    } catch (err) {
      addLog('error', `Hyperparameter update failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  },

  step: async () => {
    const { sessionId, addLog } = get();
    if (!sessionId) return;

    try {
      const res = await ApiClient.stepSession(sessionId);
      if (res.event) {
        get().handleTraceEvent(res.event);
      }
    } catch (err) {
      addLog('error', `Step failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  },

  continueRun: async () => {
    const { sessionId, addLog } = get();
    if (!sessionId) return;

    try {
      set({ status: 'running' });
      await ApiClient.continueSession(sessionId);
      addLog('info', 'Resumed execution run.');
    } catch (err) {
      addLog('error', `Continue failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  },

  stop: async () => {
    const { sessionId, ws, addLog } = get();
    if (sessionId) {
      try {
        await ApiClient.stopSession(sessionId);
      } catch {
        // Ignore stop error
      }
    }
    if (ws) {
      ws.close();
    }
    set({
      status: 'idle',
      ws: null,
      activeNodePath: null,
      isTraining: false,
      sessionId: null,
      graphHash: null,
      nodeStates: {},
      retainedSummaries: {},
      trainingMetrics: null,
      lossHistory: [],
      valLossHistory: [],
      nodeGradientNorms: {},
      parameterNorms: {},
    });
    addLog('info', 'Execution session stopped.');
  },

  handleTraceEvent: (event: TraceEvent) => {
    const { addLog, retainedSummaries, nodeStates, lossHistory } = get();
    const nodePath = event.node_path;
    const shortId = nodePath.split('/').pop() || nodePath;

    const newSummaries = { ...retainedSummaries };
    if (event.outputs) {
      for (const [pId, summary] of Object.entries(event.outputs)) {
        newSummaries[`${nodePath}:${pId}`] = summary;
        newSummaries[`${shortId}:${pId}`] = summary;
        newSummaries[nodePath] = summary;
        newSummaries[shortId] = summary;
      }
    }

    const updatedNodeStates = { ...nodeStates };

    // Update metrics if event carries them
    if (event.metrics) {
      const m = event.metrics;
      const newLossHistory = [...lossHistory];
      if (event.event === 'batch_ended') {
        newLossHistory.push({
          step: m.step,
          loss: m.loss,
          lr: m.learning_rate,
          grad_norm: m.grad_norm,
          tokens_per_sec: m.tokens_per_sec,
        });
      }
      set({
        trainingMetrics: m,
        lossHistory: newLossHistory.slice(-500),
      });
    }

    switch (event.event) {
      case 'train_started':
      case 'run_started':
        set({ status: 'running' });
        addLog('info', 'Execution started.');
        break;

      case 'batch_ended': {
        const activeSessionId = get().sessionId;
        if (activeSessionId) {
          ApiClient.getMetrics(activeSessionId)
            .then((res) =>
              set({
                nodeGradientNorms: res.node_gradient_norms,
                parameterNorms: res.parameter_norms || {},
              })
            )
            .catch(() => {});
        }
        if (event.metrics) {
          addLog(
            'info',
            `Step ${event.metrics.step}: loss=${event.metrics.loss.toFixed(4)}, lr=${event.metrics.learning_rate.toExponential(2)}, grad_norm=${event.metrics.grad_norm.toFixed(3)} [${event.metrics.grad_status}]`
          );
        }
        break;
      }

      case 'validation_finished': {
        if (event.metrics && event.metrics.val_loss != null) {
          const { valLossHistory: curValHistory } = get();
          set({
            valLossHistory: [
              ...curValHistory,
              { step: event.metrics.step, val_loss: event.metrics.val_loss },
            ].slice(-500),
          });
        }
        break;
      }

      case 'node_started':
        updatedNodeStates[shortId] = 'running';
        updatedNodeStates[nodePath] = 'running';
        set({
          activeNodePath: nodePath,
          nodeStates: updatedNodeStates,
        });
        break;

      case 'node_finished':
        updatedNodeStates[shortId] = 'completed';
        updatedNodeStates[nodePath] = 'completed';
        set({
          nodeStates: updatedNodeStates,
          retainedSummaries: newSummaries,
        });
        break;

      case 'node_paused':
        updatedNodeStates[shortId] = 'paused';
        updatedNodeStates[nodePath] = 'paused';
        set({
          status: 'paused',
          nodeStates: updatedNodeStates,
          activeNodePath: nodePath,
          retainedSummaries: newSummaries,
        });
        addLog('warn', `Execution paused at breakpoint on '${nodePath}'.`);
        break;

      case 'anomaly_detected':
      case 'node_failed':
        updatedNodeStates[shortId] = 'failed';
        updatedNodeStates[nodePath] = 'failed';
        set({
          status: 'error',
          nodeStates: updatedNodeStates,
          activeNodePath: nodePath,
        });
        addLog('error', `Anomaly / failure: ${event.error || 'Unknown error'}`);
        break;

      case 'train_finished':
      case 'run_finished':
        set({
          status: 'completed',
          activeNodePath: null,
          isTraining: false,
        });
        addLog('info', 'Execution completed successfully.');
        break;
      case 'token_generated':
        set((state) => ({
          generatedText: state.generatedText + (event.token ?? ''),
        }));
        break;

      case 'generation_finished':
        addLog('info', 'Generation finished.');
        break;

        break;

      case 'run_cancelled':
        set({
          status: 'idle',
          activeNodePath: null,
          isTraining: false,
        });
        addLog('info', 'Execution cancelled.');
        break;
    }
  },
}));
