/**
 * API client for interacting with the Neural Blueprint Studio backend runtime.
 */
import {
  AvailableTestInfo,
  CompilationResponse,
  NodeDefinitionSummary,
  Project,
  TensorSummary,
  TestSuiteResult,
  TraceEvent,
  TrainingMetrics,
  ValidationResponse,
} from './contracts';

const API_BASE = '/api/v1';

export interface HealthState {
  status: string;
  version: string;
  runtime: string;
  torch_version: string;
}

export class ApiClient {
  static async getHealth(): Promise<HealthState> {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
    return res.json();
  }

  static async getNodeCatalog(): Promise<NodeDefinitionSummary[]> {
    const res = await fetch(`${API_BASE}/registry/nodes`);
    if (!res.ok) throw new Error(`Failed to load node catalog: ${res.statusText}`);
    return res.json();
  }

  static async validateGraph(project: Project): Promise<ValidationResponse> {
    const res = await fetch(`${API_BASE}/graphs/validate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project }),
    });
    if (!res.ok) throw new Error(`Validation failed: ${res.statusText}`);
    return res.json();
  }

  static async compileModel(
    project: Project,
    device: string = 'cpu',
    mode: string = 'inference'
  ): Promise<CompilationResponse> {
    const res = await fetch(`${API_BASE}/models/compile`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project, device, mode }),
    });
    if (!res.ok) {
      const err = (await res.json()) as { detail?: { message?: string } };
      throw new Error(err.detail?.message || `Compilation failed: ${res.statusText}`);
    }
    return res.json();
  }

  static async runSession(
    sessionId: string,
    inputs: Record<string, unknown>,
    mode: string = 'inspection',
    speed: string = 'normal'
  ): Promise<{ status: string; session_id: string }> {
    const res = await fetch(`${API_BASE}/sessions/${sessionId}/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mode,
        inputs,
        trace: { enabled: true, speed },
      }),
    });
    if (!res.ok) throw new Error(`Failed to run session: ${res.statusText}`);
    return res.json();
  }

  static async stepSession(sessionId: string): Promise<{ status: string; event?: TraceEvent }> {
    const res = await fetch(`${API_BASE}/sessions/${sessionId}/step`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error(`Failed to step session: ${res.statusText}`);
    return res.json();
  }

  static async continueSession(sessionId: string): Promise<{ status: string }> {
    const res = await fetch(`${API_BASE}/sessions/${sessionId}/continue`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error(`Failed to continue session: ${res.statusText}`);
    return res.json();
  }

  static async stopSession(sessionId: string): Promise<{ status: string }> {
    const res = await fetch(`${API_BASE}/sessions/${sessionId}/stop`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error(`Failed to stop session: ${res.statusText}`);
    return res.json();
  }

  // --- Training APIs ---

  static async startTraining(
    sessionId: string,
    maxSteps?: number,
    speedDelay: number = 0.0
  ): Promise<{ status: string; session_id: string; max_steps: number }> {
    const res = await fetch(`${API_BASE}/sessions/${sessionId}/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ max_steps: maxSteps, speed_delay: speedDelay }),
    });
    if (!res.ok) throw new Error(`Failed to start training: ${res.statusText}`);
    return res.json();
  }

  static async pauseTraining(sessionId: string): Promise<{ status: string; session_id: string }> {
    const res = await fetch(`${API_BASE}/sessions/${sessionId}/pause`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error(`Failed to pause training: ${res.statusText}`);
    return res.json();
  }

  static async resumeTraining(sessionId: string): Promise<{ status: string; session_id: string }> {
    const res = await fetch(`${API_BASE}/sessions/${sessionId}/resume`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error(`Failed to resume training: ${res.statusText}`);
    return res.json();
  }

  static async stepBatch(
    sessionId: string
  ): Promise<{ status: string; session_id: string; step: number; event?: TraceEvent; metrics: TrainingMetrics }> {
    const res = await fetch(`${API_BASE}/sessions/${sessionId}/step-batch`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error(`Failed to step batch: ${res.statusText}`);
    return res.json();
  }

  static async stepEpoch(
    sessionId: string
  ): Promise<{ status: string; session_id: string; epoch: number; step: number; events_count: number; metrics: TrainingMetrics }> {
    const res = await fetch(`${API_BASE}/sessions/${sessionId}/step-epoch`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error(`Failed to step epoch: ${res.statusText}`);
    return res.json();
  }

  static async updateHyperparameters(
    sessionId: string,
    params: { learning_rate?: number; weight_decay?: number; grad_clip?: number }
  ): Promise<{ status: string; learning_rate: number; weight_decay: number; grad_clip: number }> {
    const res = await fetch(`${API_BASE}/sessions/${sessionId}/hyperparameters`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (!res.ok) throw new Error(`Failed to update hyperparameters: ${res.statusText}`);
    return res.json();
  }

  static async saveCheckpoint(
    sessionId: string,
    path?: string
  ): Promise<{ status: string; path: string; step: number; epoch: number }> {
    const res = await fetch(`${API_BASE}/sessions/${sessionId}/checkpoint/save`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    });
    if (!res.ok) throw new Error(`Failed to save checkpoint: ${res.statusText}`);
    return res.json();
  }

  static async loadCheckpoint(
    sessionId: string,
    path: string
  ): Promise<{ status: string; step: number; epoch: number; loss?: number; best_loss?: number }> {
    const res = await fetch(`${API_BASE}/sessions/${sessionId}/checkpoint/load`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    });
    if (!res.ok) throw new Error(`Failed to load checkpoint: ${res.statusText}`);
    return res.json();
  }

  static async getMetrics(
    sessionId: string
  ): Promise<{
    current_metrics: TrainingMetrics | null;
    loss_history: Array<{ step: number; loss: number; lr: number; grad_norm: number; tokens_per_sec: number }>;
    best_loss: number | null;
    node_gradient_norms: Record<string, number>;
  }> {
    const res = await fetch(`${API_BASE}/sessions/${sessionId}/metrics`);
    if (!res.ok) throw new Error(`Failed to fetch metrics: ${res.statusText}`);
    return res.json();
  }

  static async getTensorSummary(sessionId: string, tensorId: string): Promise<TensorSummary> {
    const res = await fetch(`${API_BASE}/sessions/${sessionId}/tensors/${encodeURIComponent(tensorId)}/summary`);
    if (!res.ok) throw new Error(`Failed to fetch tensor summary: ${res.statusText}`);
    return res.json();
  }

  static connectTraceWebSocket(
    sessionId: string,
    onEvent: (event: TraceEvent) => void,
    onError?: (error: Event) => void
  ): WebSocket {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/api/v1/sessions/${sessionId}/events`;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (e) => {
      try {
        const event = JSON.parse(e.data) as TraceEvent;
        onEvent(event);
      } catch (err) {
        console.error('Failed to parse trace event:', err);
      }
    };

    if (onError) {
      ws.onerror = onError;
    }

    return ws;
  }

  static async runTestSuite(project: Project, enabledTests?: string[]): Promise<TestSuiteResult> {
    const res = await fetch(`${API_BASE}/test/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project, enabled_tests: enabledTests }),
    });
    if (!res.ok) throw new Error(`Test suite execution failed: ${res.statusText}`);
    return res.json();
  }

  static async getAvailableTests(): Promise<AvailableTestInfo[]> {
    const res = await fetch(`${API_BASE}/test/suites`);
    if (!res.ok) throw new Error(`Failed to load test suites: ${res.statusText}`);
    return res.json();
  }
}
