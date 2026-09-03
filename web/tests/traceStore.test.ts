import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiClient } from '../src/api/client';
import { createInitialProject } from '../src/stores/projectStore';
import { useTraceStore } from '../src/stores/traceStore';

describe('traceStore session lifecycle', () => {
  beforeEach(() => {
    useTraceStore.setState({
      sessionId: null,
      status: 'idle',
      nodeGradientNorms: {},
      logs: [],
    });
  });

  it('stop clears sessionId so stepBatch does not reuse a stopped session', async () => {
    useTraceStore.setState({ sessionId: 's1' });
    const stepBatchSpy = vi.spyOn(ApiClient, 'stepBatch').mockResolvedValue({
      status: 'idle',
      session_id: 's2',
      step: 0,
      metrics: {
        epoch: 0,
        step: 0,
        loss: 0,
        learning_rate: 0,
        grad_norm: 0,
        grad_status: 'healthy',
        tokens_per_sec: 0,
        step_time_ms: 0,
        vram_mb: 0,
      },
    });
    vi.spyOn(ApiClient, 'stopSession').mockResolvedValue({ status: 'stopped' });

    await useTraceStore.getState().stop();
    expect(useTraceStore.getState().sessionId).toBeNull();

    await useTraceStore.getState().stepBatch(createInitialProject());
    expect(stepBatchSpy).not.toHaveBeenCalled();
  });

  it('handleTraceEvent populates nodeGradientNorms from metrics endpoint', async () => {
    useTraceStore.setState({ sessionId: 's1' });
    vi.spyOn(ApiClient, 'getMetrics').mockResolvedValue({
      current_metrics: null,
      loss_history: [],
      best_loss: null,
      node_gradient_norms: { 'node_a.weight': 0.5 },
      parameter_norms: { 'node_a.weight': 1.2 },
    });
    useTraceStore.getState().handleTraceEvent({
      sequence: 1,
      event: 'batch_ended',
      session_id: 's1',
      graph_hash: 'hash',
      node_path: 'node_a',
      timestamp_ns: 0,
      metrics: {
        epoch: 0,
        step: 1,
        loss: 1.2,
        learning_rate: 0.001,
        grad_norm: 0.5,
        grad_status: 'healthy',
        tokens_per_sec: 10,
        step_time_ms: 1,
        vram_mb: 0,
      },
    });

    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(useTraceStore.getState().nodeGradientNorms).toEqual({ 'node_a.weight': 0.5 });
    expect(useTraceStore.getState().parameterNorms).toEqual({ 'node_a.weight': 1.2 });
  });
});
