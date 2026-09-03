import { fireEvent, render } from '@testing-library/react';
import { Position, ReactFlowProvider } from '@xyflow/react';
import { describe, expect, it } from 'vitest';
import { BlueprintEdge } from '../src/canvas/BlueprintEdge';
import { useUIStore } from '../src/stores/uiStore';

const baseEdgeProps = {
  source: 'node_a',
  target: 'node_b',
  sourceX: 0,
  sourceY: 0,
  targetX: 100,
  targetY: 0,
  sourcePosition: Position.Right,
  targetPosition: Position.Left,
};

describe('BlueprintEdge', () => {
  it('selects an edge without a shape label', () => {
    useUIStore.setState({ selectedEdgeId: null });
    const { container } = render(
      <ReactFlowProvider>
        <svg>
          <BlueprintEdge
            id="e1"
            {...baseEdgeProps}
            data={{}}
          />
        </svg>
      </ReactFlowProvider>
    );
    const paths = Array.from(container.querySelectorAll('path')).filter(
      (path) => !path.closest('defs') && !path.classList.contains('blueprint-edge-flow')
    );
    expect(paths.length).toBeGreaterThan(0);
    fireEvent.click(paths[0]);
    expect(useUIStore.getState().selectedEdgeId).toBe('e1');
  });

  it('renders directional flow markers', () => {
    const { container } = render(
      <ReactFlowProvider>
        <svg>
          <BlueprintEdge
            id="e2"
            {...baseEdgeProps}
            targetX={120}
            data={{ isExec: false }}
          />
        </svg>
      </ReactFlowProvider>
    );

    expect(container.querySelector('.blueprint-edge-flow')).not.toBeNull();
    expect(container.querySelector('polygon')).not.toBeNull();
  });
});
