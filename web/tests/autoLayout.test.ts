import { describe, it, expect } from 'vitest';
import { computeAutoLayout } from '../src/canvas/autoLayout';
import type { GraphDefinition } from '../src/api/contracts';

describe('computeAutoLayout', () => {
  it('positions a three-node chain A -> B -> C with x_A < x_B < x_C', () => {
    const graph: GraphDefinition = {
      id: 'test_graph',
      name: 'Test Graph',
      kind: 'root',
      interface: { inputs: [], outputs: [] },
      nodes: [
        {
          id: 'node_A',
          definition_id: 'builtin.linear@1',
          display_name: 'Node A',
          properties: {},
          metadata: { breakpoint: false, disabled: false },
        },
        {
          id: 'node_B',
          definition_id: 'builtin.relu@1',
          display_name: 'Node B',
          properties: {},
          metadata: { breakpoint: false, disabled: false },
        },
        {
          id: 'node_C',
          definition_id: 'builtin.linear@1',
          display_name: 'Node C',
          properties: {},
          metadata: { breakpoint: false, disabled: false },
        },
      ],
      edges: [
        {
          id: 'e1',
          source: { node_id: 'node_A', port_id: 'output' },
          target: { node_id: 'node_B', port_id: 'input' },
        },
        {
          id: 'e2',
          source: { node_id: 'node_B', port_id: 'output' },
          target: { node_id: 'node_C', port_id: 'input' },
        },
      ],
    };

    const positions = computeAutoLayout(graph);
    expect(positions.node_A).toBeDefined();
    expect(positions.node_B).toBeDefined();
    expect(positions.node_C).toBeDefined();

    expect(positions.node_A.x).toBeLessThan(positions.node_B.x);
    expect(positions.node_B.x).toBeLessThan(positions.node_C.x);
  });
});
