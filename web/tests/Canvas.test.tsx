import { act, render } from '@testing-library/react';
import type { Connection } from '@xyflow/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Edge, NodeDefinitionSummary, Project } from '../src/api/contracts';

interface ReactFlowMockProps {
  onConnect: (connection: Connection) => void;
}

const flowHarness = vi.hoisted(() => ({
  onConnect: undefined as ReactFlowMockProps['onConnect'] | undefined,
}));

vi.mock('@xyflow/react', () => ({
  Background: () => null,
  BackgroundVariant: { Dots: 'dots' },
  BaseEdge: () => null,
  Controls: () => null,
  EdgeLabelRenderer: () => null,
  Handle: () => null,
  MiniMap: () => null,
  Position: { Left: 'left', Right: 'right' },
  ReactFlow: ({ onConnect }: ReactFlowMockProps) => {
    flowHarness.onConnect = onConnect;
    return null;
  },
  getBezierPath: () => ['', 0, 0],
  useReactFlow: () => ({
    screenToFlowPosition: ({ x, y }: { x: number; y: number }) => ({ x, y }),
  }),
}));

import { Canvas, validateConnection } from '../src/canvas/Canvas';
import { createInitialProject, useProjectStore } from '../src/stores/projectStore';
import { useTraceStore } from '../src/stores/traceStore';

const catalog: NodeDefinitionSummary[] = [
  {
    type_id: 'test.source@1',
    version: 1,
    display_name: 'Source',
    category: 'Inputs',
    description: 'Test source',
    is_composite: false,
    property_schema: {},
    default_inputs: [],
    default_outputs: [
      {
        id: 'out',
        display_name: 'Out',
        direction: 'output',
        kind: 'data',
        multiplicity: 'multiple',
        tensor_type: { dtype_family: 'floating' },
      },
    ],
  },
  {
    type_id: 'test.target@1',
    version: 1,
    display_name: 'Target',
    category: 'Layers',
    description: 'Test target',
    is_composite: false,
    property_schema: {},
    default_inputs: [
      {
        id: 'in',
        display_name: 'In',
        direction: 'input',
        kind: 'data',
        multiplicity: 'single',
        tensor_type: { dtype_family: 'integer' },
      },
    ],
    default_outputs: [],
  },
];

function createConnectionProject(): Project {
  const project = createInitialProject();
  project.model.root_graph_id = 'graph_test';
  project.model.graphs = {
    graph_test: {
      id: 'graph_test',
      name: 'Connection Test',
      kind: 'root',
      interface: { inputs: [], outputs: [] },
      nodes: [
        {
          id: 'source',
          definition_id: 'test.source@1',
          display_name: 'Source',
          properties: {},
          metadata: { breakpoint: false, disabled: false },
        },
        {
          id: 'target',
          definition_id: 'test.target@1',
          display_name: 'Target',
          properties: {},
          metadata: { breakpoint: false, disabled: false },
        },
      ],
      edges: [],
    },
  };
  project.ui.open_graph_id = 'graph_test';
  project.ui.node_positions = { graph_test: {} };
  project.ui.graph_viewports = {};
  return project;
}

describe('Canvas connection authoring', () => {
  let originalConnectEdge: (edge: Edge) => void;

  beforeEach(() => {
    localStorage.clear();
    flowHarness.onConnect = undefined;
    useTraceStore.setState({ logs: [] });
    useProjectStore.getState().loadProject(createConnectionProject());
    originalConnectEdge = useProjectStore.getState().connectEdge;
  });

  afterEach(() => {
    useProjectStore.setState({ connectEdge: originalConnectEdge });
  });

  it('logs and rejects an incompatible dtype without calling connectEdge', () => {
    const connectEdge = vi.fn<(edge: Edge) => void>();
    useProjectStore.setState({ connectEdge });
    render(<Canvas catalog={catalog} />);

    expect(flowHarness.onConnect).toBeDefined();
    act(() => {
      flowHarness.onConnect?.({
        source: 'source',
        sourceHandle: 'out',
        target: 'target',
        targetHandle: 'in',
      });
    });

    expect(connectEdge).not.toHaveBeenCalled();
    expect(useTraceStore.getState().logs.at(-1)?.level).toBe('warn');
    expect(useTraceStore.getState().logs.at(-1)?.message).toMatch(/dtype families/i);
  });

  it('rejects a second edge into a catalog port with single multiplicity', () => {
    const project = createConnectionProject();
    const graph = project.model.graphs.graph_test;
    graph.edges.push({
      id: 'existing_edge',
      source: { node_id: 'source', port_id: 'out' },
      target: { node_id: 'target', port_id: 'in' },
    });
    const compatibleCatalog = catalog.map((definition) =>
      definition.type_id === 'test.target@1'
        ? {
            ...definition,
            default_inputs: definition.default_inputs.map((port) => ({
              ...port,
              tensor_type: { dtype_family: 'floating' as const },
            })),
          }
        : definition
    );

    const result = validateConnection(
      {
        source: 'source',
        sourceHandle: 'out',
        target: 'target',
        targetHandle: 'in',
      },
      graph,
      compatibleCatalog
    );

    expect(result.valid).toBe(false);
    if (result.valid) {
      throw new Error('Expected the occupied single-input port to be rejected');
    }
    expect(result.reason).toMatch(/only one connection/i);
  });
});
