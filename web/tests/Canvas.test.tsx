import { act, createEvent, fireEvent, render } from '@testing-library/react';
import type { Connection } from '@xyflow/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Edge, NodeDefinitionSummary, Project } from '../src/api/contracts';

interface ReactFlowMockProps {
  onConnect?: (connection: Connection) => void;
  onEdgeDoubleClick?: (event: React.MouseEvent, edge: { id: string }) => void;
}

const flowHarness = vi.hoisted(() => ({
  onConnect: undefined as ReactFlowMockProps['onConnect'] | undefined,
  onEdgeDoubleClick: undefined as ReactFlowMockProps['onEdgeDoubleClick'] | undefined,
  fitView: vi.fn(),
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
  SelectionMode: { Partial: 'partial', Full: 'full' },
  ReactFlow: (props: ReactFlowMockProps) => {
    flowHarness.onConnect = props.onConnect;
    flowHarness.onEdgeDoubleClick = props.onEdgeDoubleClick;
    return null;
  },
  getBezierPath: () => ['', 0, 0],
  useReactFlow: () => ({
    screenToFlowPosition: ({ x, y }: { x: number; y: number }) => ({ x, y }),
    fitView: flowHarness.fitView,
  }),
}));

import { Canvas, validateConnection } from '../src/canvas/Canvas';
import { createInitialProject, useProjectStore } from '../src/stores/projectStore';
import { useTraceStore } from '../src/stores/traceStore';
import { useUIStore } from '../src/stores/uiStore';

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
  {
    type_id: 'test.middle@1',
    version: 1,
    display_name: 'Middle Node',
    category: 'Layers',
    description: 'Test middle node for splicing',
    is_composite: false,
    property_schema: {},
    default_inputs: [
      {
        id: 'in',
        display_name: 'In',
        direction: 'input',
        kind: 'data',
        multiplicity: 'single',
        tensor_type: { dtype_family: 'floating' },
      },
    ],
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
  it('deletes selected node on Delete key and restores on Ctrl+Z', () => {
    const project = createConnectionProject();
    project.model.graphs.graph_test.nodes.push({
      id: 'n1',
      definition_id: 'test.source@1',
      display_name: 'Node 1',
      properties: {},
      metadata: { breakpoint: false, disabled: false },
    });

    useProjectStore.setState({
      project,
      openGraphId: 'graph_test',
    });

    render(<Canvas catalog={catalog} />);

    act(() => {
      useUIStore.getState().selectNode('n1');
    });

    act(() => {
      fireEvent.keyDown(window, { key: 'Delete' });
    });

    expect(
      useProjectStore.getState().project.model.graphs.graph_test.nodes.find((n) => n.id === 'n1')
    ).toBeUndefined();

    act(() => {
      fireEvent.keyDown(window, { key: 'z', ctrlKey: true });
    });

    expect(
      useProjectStore.getState().project.model.graphs.graph_test.nodes.find((n) => n.id === 'n1')
    ).toBeDefined();
  });
  it('copies and pastes selected nodes via Ctrl+C and Ctrl+V', () => {
    const project = createConnectionProject();
    project.model.graphs.graph_test.nodes.push({
      id: 'n1',
      definition_id: 'test.source@1',
      display_name: 'Node 1',
      properties: {},
      metadata: { breakpoint: false, disabled: false },
    });

    useProjectStore.setState({
      project,
      openGraphId: 'graph_test',
    });

    render(<Canvas catalog={catalog} />);

    act(() => {
      useUIStore.getState().selectNodes(['n1']);
    });

    act(() => {
      fireEvent.keyDown(window, { key: 'c', ctrlKey: true });
    });

    act(() => {
      fireEvent.keyDown(window, { key: 'v', ctrlKey: true });
    });

    const nodes = useProjectStore.getState().project.model.graphs.graph_test.nodes;
    expect(nodes.length).toBe(4);
    expect(nodes.some((n) => n.id.startsWith('node_source_'))).toBe(true);
  });

  it('duplicates selected nodes via Ctrl+D', () => {
    const project = createConnectionProject();
    project.model.graphs.graph_test.nodes.push({
      id: 'n1',
      definition_id: 'test.source@1',
      display_name: 'Node 1',
      properties: {},
      metadata: { breakpoint: false, disabled: false },
    });

    useProjectStore.setState({
      project,
      openGraphId: 'graph_test',
    });

    render(<Canvas catalog={catalog} />);

    act(() => {
      useUIStore.getState().selectNodes(['n1']);
    });

    act(() => {
      fireEvent.keyDown(window, { key: 'd', ctrlKey: true });
    });

    const nodes = useProjectStore.getState().project.model.graphs.graph_test.nodes;
    expect(nodes.length).toBe(4);
  });
  it('splices a dropped node onto an existing wire between two connected nodes', () => {
    const project = createConnectionProject();
    project.ui.node_positions.graph_test = {
      source: { x: 100, y: 100 },
      target: { x: 300, y: 100 },
    };
    project.model.graphs.graph_test.edges.push({
      id: 'e_orig',
      source: { node_id: 'source', port_id: 'out' },
      target: { node_id: 'target', port_id: 'in' },
    });

    useProjectStore.setState({
      project,
      openGraphId: 'graph_test',
    });

    const { container } = render(<Canvas catalog={catalog} />);
    const canvasDiv = container.firstChild as HTMLElement;

    act(() => {
      const dropEvent = createEvent.drop(canvasDiv);
      Object.defineProperty(dropEvent, 'clientX', { value: 200 });
      Object.defineProperty(dropEvent, 'clientY', { value: 100 });
      Object.defineProperty(dropEvent, 'dataTransfer', {
        value: {
          getData: (format: string) => (format === 'application/bptorch-node' ? 'test.middle@1' : ''),
        },
      });
      fireEvent(canvasDiv, dropEvent);
    });
    const g = useProjectStore.getState().project.model.graphs.graph_test;
    // Original edge should be removed
    expect(g.edges.some((e) => e.id === 'e_orig')).toBe(false);
    // Middle node should be added
    const middleNode = g.nodes.find((n) => n.definition_id === 'test.middle@1');
    expect(middleNode).toBeDefined();
    // Two rewired edges should exist connecting source -> middle and middle -> target
    expect(g.edges.some((e) => e.source.node_id === 'source' && e.target.node_id === middleNode!.id)).toBe(true);
    expect(g.edges.some((e) => e.source.node_id === middleNode!.id && e.target.node_id === 'target')).toBe(true);
  });

  it('groups selected nodes into a new module graph via Ctrl+G', () => {
    const project = createConnectionProject();
    project.model.graphs.graph_test.nodes.push({
      id: 'n1',
      definition_id: 'test.source@1',
      display_name: 'Node 1',
      properties: {},
      metadata: { breakpoint: false, disabled: false },
    });

    useProjectStore.setState({
      project,
      openGraphId: 'graph_test',
    });

    render(<Canvas catalog={catalog} />);

    act(() => {
      useUIStore.getState().selectNodes(['n1']);
    });

    act(() => {
      fireEvent.keyDown(window, { key: 'g', ctrlKey: true });
    });

    const currentNodes = useProjectStore.getState().project.model.graphs.graph_test.nodes;
    expect(currentNodes.some((n) => n.definition_id.startsWith('custom.graph_custom_'))).toBe(true);
    const graphKeys = Object.keys(useProjectStore.getState().project.model.graphs);
    expect(graphKeys.some((k) => k.startsWith('graph_custom_'))).toBe(true);
  });

  it('nudges selected nodes by 10px on ArrowRight keydown', () => {
    const project = createConnectionProject();
    project.model.graphs.graph_test.nodes.push({
      id: 'n_nudge',
      definition_id: 'test.source@1',
      display_name: 'Node Nudge',
      properties: {},
      metadata: { breakpoint: false, disabled: false },
    });
    project.ui.node_positions.graph_test = {
      n_nudge: { x: 50, y: 50 },
    };

    useProjectStore.setState({
      project,
      openGraphId: 'graph_test',
    });

    render(<Canvas catalog={catalog} />);

    act(() => {
      useUIStore.getState().selectNodes(['n_nudge']);
    });

    act(() => {
      fireEvent.keyDown(window, { key: 'ArrowRight' });
    });

    const pos = useProjectStore.getState().project.ui.node_positions.graph_test?.n_nudge;
    expect(pos?.x).toBe(60);
  });

  it('appends an edge waypoint on edge double click', () => {
    const project = createConnectionProject();
    project.ui.edge_waypoints = {};

    useProjectStore.setState({
      project,
      openGraphId: 'graph_test',
    });

    render(<Canvas catalog={catalog} />);

    act(() => {
      flowHarness.onEdgeDoubleClick?.(
        { clientX: 150, clientY: 120, preventDefault: () => {} } as unknown as React.MouseEvent,
        { id: 'e_test' }
      );
    });

    const waypoints = useProjectStore.getState().project.ui.edge_waypoints?.graph_test?.e_test;
    expect(waypoints).toBeDefined();
    expect(waypoints?.length).toBe(1);
    expect(waypoints?.[0]).toEqual({ x: 150, y: 120 });
  });

  it('triggers fitView on Shift+2 keydown for selected nodes', () => {
    const project = createConnectionProject();
    useProjectStore.setState({
      project,
      openGraphId: 'graph_test',
    });

    render(<Canvas catalog={catalog} />);

    act(() => {
      useUIStore.getState().selectNodes(['source', 'target']);
    });

    act(() => {
      fireEvent.keyDown(window, { key: '2', shiftKey: true });
    });

    expect(flowHarness.fitView).toHaveBeenCalledWith({
      nodes: [{ id: 'source' }, { id: 'target' }],
      padding: 0.2,
    });
  });
});
