import { fireEvent, render, screen } from '@testing-library/react';
import { describe, it, expect, beforeEach } from 'vitest';
import { PropertyInspector } from '../src/inspector/PropertyInspector';
import { useProjectStore } from '../src/stores/projectStore';
import { useUIStore } from '../src/stores/uiStore';

describe('PropertyInspector Component', () => {
  beforeEach(() => {
    useUIStore.setState({ selectedNodeId: null });
  });

  it('renders model configuration when no node is selected', () => {
    render(<PropertyInspector catalog={[]} />);
    expect(screen.getByText('Model Configuration')).toBeInTheDocument();
    expect(screen.getByText('block size')).toBeInTheDocument();
    expect(screen.getByText('vocab size')).toBeInTheDocument();
    expect(screen.getByText('n embd')).toBeInTheDocument();
  });

  it('display name edits mark dirty and undo restores the prior name', () => {
    const base = useProjectStore.getState().project;
    const graphId = base.model.root_graph_id;
    const graph = base.model.graphs[graphId];
    const nextProject = {
      ...base,
      model: {
        ...base.model,
        graphs: {
          ...base.model.graphs,
          [graphId]: {
            ...graph,
            nodes: [
              {
                id: 'node_a',
                definition_id: 'builtin.linear@1',
                display_name: 'Original',
                properties: { in_features: 16, out_features: 16 },
                metadata: { breakpoint: false },
              },
            ],
          },
        },
      },
    };
    useProjectStore.setState({ project: nextProject, openGraphId: graphId, isDirty: false, undoStack: [], redoStack: [] });
    useUIStore.setState({ selectedNodeId: 'node_a' });

    render(<PropertyInspector catalog={[]} />);
    fireEvent.change(screen.getByDisplayValue('Original'), { target: { value: 'Renamed' } });
    expect(useProjectStore.getState().isDirty).toBe(true);
    useProjectStore.getState().undo();
    expect(useProjectStore.getState().project.model.graphs[graphId].nodes[0].display_name).toBe('Original');
  });
});
