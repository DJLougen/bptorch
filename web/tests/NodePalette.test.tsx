import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, beforeEach } from 'vitest';
import { NodeDefinitionSummary } from '../src/api/contracts';
import { NodePalette } from '../src/palette/NodePalette';
import { useUIStore } from '../src/stores/uiStore';
const mockCatalog: NodeDefinitionSummary[] = [
  {
    type_id: 'builtin.linear@1',
    version: 1,
    display_name: 'Linear',
    category: 'Layers',
    description: 'Linear transformation',
    icon: 'GitCommit',
    is_composite: false,
    property_schema: {},
    default_inputs: [],
    default_outputs: [],
  },
  {
    type_id: 'builtin.gelu@1',
    version: 1,
    display_name: 'GELU',
    category: 'Layers',
    description: 'GELU activation',
    icon: 'Activity',
    is_composite: false,
    property_schema: {},
    default_inputs: [],
    default_outputs: [],
  },
];

describe('NodePalette Component', () => {
  beforeEach(() => {
    useUIStore.setState({ paletteSearchQuery: '', isPaletteOpen: true });
  });

  it('renders categories and filters nodes with search', () => {
    render(<NodePalette catalog={mockCatalog} />);

    expect(screen.getByText(/Layers/i)).toBeInTheDocument();
    expect(screen.getByText('Linear')).toBeInTheDocument();
    expect(screen.getByText('GELU')).toBeInTheDocument();

    // Search filter
    const searchInput = screen.getByPlaceholderText('Search nodes...');
    fireEvent.change(searchInput, { target: { value: 'Linear' } });

    expect(screen.getByText('Linear')).toBeInTheDocument();
    expect(screen.queryByText('GELU')).not.toBeInTheDocument();
  });

  it('displays Debug category or Comment when catalog includes builtin.comment@1', () => {
    const catalogWithComment: NodeDefinitionSummary[] = [
      ...mockCatalog,
      {
        type_id: 'builtin.comment@1',
        version: 1,
        display_name: 'Comment',
        category: 'Debug',
        description: 'Documentation comment node',
        icon: 'Terminal',
        is_composite: false,
        property_schema: {},
        default_inputs: [],
        default_outputs: [],
      },
    ];

    render(<NodePalette catalog={catalogWithComment} />);
    expect(screen.getByText('Debug')).toBeInTheDocument();
    expect(screen.getByText('Comment')).toBeInTheDocument();
  });
});
