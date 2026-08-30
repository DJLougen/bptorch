import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { App } from '../src/App';
import {
  PROJECT_STORAGE_KEY,
  createInitialProject,
  serializeProject,
} from '../src/stores/projectStore';

describe('App Component', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    localStorage.clear();
  });

  it('renders application workspace', async () => {
    // Mock global fetch
    globalThis.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/examples/samples.json')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ count: 0, samples: [] }),
        } as Response);
      }
      if (url.includes('/api/v1/registry/nodes')) {
        return Promise.resolve({
          ok: true,
          json: async () => [],
        } as Response);
      }
      if (url.includes('/api/v1/graphs/validate')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            valid: true,
            graph_hash: 'test_hash',
            resolved_shapes: {},
            parameter_summary: {
              total_unique: 1000,
              trainable: 1000,
              frozen: 0,
              shared_references: 0,
            },
            diagnostics: [],
          }),
        } as Response);
      }
      return Promise.resolve({
        ok: true,
        json: async () => ({}),
      } as Response);
    });

    render(<App />);
    expect(screen.getByText(/bpTorch/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText(/Graph Valid/i)).toBeInTheDocument();
    });
  });

  it('restores saved state without requesting the bundled default template', async () => {
    const savedProject = createInitialProject();
    savedProject.project.id = 'saved_project';
    savedProject.project.name = 'Saved Browser Project';
    localStorage.setItem(PROJECT_STORAGE_KEY, serializeProject(savedProject));

    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/api/v1/registry/nodes')) {
        return Promise.resolve({
          ok: true,
          json: async () => [],
        } as Response);
      }
      if (url.includes('/api/v1/graphs/validate')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            valid: true,
            graph_hash: 'saved_hash',
            resolved_shapes: {},
            parameter_summary: {
              total_unique: 0,
              trainable: 0,
              frozen: 0,
              shared_references: 0,
            },
            diagnostics: [],
          }),
        } as Response);
      }
      return Promise.resolve({
        ok: true,
        json: async () => createInitialProject(),
      } as Response);
    });
    globalThis.fetch = fetchMock;

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('Saved Browser Project')).toBeInTheDocument();
    });
    expect(
      fetchMock.mock.calls.some(([url]) =>
        String(url).includes('/examples/arch_1_nanogpt_tiny/arch_1_nanogpt_tiny.nbp.json')
      )
    ).toBe(false);
  });

  it('loads the bundled default template when no saved state exists', async () => {
    const bundledProject = createInitialProject();
    bundledProject.project.id = 'nanogpt_default';
    bundledProject.project.name = 'Bundled Default Project';

    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/api/v1/registry/nodes')) {
        return Promise.resolve({
          ok: true,
          json: async () => [],
        } as Response);
      }
      if (url.includes('/examples/arch_1_nanogpt_tiny/arch_1_nanogpt_tiny.nbp.json')) {
        return Promise.resolve({
          ok: true,
          json: async () => bundledProject,
        } as Response);
      }
      return Promise.resolve({
        ok: true,
        json: async () => ({
          valid: true,
          graph_hash: 'default_hash',
          resolved_shapes: {},
          parameter_summary: {
            total_unique: 0,
            trainable: 0,
            frozen: 0,
            shared_references: 0,
          },
          diagnostics: [],
        }),
      } as Response);
    });
    globalThis.fetch = fetchMock;

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('Bundled Default Project')).toBeInTheDocument();
    });
    expect(
      fetchMock.mock.calls.some(([url]) =>
        String(url).includes('/examples/arch_1_nanogpt_tiny/arch_1_nanogpt_tiny.nbp.json')
      )
    ).toBe(true);
  });
});
