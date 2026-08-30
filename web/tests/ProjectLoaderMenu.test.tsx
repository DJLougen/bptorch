import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ProjectLoaderMenu } from '../src/components/ProjectLoaderMenu';
import { createInitialProject, useProjectStore } from '../src/stores/projectStore';
import { useTraceStore } from '../src/stores/traceStore';

describe('ProjectLoaderMenu', () => {
  beforeEach(() => {
    useProjectStore.getState().loadProject(createInitialProject());
    useTraceStore.setState({ logs: [], status: 'idle' });
    vi.restoreAllMocks();
  });

  it('opens nested architecture samples submenu from the templates menu', async () => {
    globalThis.fetch = vi.fn().mockImplementation(async (url: string) => {
      if (url === '/examples/samples.json') {
        return {
          ok: true,
          json: async () => ({
            count: 1,
            samples: [
              {
                id: 'arch_1_nanogpt_tiny',
                name: 'Arch 1: nanoGPT Tiny',
                category: 'Transformers',
                description: 'Tiny GPT',
                highlight: 'Starter',
                tags: ['gpt'],
                difficulty: 'beginner',
                path: '/examples/arch_1_nanogpt_tiny/arch_1_nanogpt_tiny.nbp.json',
              },
            ],
          }),
        };
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });

    render(<ProjectLoaderMenu />);
    fireEvent.click(screen.getByRole('button', { name: 'Load blueprint template' }));

    expect(screen.getByRole('menu', { name: 'Blueprint templates' })).toBeInTheDocument();
    expect(screen.getByRole('menuitem', { name: /Architecture Samples/ })).toBeInTheDocument();

    fireEvent.mouseEnter(screen.getByRole('menuitem', { name: /Architecture Samples/ }));

    await waitFor(() => {
      expect(screen.getByRole('menu', { name: 'Architecture samples' })).toBeInTheDocument();
    });
    expect(screen.getByRole('menuitem', { name: 'Arch 1: nanoGPT Tiny' })).toBeInTheDocument();
    expect(screen.getByText('Transformers')).toBeInTheDocument();
  });

  it('keeps the samples submenu open after click when the pointer leaves', async () => {
    globalThis.fetch = vi.fn().mockImplementation(async (url: string) => {
      if (url === '/examples/samples.json') {
        return {
          ok: true,
          json: async () => ({
            count: 1,
            samples: [
              {
                id: 'arch_1_nanogpt_tiny',
                name: 'Arch 1: nanoGPT Tiny',
                category: 'Transformers',
                description: 'Tiny GPT',
                highlight: 'Starter',
                tags: ['gpt'],
                difficulty: 'beginner',
                path: '/examples/arch_1_nanogpt_tiny/arch_1_nanogpt_tiny.nbp.json',
              },
            ],
          }),
        };
      }
      throw new Error(`Unexpected fetch: ${url}`);
    });

    render(<ProjectLoaderMenu />);
    fireEvent.click(screen.getByRole('button', { name: 'Load blueprint template' }));

    const samplesItem = screen.getByRole('menuitem', { name: /Architecture Samples/ });
    fireEvent.click(samplesItem);

    await waitFor(() => {
      expect(screen.getByRole('menu', { name: 'Architecture samples' })).toBeInTheDocument();
    });

    fireEvent.mouseLeave(samplesItem);

    expect(screen.getByRole('menu', { name: 'Architecture samples' })).toBeInTheDocument();
  });
});
