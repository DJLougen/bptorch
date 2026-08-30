import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { SamplesGallery } from '../src/components/SamplesGallery';

describe('SamplesGallery', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('renders gallery dialog when open', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        count: 2,
        description: 'Test gallery',
        samples: [
          {
            id: 'arch_1',
            name: 'Sample One',
            category: 'Transformers',
            description: 'Desc one',
            highlight: 'Highlight one',
            tags: ['gpt'],
            difficulty: 'beginner',
            path: '/examples/arch_1/arch_1.nbp.json',
          },
        ],
      }),
    });

    render(<SamplesGallery open={true} onClose={() => {}} />);
    expect(await screen.findByRole('dialog', { name: 'Sample Gallery' })).toBeInTheDocument();
    expect(screen.getByText('Sample One')).toBeInTheDocument();
    expect(screen.getByText('Highlight one')).toBeInTheDocument();
  });
});
