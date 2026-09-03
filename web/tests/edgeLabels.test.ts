import { describe, expect, it } from 'vitest';
import { compactEdgeLabel } from '../src/canvas/edgeLabels';

describe('compactEdgeLabel zoom adaptation', () => {
  it('formats edge labels based on zoom levels', () => {
    expect(compactEdgeLabel('float32 [B, T, 128]', 1)).toBe('float32 [B, T, 128]');
    expect(compactEdgeLabel('float32 [B, T, 128]', 0.5)).toBe('[B, T, 128]');
    expect(compactEdgeLabel('float32 [B, T, 128]', 0.2)).toBeUndefined();
  });
});
