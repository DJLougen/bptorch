import { describe, expect, it } from 'vitest';
import type { PortDefinition, TensorSpec } from '../src/api/contracts';
import { areShapesCompatible } from '../src/canvas/connectionValidation';

describe('areShapesCompatible', () => {
  const targetPort: PortDefinition = {
    id: 'input',
    display_name: 'Input',
    direction: 'input',
    default_shape: [
      { kind: 'symbol', name: 'B' },
      { kind: 'literal', value: 16 },
    ],
  };

  it('allows unknown source shapes', () => {
    expect(areShapesCompatible(undefined, targetPort).compatible).toBe(true);
  });

  it('rejects rank mismatches', () => {
    const source: TensorSpec = {
      dtype: 'float32',
      shape: [{ kind: 'symbol', name: 'B' }],
    };
    const result = areShapesCompatible(source, targetPort);
    expect(result.compatible).toBe(false);
    expect(result.reason).toMatch(/ranks do not match/i);
  });

  it('rejects incompatible literal dimensions', () => {
    const source: TensorSpec = {
      dtype: 'float32',
      shape: [
        { kind: 'symbol', name: 'B' },
        { kind: 'literal', value: 32 },
      ],
    };
    const result = areShapesCompatible(source, targetPort);
    expect(result.compatible).toBe(false);
    expect(result.reason).toMatch(/shape mismatch/i);
  });

  it('accepts matching symbolic and literal dimensions', () => {
    const source: TensorSpec = {
      dtype: 'float32',
      shape: [
        { kind: 'symbol', name: 'B' },
        { kind: 'literal', value: 16 },
      ],
    };
    expect(areShapesCompatible(source, targetPort).compatible).toBe(true);
  });
});
