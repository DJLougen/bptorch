import type { CSSProperties } from 'react';
import { useConnection } from '@xyflow/react';
import type { Connection } from '@xyflow/react';
import type { ConnectionValidationResult } from './connectionValidation';

export type PortConnectionState = 'idle' | 'active' | 'compatible' | 'incompatible';

function resolveHandleId(
  handle: string | { id?: string | null } | null | undefined
): string | null {
  if (typeof handle === 'string') {
    return handle;
  }
  if (handle && typeof handle === 'object' && 'id' in handle) {
    return handle.id ?? null;
  }
  return null;
}

export function usePortConnectionState(
  nodeId: string,
  portId: string,
  isOutput: boolean,
  validate: (connection: Connection) => ConnectionValidationResult
): PortConnectionState {
  const connection = useConnection();

  if (!connection.inProgress) {
    return 'idle';
  }

  const fromHandleId = resolveHandleId(connection.fromHandle);

  if (isOutput) {
    if (connection.fromNode?.id === nodeId && fromHandleId === portId) {
      return 'active';
    }
    return 'idle';
  }

  if (!connection.fromNode?.id || !fromHandleId) {
    return 'idle';
  }

  const result = validate({
    source: connection.fromNode.id,
    sourceHandle: fromHandleId,
    target: nodeId,
    targetHandle: portId,
  });

  return result.valid ? 'compatible' : 'incompatible';
}

export function withConnectionHighlight(
  baseStyle: CSSProperties,
  state: PortConnectionState
): CSSProperties {
  if (state === 'active') {
    return {
      ...baseStyle,
      boxShadow: '0 0 0 3px rgba(56, 189, 248, 0.55)',
      border: '2px solid #38bdf8',
    };
  }
  if (state === 'compatible') {
    return {
      ...baseStyle,
      boxShadow: '0 0 0 3px rgba(34, 197, 94, 0.45)',
      border: '2px solid #22c55e',
    };
  }
  if (state === 'incompatible') {
    return {
      ...baseStyle,
      boxShadow: '0 0 0 3px rgba(239, 68, 68, 0.35)',
      border: '2px solid #ef4444',
      opacity: 0.55,
    };
  }
  return baseStyle;
}
