/**
 * Port handle with live compatibility highlighting while dragging connections.
 */

import type { CSSProperties } from 'react';
import { Handle, Position } from '@xyflow/react';
import type { PortDefinition } from '../api/contracts';
import { useConnectionAuthoring } from './ConnectionAuthoringContext';
import { usePortConnectionState, withConnectionHighlight } from './portConnectionState';

interface BlueprintPortHandleProps {
  nodeId: string;
  port: PortDefinition;
  isOutput: boolean;
  baseStyle: CSSProperties;
}

export function BlueprintPortHandle({
  nodeId,
  port,
  isOutput,
  baseStyle,
}: BlueprintPortHandleProps) {
  const { validate } = useConnectionAuthoring();
  const portState = usePortConnectionState(nodeId, port.id, isOutput, validate);

  const title =
    portState === 'compatible'
      ? 'Compatible connection'
      : portState === 'incompatible'
        ? 'Incompatible connection'
        : portState === 'active'
          ? 'Dragging from this output'
          : port.display_name;

  return (
    <Handle
      type={isOutput ? 'source' : 'target'}
      position={isOutput ? Position.Right : Position.Left}
      id={port.id}
      style={withConnectionHighlight(baseStyle, portState)}
      title={title}
    />
  );
}
