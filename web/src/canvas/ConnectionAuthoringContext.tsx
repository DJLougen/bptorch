/**
 * Context for live connection compatibility while dragging wires on the canvas.
 */

import type { Connection } from '@xyflow/react';
import { createContext, useContext } from 'react';
import type { ConnectionValidationResult } from './connectionValidation';

export interface ConnectionAuthoringContextValue {
  validate: (connection: Connection) => ConnectionValidationResult;
}

export const ConnectionAuthoringContext = createContext<ConnectionAuthoringContextValue>({
  validate: () => ({ valid: false, reason: 'connection validation is unavailable' }),
});

export function useConnectionAuthoring(): ConnectionAuthoringContextValue {
  return useContext(ConnectionAuthoringContext);
}
