/**
 * Validation state and diagnostics Zustand store.
 */

import { create } from 'zustand';
import { ApiClient } from '../api/client';
import {
  Diagnostic,
  ParameterSummary,
  Project,
  TensorSpec,
} from '../api/contracts';

interface ValidationState {
  isValid: boolean;
  graphHash: string;
  diagnostics: Diagnostic[];
  resolvedShapes: Record<string, Record<string, TensorSpec>>;
  parameterSummary: ParameterSummary;
  isValidating: boolean;
  lastValidatedTimestamp: number;

  runValidation: (project: Project) => Promise<void>;
}

export const useValidationStore = create<ValidationState>((set) => ({
  isValid: true,
  graphHash: '',
  diagnostics: [],
  resolvedShapes: {},
  parameterSummary: {
    total_unique: 0,
    trainable: 0,
    frozen: 0,
    shared_references: 0,
  },
  isValidating: false,
  lastValidatedTimestamp: 0,

  runValidation: async (project: Project) => {
    set({ isValidating: true });
    try {
      const response = await ApiClient.validateGraph(project);
      set({
        isValid: response.valid,
        graphHash: response.graph_hash,
        diagnostics: response.diagnostics,
        resolvedShapes: response.resolved_shapes,
        parameterSummary: response.parameter_summary,
        isValidating: false,
        lastValidatedTimestamp: Date.now(),
      });
    } catch (err) {
      set({
        isValid: false,
        diagnostics: [
          {
            code: 'E_VALIDATION_NETWORK',
            severity: 'error',
            message: err instanceof Error ? err.message : 'Validation network request failed',
            suggestions: ['Ensure backend server is running on :8000'],
          },
        ],
        isValidating: false,
      });
    }
  },
}));
