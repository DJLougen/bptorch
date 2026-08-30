#!/usr/bin/env python3
"""Generate JSON Schema and TypeScript contracts from canonical Pydantic models."""

import json
from pathlib import Path
from neural_blueprint.ir.models import Project
from neural_blueprint.registry.registry import global_registry

ROOT_DIR = Path(__file__).parent.parent
JSON_SCHEMA_DIR = ROOT_DIR / "packages" / "contracts" / "json-schema"
TS_DIR = ROOT_DIR / "packages" / "contracts" / "generated-typescript"
WEB_API_DIR = ROOT_DIR / "web" / "src" / "api"


def generate():
    JSON_SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    TS_DIR.mkdir(parents=True, exist_ok=True)
    WEB_API_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Generate project.schema.json
    project_schema = Project.model_json_schema()
    schema_path = JSON_SCHEMA_DIR / "project.schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(project_schema, f, indent=2)
    print(f"Wrote JSON Schema: {schema_path}")

    # 2. Generate node_catalog.json
    catalog = global_registry.export_catalog()
    catalog_path = JSON_SCHEMA_DIR / "node_catalog.json"
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
    print(f"Wrote Node Catalog: {catalog_path}")

    # 3. Generate TypeScript contract file
    ts_content = """/* eslint-disable */
/**
 * Auto-generated Canonical IR TypeScript Contracts.
 * Generated from Neural Blueprint Studio Pydantic models.
 */

export type ExpressionOp = 'add' | 'subtract' | 'multiply' | 'integer_divide' | 'minimum' | 'maximum';

export interface ConfigRefValue {
  kind: 'config_ref';
  key: string;
}

export interface ParentPropertyRefValue {
  kind: 'parent_property_ref';
  property_name: string;
}

export interface LiteralValue {
  kind: 'literal';
  value: unknown;
}

export interface SafeExpression {
  op: ExpressionOp;
  left: number | string | ConfigRefValue | ParentPropertyRefValue | SafeExpression;
  right: number | string | ConfigRefValue | ParentPropertyRefValue | SafeExpression;
}

export interface ExpressionValue {
  kind: 'expression';
  expression: SafeExpression;
}

export type PropertyValue = ConfigRefValue | ParentPropertyRefValue | LiteralValue | ExpressionValue | unknown;

export type ShapeDim =
  | { kind: 'symbol'; name: string }
  | { kind: 'config_ref'; key: string }
  | { kind: 'literal'; value: number }
  | { kind: 'unknown' };

export interface TensorType {
  dtype_family?: 'floating' | 'integer' | 'boolean' | 'any';
  rank?: number | null;
}

export interface TensorSpec {
  dtype: string;
  shape: ShapeDim[];
  device?: string;
}

export type PortKind = 'exec' | 'data';

export interface PortDefinition {
  id: string;
  display_name: string;
  direction: 'input' | 'output';
  kind?: PortKind;
  required?: boolean;
  multiplicity?: 'single' | 'multiple';
  tensor_type?: TensorType | null;
  default_shape?: ShapeDim[] | null;
  description?: string | null;
}
export interface PortReference {
  node_id: string;
  port_id: string;
}

export interface Edge {
  id: string;
  source: PortReference;
  target: PortReference;
}

export interface NodeMetadata {
  breakpoint?: boolean;
  disabled?: boolean;
  notes?: string | null;
}

export interface NodeInstance {
  id: string;
  definition_id: string;
  display_name: string;
  properties: Record<string, unknown>;
  metadata: NodeMetadata;
}

export interface GraphInterface {
  inputs: PortDefinition[];
  outputs: PortDefinition[];
}

export interface VariableDefinition {
  id: string;
  name: string;
  type: string;
  default_value?: unknown;
}

export type GraphKind =
  | 'root'
  | 'module'
  | 'repeat'
  | 'architecture'
  | 'training_event'
  | 'function'
  | 'macro';

export interface GraphDefinition {
  id: string;
  name: string;
  kind: GraphKind;
  interface: GraphInterface;
  variables?: VariableDefinition[];
  nodes: NodeInstance[];
  edges: Edge[];
  repeat_count?: number | ConfigRefValue | null;
  target_graph_id?: string | null;
}
export interface WeightBindingEndpoint {
  node_id: string;
  parameter: string;
}

export interface WeightBinding {
  source: WeightBindingEndpoint;
  target: WeightBindingEndpoint;
  mode: 'share' | 'copy';
}

export interface TrainingConfig {
  device: string;
  precision: 'fp32' | 'fp16' | 'bf16' | 'fp8';
  ddp_enabled: boolean;
  seed: number;
  max_epochs: number;
  max_steps?: number | null;
  learning_rate: number;
  weight_decay: number;
  grad_accum_steps: number;
  grad_clip: number;
  batch_size: number;
  checkpoint_interval: number;
  eval_interval: number;
}

export interface ModelDefinition {
  root_graph_id: string;
  config: Record<string, unknown>;
  training?: TrainingConfig;
  graphs: Record<string, GraphDefinition>;
  weight_bindings: WeightBinding[];
}

export interface NodePosition {
  x: number;
  y: number;
}

export interface Viewport {
  x: number;
  y: number;
  zoom: number;
}

export interface UIState {
  graph_viewports: Record<string, Viewport>;
  node_positions: Record<string, Record<string, NodePosition>>;
  open_graph_id: string;
}

export interface ProjectMetadata {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
}

export interface Project {
  schema_version: number;
  project: ProjectMetadata;
  model: ModelDefinition;
  ui: UIState;
}

export interface NodeDefinitionSummary {
  type_id: string;
  version: number;
  display_name: string;
  category: string;
  description: string;
  icon?: string | null;
  is_composite: boolean;
  property_schema: Record<string, unknown>;
  default_inputs: PortDefinition[];
  default_outputs: PortDefinition[];
}

export interface Diagnostic {
  code: string;
  severity: 'error' | 'warning' | 'info';
  message: string;
  node_id?: string | null;
  port_id?: string | null;
  edge_id?: string | null;
  expected?: string | null;
  actual?: string | null;
  suggestions: string[];
}

export interface ParameterSummary {
  total_unique: number;
  trainable: number;
  frozen: number;
  shared_references: number;
  breakdown_by_node?: Record<string, unknown>;
}

export interface ValidationResponse {
  valid: boolean;
  graph_hash: string;
  resolved_shapes: Record<string, Record<string, TensorSpec>>;
  parameter_summary: ParameterSummary;
  diagnostics: Diagnostic[];
}

export interface CompilationResponse {
  session_id: string;
  graph_hash: string;
  device: string;
  parameter_summary: ParameterSummary;
}

export interface TensorSummary {
  shape: number[];
  dtype: string;
  device: string;
  numel: number;
  mean?: number | null;
  std?: number | null;
  min?: number | null;
  max?: number | null;
  l2_norm?: number | null;
  zero_fraction?: number | null;
  nan_count?: number;
  pos_inf_count?: number;
  neg_inf_count?: number;
  sample_values?: unknown[];
}
export interface TrainingMetrics {
  epoch: number;
  step: number;
  loss: number;
  avg_loss?: number | null;
  learning_rate: number;
  grad_norm: number;
  grad_status: string;
  tokens_per_sec: number;
  step_time_ms: number;
  vram_mb: number;
  val_loss?: number | null;
  val_accuracy?: number | null;
  best_loss?: number | null;
  custom_metrics?: Record<string, number>;
}

export type TraceEventType =
  | 'run_started'
  | 'node_started'
  | 'node_finished'
  | 'node_paused'
  | 'node_failed'
  | 'run_finished'
  | 'run_cancelled'
  | 'train_started'
  | 'epoch_started'
  | 'batch_ended'
  | 'validation_finished'
  | 'checkpoint_saved'
  | 'anomaly_detected'
  | 'train_finished'
  | 'hyperparameter_updated';

export interface TraceEvent {
  sequence: number;
  event: TraceEventType;
  session_id: string;
  graph_hash: string;
  node_path: string;
  timestamp_ns: number;
  duration_ns?: number;
  outputs?: Record<string, TensorSummary>;
  inputs?: Record<string, TensorSummary>;
  metrics?: TrainingMetrics | null;
  error?: string | null;
}

export interface TestCaseResult {
  id: string;
  name: string;
  status: 'passed' | 'failed' | 'skipped';
  duration_ms: number;
  description: string;
  metrics: Record<string, unknown>;
  error?: string | null;
  details?: string | null;
}

export interface TestSuiteResult {
  suite_id: string;
  project_name: string;
  total: number;
  passed: number;
  failed: number;
  duration_ms: number;
  cases: TestCaseResult[];
}

export interface AvailableTestInfo {
  id: string;
  name: string;
  description: string;
}

export interface TestRunRequest {
  project: Project;
  enabled_tests?: string[] | null;
}
"""

    ts_path = TS_DIR / "index.ts"
    with open(ts_path, "w", encoding="utf-8") as f:
        f.write(ts_content)
    print(f"Wrote TypeScript contracts: {ts_path}")

    # Copy to web/src/api/contracts.ts
    web_ts_path = WEB_API_DIR / "contracts.ts"
    with open(web_ts_path, "w", encoding="utf-8") as f:
        f.write(ts_content)
    print(f"Wrote Web TypeScript contracts: {web_ts_path}")


if __name__ == "__main__":
    generate()
