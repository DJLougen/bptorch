/**
 * Shared sample gallery manifest types and fetch helpers.
 */

import type { Project } from '../api/contracts';
import { isProject } from '../stores/projectStore';

export interface SampleManifestEntry {
  id: string;
  name: string;
  category: string;
  description: string;
  highlight: string;
  tags: string[];
  difficulty: string;
  path: string;
  node_count?: number;
  graph_count?: number;
}

export interface SamplesManifest {
  count: number;
  title?: string;
  description?: string;
  samples: SampleManifestEntry[];
}

export async function fetchSamplesManifest(): Promise<SamplesManifest> {
  const response = await fetch('/examples/samples.json');
  if (!response.ok) {
    throw new Error(`Failed to load samples manifest (HTTP ${response.status})`);
  }
  return response.json() as Promise<SamplesManifest>;
}

export async function fetchProjectFromPath(path: string): Promise<Project> {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load project (HTTP ${response.status})`);
  }
  const data: unknown = await response.json();
  if (!isProject(data)) {
    throw new Error('Invalid project JSON');
  }
  return data;
}

export function groupSamplesByCategory(
  samples: SampleManifestEntry[] | null | undefined,
): Array<{ category: string; samples: SampleManifestEntry[] }> {
  const grouped = new Map<string, SampleManifestEntry[]>();
  for (const sample of samples ?? []) {
    const bucket = grouped.get(sample.category) ?? [];
    bucket.push(sample);
    grouped.set(sample.category, bucket);
  }
  return [...grouped.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([category, entries]) => ({
      category,
      samples: entries.sort((left, right) => left.name.localeCompare(right.name)),
    }));
}
