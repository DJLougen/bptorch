/**
 * Zoom-adaptive edge label formatting helper.
 */

export function compactEdgeLabel(
  label: string | undefined | null,
  zoom: number
): string | undefined {
  if (!label || typeof label !== 'string' || label.trim() === '') {
    return undefined;
  }

  if (zoom < 0.35) {
    return undefined;
  }

  if (zoom < 0.55) {
    const match = label.match(/^[a-zA-Z0-9_]+\s+(\[.*\])$/);
    if (match) {
      return match[1];
    }
    const spaceIdx = label.indexOf(' ');
    if (spaceIdx !== -1 && label.includes('[')) {
      return label.slice(spaceIdx + 1);
    }
    return label;
  }

  return label;
}
