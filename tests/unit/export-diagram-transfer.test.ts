/**
 * Export lifts mermaid drawings out of their sandboxed frames.
 *
 * The walk itself needs a real DOM and is driven in the packaged tests; what
 * is checked here is that the transfer is actually wired into the serializer
 * that export calls, rather than living in a comment about what should happen.
 */

import { readFile } from 'node:fs/promises';
import { describe, expect, it } from 'vitest';

describe('documentDomToHtml', () => {
  it('transfers diagram SVGs from the live frames into the clone', async () => {
    const source = await readFile(
      new URL('../../src/renderer/editor/noto/clipboard.ts', import.meta.url),
      'utf8',
    );
    expect(source).toContain('transferDiagramDrawings');
    expect(source).toContain('materializeDiagram');
    expect(source).toContain('contentDocument');
    expect(source).toContain('iframe.noto-diagram-frame');
    // And the serializer that export calls actually runs the transfer, rather
    // than only exporting the helpers beside it.
    const body = source.slice(source.indexOf('export function documentDomToHtml'));
    expect(body).toContain('transferDiagramDrawings(root, copy)');
    expect(body).toContain('noto-alert-editing');
  });
});
