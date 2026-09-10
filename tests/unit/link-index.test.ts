import { describe, expect, it } from 'vitest';
import { buildLinkIndex, neighbourhood } from '../../src/main/workspace/link-index';
import type { WorkspaceIndexEntryV1 } from '../../src/shared/workspace/v1/contracts';

const entry = (relativePath: string): WorkspaceIndexEntryV1 => ({
  path: `/vault/${relativePath}`,
  name: relativePath.split('/').pop() ?? relativePath,
  relativePath,
});

describe('the explicit link index', () => {
  it('resolves wiki links and inverts them into backlinks', async () => {
    const files: Record<string, string> = {
      'topics/embedding.md': '# Embedding\n\nSee [[batch]].\n',
      'topics/batch.md': '# Batch\n\nThe batch note.\n',
      'journal/monday.md': '# Monday\n\nSee [[topics/embedding]].\n',
    };
    const entries = Object.keys(files).map(entry);
    const index = await buildLinkIndex(entries, {
      read: async (path) => files[path.replace('/vault/', '')],
    });
    const pathOf = (relativePath: string) => `/vault/${relativePath}`;
    const embedding = neighbourhood(index, 'topics/embedding.md', pathOf);
    expect(embedding?.outgoing.map((item) => item.relativePath)).toEqual(['topics/batch.md']);
    expect(embedding?.outgoing[0].title).toBe('Batch');
    expect(embedding?.backlinks.map((item) => item.title)).toEqual(['Monday']);

    const batch = neighbourhood(index, 'topics/batch.md', pathOf);
    expect(batch?.backlinks.map((item) => item.relativePath)).toEqual(['topics/embedding.md']);
    expect(batch?.outgoing).toEqual([]);
  });
});
