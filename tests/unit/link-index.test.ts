import { describe, expect, it } from 'vitest';
import { buildLinkIndex, neighbourhood, patchLinkIndex } from '../../src/main/workspace/link-index';
import {
  buildWikiLookup,
  wikiCandidates,
  wikiCandidatesFromLookup,
  type WikiCandidate,
} from '../../src/shared/markdown/wiki-target';
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

  it('patches one note without rereading the rest', async () => {
    const files: Record<string, string> = {
      'topics/embedding.md': '# Embedding\n\nNo links yet.\n',
      'topics/batch.md': '# Batch\n\nThe batch note.\n',
    };
    const entries = Object.keys(files).map(entry);
    const index = await buildLinkIndex(entries, {
      read: async (path) => files[path.replace('/vault/', '')],
    });
    const pathOf = (relativePath: string) => `/vault/${relativePath}`;

    expect(index.outgoing.get('topics/embedding.md')).toEqual([]);
    expect(index.incoming.get('topics/batch.md')).toEqual([]);

    patchLinkIndex(index, 'topics/embedding.md', '# Embedding\n\nSee [[batch]].\n', entries);
    expect(index.outgoing.get('topics/embedding.md')).toEqual(['topics/batch.md']);
    expect(index.incoming.get('topics/batch.md')).toEqual(['topics/embedding.md']);
    expect(neighbourhood(index, 'topics/embedding.md', pathOf)?.outgoing.map((item) => item.relativePath))
      .toEqual(['topics/batch.md']);
    expect(neighbourhood(index, 'topics/batch.md', pathOf)?.backlinks.map((item) => item.relativePath))
      .toEqual(['topics/embedding.md']);

    patchLinkIndex(index, 'topics/embedding.md', '# Embedding\n\nNo links again.\n', entries);
    expect(index.outgoing.get('topics/embedding.md')).toEqual([]);
    expect(index.incoming.get('topics/batch.md')).toEqual([]);
    expect(neighbourhood(index, 'topics/batch.md', pathOf)?.backlinks).toEqual([]);
  });
});

describe('wiki lookup from a built index', () => {
  const asWiki = (relativePath: string): WikiCandidate => ({
    path: `/vault/${relativePath}`,
    relativePath,
    name: relativePath.split('/').at(-1)!,
  });

  const vault = [
    asWiki('E000_Works/Openjobs-ai/00_索引.md'),
    asWiki('E000_Works/Openjobs-ai/vpn网络搭建规划/00_索引.md'),
    asWiki('E000_Works/Openjobs-ai/数据部门/00_索引.md'),
    asWiki('A000_Theoretical_Knowledge/00_索引.md'),
    asWiki('00_索引.md'),
  ];

  it('keeps the same candidate order as building on every call', () => {
    const lookup = buildWikiLookup(vault);
    const from = 'E000_Works/Openjobs-ai/00_索引.md';
    for (const target of ['vpn网络搭建规划/00_索引', '00_索引', 'nowhere']) {
      expect(wikiCandidatesFromLookup(target, from, lookup).map((item) => item.relativePath))
        .toEqual(wikiCandidates(target, from, vault).map((item) => item.relativePath));
    }
  });
});
