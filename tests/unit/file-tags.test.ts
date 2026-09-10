import { describe, expect, it } from 'vitest';
import {
  frontmatterBody,
  parseTagsFromFrontmatter,
  parseTagsFromMarkdown,
  splitFlowList,
  tagsEqual,
  unquoteYamlScalar,
} from '../../src/shared/tags/parse';
import { buildTagIndex, notesForTag, tagsForPath } from '../../src/main/workspace/tag-index';

describe('frontmatter tag parsing', () => {
  it('reads a flow list and a block list the vault already uses', () => {
    expect(parseTagsFromMarkdown('---\ntags: [agent, cloudflare, 沙箱]\n---\n\nBody.\n'))
      .toEqual(['agent', 'cloudflare', '沙箱']);
    expect(parseTagsFromMarkdown('---\ntags:\n  - Matter\n  - 智能家居\n---\n\nBody.\n'))
      .toEqual(['Matter', '智能家居']);
  });

  it('returns nothing when there is no frontmatter or no tags key', () => {
    expect(parseTagsFromMarkdown('# Just a heading\n')).toEqual([]);
    expect(parseTagsFromMarkdown('---\ntitle: Only\n---\n')).toEqual([]);
    expect(frontmatterBody('nope')).toBeNull();
  });

  it('de-duplicates case-insensitively and keeps the first spelling', () => {
    expect(parseTagsFromFrontmatter('tags: [AI, ai, Ai]')).toEqual(['AI']);
  });

  it('strips quotes and ignores an inline comment', () => {
    expect(unquoteYamlScalar('"smart home"')).toBe('smart home');
    expect(splitFlowList('[a, "b, c", d]')).toEqual(['a', 'b, c', 'd']);
    expect(parseTagsFromFrontmatter('tags: [agent] # note')).toEqual(['agent']);
  });

  it('compares tags the way the index does', () => {
    expect(tagsEqual('AI', 'ai')).toBe(true);
    expect(tagsEqual('AI', 'ML')).toBe(false);
  });
});

describe('the vault tag index', () => {
  it('groups notes by the tags in their frontmatter', async () => {
    const files = new Map<string, string>([
      ['/vault/a.md', '---\ntags: [agent, linux]\n---\n\nA.\n'],
      ['/vault/b.md', '---\ntags:\n  - Linux\n  - rust\n---\n\nB.\n'],
      ['/vault/c.md', '# No tags\n'],
      ['/vault/d.txt', 'tags: [ignored]\n'],
    ]);
    const index = await buildTagIndex(
      [
        { path: '/vault/a.md', name: 'a.md', relativePath: 'a.md' },
        { path: '/vault/b.md', name: 'b.md', relativePath: 'b.md' },
        { path: '/vault/c.md', name: 'c.md', relativePath: 'c.md' },
        { path: '/vault/d.txt', name: 'd.txt', relativePath: 'd.txt' },
      ],
      { read: async (path) => files.get(path) ?? '' },
    );
    expect(index.tags.map((entry) => entry.tag)).toEqual(['agent', 'linux', 'rust']);
    expect(notesForTag(index, 'Linux').map((note) => note.name)).toEqual(['a.md', 'b.md']);
    expect(tagsForPath(index, '/vault/b.md')).toEqual(['linux', 'rust']);
    expect(index.truncated).toBe(false);
  });

  it('stops when the budget runs out and says so', async () => {
    let reads = 0;
    const index = await buildTagIndex(
      [
        { path: '/vault/a.md', name: 'a.md', relativePath: 'a.md' },
        { path: '/vault/b.md', name: 'b.md', relativePath: 'b.md' },
      ],
      {
        budgetMs: 0,
        now: () => (reads === 0 ? 0 : 10),
        read: async () => {
          reads += 1;
          return '---\ntags: [x]\n---\n';
        },
      },
    );
    expect(index.truncated).toBe(true);
  });
});
