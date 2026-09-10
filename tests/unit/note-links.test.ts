import { describe, expect, it } from 'vitest';
import { extractLinkTargets, titleFromNote } from '../../src/shared/markdown/note-links';

describe('links written in a note', () => {
  it('reads wiki targets and markdown hrefs, and skips images and remotes', () => {
    const found = extractLinkTargets([
      '# Title',
      '',
      'See [[topics/batch|Batch]] and [[batch]].',
      'Also [monday](../journal/monday.md) and [the web](https://example.com).',
      '![shot](./pic.png)',
      '',
      '```md',
      '[[inside a fence]]',
      '```',
      '',
      'And `[[inline code]]`.',
    ].join('\n')).map((item) => item.target);
    expect(found).toEqual(['topics/batch', 'batch', '../journal/monday.md']);
  });
});

describe('a note title', () => {
  it('is the first heading, or the file name', () => {
    expect(titleFromNote('# 微调 embedding 模型\n\nBody.\n', 'embedding.md')).toBe('微调 embedding 模型');
    expect(titleFromNote('No heading here.\n', 'stray.md')).toBe('stray');
  });
});
