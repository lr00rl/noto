import { describe, expect, it } from 'vitest';
import { blockToMarkdown } from '../../src/shared/markdown/v3/pm/to-mdast';
import { docFromSpans, blockFromSpan } from '../../src/shared/markdown/v3/pm/from-mdast';
import {
  cloneParsedSpans,
  parseDocumentSpansSync,
} from '../../src/renderer/editor/noto/parse-document';

/**
 * The open-path Worker posts BlockSpan values, mdast trees included, through
 * structured clone. If that round trip dropped anything docFromSpans needs,
 * a large open would build the wrong document with no second chance to parse.
 */
describe('open-path parse payloads survive structured clone', () => {
  const sample = [
    '---',
    'title: Note',
    '---',
    '',
    '# Heading',
    '',
    'A paragraph with **bold** and a [link](https://example.com).',
    '',
    '- bullet',
    '- [ ] task',
    '',
    '```ts',
    'const n = 1;',
    '```',
    '',
    '| a | b |',
    '| - | - |',
    '| 1 | 2 |',
    '',
    '$$',
    'x = 1',
    '$$',
    '',
    '> quote',
    '',
    '***',
    '',
    '[id]: https://example.com',
    '',
    '[^fn]: a footnote',
    '',
  ].join('\n');

  it('builds the same ProseMirror document after a structured clone', () => {
    const original = parseDocumentSpansSync(sample);
    const cloned = cloneParsedSpans(original);
    expect(cloned).toHaveLength(original.length);

    for (let index = 0; index < original.length; index += 1) {
      const left = original[index];
      const right = cloned[index];
      expect(right.kind).toBe(left.kind);
      expect(right.start).toBe(left.start);
      expect(right.end).toBe(left.end);
      expect(right.markdown).toBe(left.markdown);
      expect(right.semanticKey).toBe(left.semanticKey);
      expect(blockToMarkdown(blockFromSpan(right))).toBe(blockToMarkdown(blockFromSpan(left)));
    }

    const fromOriginal = docFromSpans(original);
    const fromCloned = docFromSpans(cloned);
    expect(fromCloned.toJSON()).toEqual(fromOriginal.toJSON());
  });

  it('keeps fence versus indent hints that mdast alone would drop', () => {
    const fenced = parseDocumentSpansSync('```\ncode\n```\n')[0];
    const indented = parseDocumentSpansSync('    indented\n')[0];
    expect(fenced.kind).toBe('fenced-code');
    expect(indented.kind).toBe('indented-code');
    expect(cloneParsedSpans([fenced])[0].kind).toBe('fenced-code');
    expect(cloneParsedSpans([indented])[0].kind).toBe('indented-code');
  });
});
