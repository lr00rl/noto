import { describe, expect, it } from 'vitest';
import { blockSpansFromWire, splitBlocks } from '../../src/shared/markdown/v3/blocks';
import { blockToMarkdown } from '../../src/shared/markdown/v3/pm/to-mdast';
import { blockFromSpan, docFromSpans } from '../../src/shared/markdown/v3/pm/from-mdast';
import { parseDocument, toWire } from '../../src/shared/markdown/v3/document';
import { cloneParsedSpans } from '../../src/renderer/editor/noto/parse-document';

/**
 * Open ships mdast nodes on the wire so the renderer can skip its Worker parse.
 * These checks are the safety net: wire nodes must build the same ProseMirror
 * document as a fresh splitBlocks, including after the structured clone IPC
 * applies.
 */
describe('blockSpansFromWire', () => {
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
    '    indented',
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

  it('returns null when nodes were not shipped', () => {
    const parsed = parseDocument(Buffer.from(sample, 'utf8'));
    if (parsed.status !== 'parsed') throw new Error(parsed.message);
    const wire = { ...toWire(parsed.document), nodes: null };
    expect(blockSpansFromWire(wire)).toBeNull();
  });

  it('builds the same ProseMirror document as a fresh splitBlocks', () => {
    const parsed = parseDocument(Buffer.from(sample, 'utf8'));
    if (parsed.status !== 'parsed') throw new Error(parsed.message);
    const wire = toWire(parsed.document);
    expect(wire.nodes).not.toBeNull();
    expect(wire.nodes).toHaveLength(wire.spans.length);

    const fromWire = blockSpansFromWire(wire);
    expect(fromWire).not.toBeNull();
    const fresh = splitBlocks(sample).spans;

    expect(fromWire!).toHaveLength(fresh.length);
    for (let index = 0; index < fresh.length; index += 1) {
      const left = fresh[index]!;
      const right = fromWire![index]!;
      expect(right.kind).toBe(left.kind);
      expect(right.start).toBe(left.start);
      expect(right.end).toBe(left.end);
      expect(right.markdown).toBe(left.markdown);
      expect(right.semanticKey).toBe(left.semanticKey);
      expect(blockToMarkdown(blockFromSpan(right))).toBe(blockToMarkdown(blockFromSpan(left)));
    }

    expect(docFromSpans(fromWire!).toJSON()).toEqual(docFromSpans(fresh).toJSON());
  });

  it('survives structured clone the way Electron IPC does', () => {
    const parsed = parseDocument(Buffer.from(sample, 'utf8'));
    if (parsed.status !== 'parsed') throw new Error(parsed.message);
    const cloned = structuredClone(toWire(parsed.document));
    const spans = blockSpansFromWire(cloned);
    expect(spans).not.toBeNull();
    const again = cloneParsedSpans(spans!);
    expect(docFromSpans(again).toJSON()).toEqual(docFromSpans(spans!).toJSON());
  });
});
