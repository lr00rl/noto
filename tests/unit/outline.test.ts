import { describe, expect, it } from 'vitest';
import { outlineFromDocument, outlineOf } from '../../src/renderer/outline';
import { parseDocument, toWire } from '../../src/shared/markdown/v3/document';

function wireOf(source: string) {
  const result = parseDocument(Buffer.from(source, 'utf8'));
  if (result.status !== 'parsed') throw new Error(result.message);
  return toWire(result.document);
}

describe('document outline', () => {
  it('lists headings with their depth and block position', () => {
    const source = '# Title\n\nBody.\n\n## Section\n\nMore.\n\n### Detail\n';
    expect(outlineOf(source)).toEqual([
      { blockIndex: 0, depth: 1, text: 'Title' },
      { blockIndex: 2, depth: 2, text: 'Section' },
      { blockIndex: 4, depth: 3, text: 'Detail' },
    ]);
    expect(outlineFromDocument(wireOf(source))).toEqual(outlineOf(source));
  });

  it('strips the marker run without eating heading text', () => {
    expect(outlineOf('### C# and F#\n')[0].text).toBe('C# and F#');
    expect(outlineOf('## Closed heading ##\n')[0].text).toBe('Closed heading');
    expect(outlineFromDocument(wireOf('### C# and F#\n'))[0].text).toBe('C# and F#');
    expect(outlineFromDocument(wireOf('## Closed heading ##\n'))[0].text).toBe('Closed heading');
  });

  it('understands setext headings', () => {
    expect(outlineOf('Title\n=====\n\nSection\n-------\n')).toEqual([
      { blockIndex: 0, depth: 1, text: 'Title' },
      { blockIndex: 1, depth: 2, text: 'Section' },
    ]);
    expect(outlineFromDocument(wireOf('Title\n=====\n\nSection\n-------\n'))).toEqual(
      outlineOf('Title\n=====\n\nSection\n-------\n'),
    );
  });

  it('ignores hashes that are not headings', () => {
    // A fence keeps its contents out of the outline, and so does a paragraph
    // that merely mentions a hash.
    expect(outlineOf('```\n# not a heading\n```\n\nA # in prose.\n')).toEqual([]);
    expect(outlineFromDocument(wireOf('```\n# not a heading\n```\n\nA # in prose.\n'))).toEqual([]);
  });

  it('shows a heading that has no text yet rather than hiding it', () => {
    // Typing `# ` creates the heading before its text exists. The outline
    // mirrors the document, so the entry appears immediately and fills in.
    expect(outlineOf('#\u0020\n')).toEqual([{ blockIndex: 0, depth: 1, text: 'Untitled heading' }]);
    expect(outlineOf('## \n')).toEqual([{ blockIndex: 0, depth: 2, text: 'Untitled heading' }]);
    expect(outlineFromDocument(wireOf('#\u0020\n'))).toEqual(outlineOf('#\u0020\n'));
    expect(outlineFromDocument(wireOf('## \n'))).toEqual(outlineOf('## \n'));
  });

  it('returns nothing for a document without headings', () => {
    expect(outlineOf('Just a paragraph.\n\n- and a list\n')).toEqual([]);
    expect(outlineFromDocument(wireOf('Just a paragraph.\n\n- and a list\n'))).toEqual([]);
  });

  it('tracks block index across non-heading blocks so navigation lands right', () => {
    const source = '| a |\n| --- |\n| 1 |\n\n# After a table\n';
    expect(outlineOf(source)).toEqual([{ blockIndex: 1, depth: 1, text: 'After a table' }]);
    expect(outlineFromDocument(wireOf(source))).toEqual(outlineOf(source));
  });
});
