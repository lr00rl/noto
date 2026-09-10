import { describe, expect, it } from 'vitest';
import { TAB_MARKER_CLASS, tabRanges } from '../../src/renderer/editor/noto/tab-markers';

describe('finding the tabs in a fence', () => {
  it('marks every tab character and nothing else', () => {
    const text = 'a\tb\t\tc';
    expect(tabRanges(text)).toEqual([
      { from: 1, to: 2 },
      { from: 3, to: 4 },
      { from: 4, to: 5 },
    ]);
  });

  it('finds a tab that opens a line, and one in the middle of code', () => {
    const text = '\tif x:\n    return\ty\n';
    const ranges = tabRanges(text);
    expect(ranges).toHaveLength(2);
    expect(text.slice(ranges[0].from, ranges[0].to)).toBe('\t');
    expect(text.slice(ranges[1].from, ranges[1].to)).toBe('\t');
    expect(ranges[0].from).toBe(0);
    expect(ranges[1].from).toBe(text.indexOf('\t', 1));
  });

  it('gives nothing for a block that uses only spaces', () => {
    expect(tabRanges('def f():\n    return 1\n')).toEqual([]);
  });

  it('gives nothing for an empty block', () => {
    expect(tabRanges('')).toEqual([]);
  });
});

describe('the class the stylesheet paints', () => {
  it('is the name the decorations and the CSS agree on', () => {
    expect(TAB_MARKER_CLASS).toBe('noto-code-tab');
  });
});
