import { describe, expect, it } from 'vitest';
import { slashToken } from '../../src/renderer/editor/noto/slash-query';

describe('the slash token a block is holding', () => {
  it('is the letters after a slash at the start of the block', () => {
    expect(slashToken('paragraph', '/table', 6)).toEqual({ query: 'table', start: 0, end: 6 });
    expect(slashToken('paragraph', '/', 1)).toEqual({ query: '', start: 0, end: 1 });
    expect(slashToken('paragraph', '/h1', 3)).toEqual({ query: 'h1', start: 0, end: 3 });
  });

  it('says nothing when the slash is not the first thing', () => {
    expect(slashToken('paragraph', 'see /path', 9)).toBeNull();
    expect(slashToken('paragraph', '/table ', 7)).toBeNull();
    expect(slashToken('paragraph', '', 0)).toBeNull();
  });

  it('does not open inside a fence, a formula or a table cell', () => {
    expect(slashToken('code_block', '/table', 6)).toBeNull();
    expect(slashToken('math_block', '/table', 6)).toBeNull();
    expect(slashToken('table_cell', '/table', 6)).toBeNull();
    expect(slashToken('source_block', '/table', 6)).toBeNull();
  });
});
