import { describe, expect, it } from 'vitest';
import { PLAIN_FLAGS } from '../../src/shared/search/pattern';
import { findBarFromNeedles, parseContentNeedles } from '../../src/shared/search/content-query';
import { tokenizeQuery } from '../../src/shared/search/query-terms';

describe('splitting a query into terms', () => {
  it('splits on spaces and keeps a quoted phrase whole', () => {
    expect(tokenizeQuery('auth redis')).toEqual(['auth', 'redis']);
    expect(tokenizeQuery('keep "auth redis" together')).toEqual(['keep', 'auth redis', 'together']);
  });

  it('drops duplicate terms without regard to case', () => {
    expect(tokenizeQuery('Auth AUTH auth')).toEqual(['Auth']);
  });

  it('splits a Latin and CJK mix into separate terms', () => {
    expect(tokenizeQuery('open工作')).toEqual(['open', '工作']);
  });

  it('keeps a quoted Latin and CJK mix as one term', () => {
    expect(tokenizeQuery('"open工作"')).toEqual(['open工作']);
  });

  it('keeps an all-CJK term whole', () => {
    expect(tokenizeQuery('你好世界')).toEqual(['你好世界']);
  });
});

describe('what a content query is looking for', () => {
  it('leaves a regex switch as one expression', () => {
    expect(parseContentNeedles('auth redis', { ...PLAIN_FLAGS, regex: true }))
      .toEqual({ regex: true, needles: ['auth redis'] });
  });

  it('treats re: as regex when the switch is off', () => {
    expect(parseContentNeedles('re:foo.*bar', PLAIN_FLAGS))
      .toEqual({ regex: true, needles: ['foo.*bar'] });
  });

  it('tokenises everything else', () => {
    expect(parseContentNeedles('auth redis', PLAIN_FLAGS))
      .toEqual({ regex: false, needles: ['auth', 'redis'] });
  });
});

describe('what the find bar should search after a vault hit', () => {
  it('keeps a single literal as a literal', () => {
    expect(findBarFromNeedles({ regex: false, needles: ['auth'] }))
      .toEqual({ query: 'auth', regex: false });
  });

  it('turns AND terms into an expression that finds any of them', () => {
    expect(findBarFromNeedles({ regex: false, needles: ['auth', 'redis'] }))
      .toEqual({ query: 'auth|redis', regex: true });
  });

  it('keeps re: as a regular expression', () => {
    expect(findBarFromNeedles({ regex: true, needles: ['foo.*bar'] }))
      .toEqual({ query: 'foo.*bar', regex: true });
  });
});
