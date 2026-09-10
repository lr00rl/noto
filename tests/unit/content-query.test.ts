import { describe, expect, it } from 'vitest';
import { PLAIN_FLAGS } from '../../src/shared/search/pattern';
import { parseContentNeedles } from '../../src/shared/search/content-query';
import { tokenizeQuery } from '../../src/shared/search/query-terms';

describe('splitting a query into terms', () => {
  it('splits on spaces and keeps a quoted phrase whole', () => {
    expect(tokenizeQuery('auth redis')).toEqual(['auth', 'redis']);
    expect(tokenizeQuery('keep "auth redis" together')).toEqual(['keep', 'auth redis', 'together']);
  });

  it('drops duplicate terms without regard to case', () => {
    expect(tokenizeQuery('Auth AUTH auth')).toEqual(['Auth']);
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
