/**
 * What a content search is actually looking for.
 *
 * The find bar and the sidebar already have a regex switch. Quick open does
 * not: it is one box, and the Typora plugin this is ported from used ripgrep's
 * default of "every word must appear". `re:` is the way to opt into an
 * expression without a second control. Quoted phrases stay exact; everything
 * else is AND of literals. The rail's regex flag still wins when it is on, so
 * the two surfaces do not disagree about the same switches.
 */

import { tokenizeQuery } from './query-terms';
import type { SearchFlags } from './pattern';

export interface ContentNeedles {
  /** One expression, or several literals that all have to appear. */
  readonly regex: boolean;
  readonly needles: readonly string[];
}

const REGEX_PREFIX = /^re:\s*(.*)$/is;

/**
 * The needles a content query names, after `type:` and `scope:` have already
 * been taken out.
 */
export function parseContentNeedles(query: string, flags: SearchFlags): ContentNeedles {
  const trimmed = query.trim();
  if (flags.regex) return { regex: true, needles: trimmed.length === 0 ? [] : [trimmed] };
  const prefixed = REGEX_PREFIX.exec(trimmed);
  if (prefixed) {
    const body = prefixed[1].trim();
    return { regex: true, needles: body.length === 0 ? [] : [body] };
  }
  return { regex: false, needles: tokenizeQuery(trimmed) };
}
