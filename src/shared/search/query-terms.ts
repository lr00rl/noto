/**
 * Words in a search box, the way fzf and ripgrep split a query.
 *
 * A space is AND, not a character that has to appear in the filename. That is
 * why `open jobs` finds `openjobs.md` in the Typora plugin when fzf is on the
 * path, and why a quoted phrase stays one term. Shared so name ranking and
 * content search cannot drift into two grammars for the same box.
 */

/** One Han, kana or Hangul syllable, for splitting a mixed term. */
const CJK_CHAR = /[\u3040-\u309f\u30a0-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]/;

/** Latin letters and digits, the other half of a mixed term. */
const LATIN_DIGIT = /[a-zA-Z0-9]/;

/** Latin/digit runs and CJK runs in a term that mixes both. */
const MIXED_RUN = /[a-zA-Z0-9]+|[\u3040-\u309f\u30a0-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]+/g;

/**
 * Split `open工作` into `open` and `工作`, leave `你好世界` and `auth-redis` whole.
 *
 * Only terms that contain both Latin/digit and CJK are split, at the boundary
 * between those runs; punctuation between them is not its own term.
 */
function splitMixedTerm(term: string): string[] {
  if (!LATIN_DIGIT.test(term) || !CJK_CHAR.test(term)) return [term];
  const parts: string[] = [];
  let match: RegExpExecArray | null;
  MIXED_RUN.lastIndex = 0;
  while ((match = MIXED_RUN.exec(term)) !== null) parts.push(match[0]);
  return parts.length > 0 ? parts : [term];
}

/** Whitespace-delimited terms, with `"quoted phrases"` kept whole. */
export function tokenizeQuery(query: string): string[] {
  const seen = new Set<string>();
  const terms: string[] = [];
  const token = /"([^"]+)"|(\S+)/g;
  let match: RegExpExecArray | null;
  const input = query.trim();
  while ((match = token.exec(input)) !== null) {
    const quoted = match[1] !== undefined;
    const term = (match[1] ?? match[2] ?? '').trim();
    if (term.length === 0) continue;
    const pieces = quoted ? [term] : splitMixedTerm(term);
    for (const piece of pieces) {
      const key = piece.toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      terms.push(piece);
    }
  }
  return terms;
}
