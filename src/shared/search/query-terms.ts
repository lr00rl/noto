/**
 * Words in a search box, the way fzf and ripgrep split a query.
 *
 * A space is AND, not a character that has to appear in the filename. That is
 * why `open jobs` finds `openjobs.md` in the Typora plugin when fzf is on the
 * path, and why a quoted phrase stays one term. Shared so name ranking and
 * content search cannot drift into two grammars for the same box.
 */

/** Whitespace-delimited terms, with `"quoted phrases"` kept whole. */
export function tokenizeQuery(query: string): string[] {
  const seen = new Set<string>();
  const terms: string[] = [];
  const token = /"([^"]+)"|(\S+)/g;
  let match: RegExpExecArray | null;
  const input = query.trim();
  while ((match = token.exec(input)) !== null) {
    const term = (match[1] ?? match[2] ?? '').trim();
    if (term.length === 0) continue;
    const key = term.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    terms.push(term);
  }
  return terms;
}
