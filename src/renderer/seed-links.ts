/**
 * Neighbours written in the note itself, for when the graph has never met it.
 *
 * Hub MOCs are often `moc: true` and therefore absent from
 * `.note-assistant/graph.json`. The Links rail would otherwise say the graph
 * has not met the note — while the note is a dense map of `[[wiki links]]`.
 * This reads those targets (preferring an index region when one is present)
 * and resolves them against the same index wiki follow uses, so the rail can
 * still offer a "Links to" list without inventing graph rows.
 */

import { wikiCandidates, type WikiCandidate } from './wiki-target';

const WIKI = /\[\[([^\]|#]+)(?:\|([^\]]*))?\]\]/g;
const INDEX_START = /<!--\s*note-assistant:index:start\s*-->/;
const INDEX_END = /<!--\s*note-assistant:index:end\s*-->/;

export interface SeededWikiLink {
  readonly target: string;
  readonly title: string;
}

export interface SeededOutboundLink {
  readonly path: string;
  readonly relativePath: string;
  readonly title: string;
}

/** Wiki targets written in markdown, first occurrence of each target wins. */
export function wikiLinksInMarkdown(markdown: string): SeededWikiLink[] {
  const found: SeededWikiLink[] = [];
  const seen = new Set<string>();
  WIKI.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = WIKI.exec(markdown)) !== null) {
    const target = match[1].trim();
    if (target.length === 0 || seen.has(target)) continue;
    seen.add(target);
    const title = (match[2] ?? '').trim() || target.split('/').pop() || target;
    found.push({ target, title });
  }
  return found;
}

/**
 * The stretch between index markers when present; otherwise the whole note.
 * Related-notes chrome (`note-assistant:start/end`) is left alone — those are
 * issued suggestions, not the hub's own map.
 */
export function markdownForLinkSeed(markdown: string): string {
  const start = INDEX_START.exec(markdown);
  if (!start) return markdown;
  const afterStart = start.index + start[0].length;
  const endMatch = INDEX_END.exec(markdown.slice(afterStart));
  if (!endMatch) return markdown;
  return markdown.slice(afterStart, afterStart + endMatch.index);
}

/**
 * Resolved outbound links for the Links rail, best match per written target.
 * Unresolvable targets are skipped rather than shown as dead rows.
 */
export function seedOutboundLinks(
  markdown: string,
  fromRelativePath: string | null,
  entries: readonly WikiCandidate[],
): SeededOutboundLink[] {
  const seeded: SeededOutboundLink[] = [];
  const seen = new Set<string>();
  for (const link of wikiLinksInMarkdown(markdownForLinkSeed(markdown))) {
    const match = wikiCandidates(link.target, fromRelativePath, entries)[0];
    if (!match || seen.has(match.path)) continue;
    seen.add(match.path);
    seeded.push({
      path: match.path,
      relativePath: match.relativePath,
      title: link.title,
    });
  }
  return seeded;
}
