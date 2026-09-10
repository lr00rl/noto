/**
 * Explicit wiki and markdown links for every note in the open folder.
 *
 * Built the same way content search is: a bounded concurrent read, no
 * external graph file. Cached on the session until the tree changes. Related
 * notes are not computed here; those stay with note-assistant's graph.
 */

import { readFile } from 'node:fs/promises';
import { extractLinkTargets, titleFromNote } from '../../shared/markdown/note-links';
import {
  buildWikiLookup,
  wikiCandidatesFromLookup,
  type WikiCandidate,
} from '../../shared/markdown/wiki-target';
import type { WorkspaceIndexEntryV1 } from '../../shared/workspace/v1/contracts';

const CONCURRENCY = 24;

export interface IndexedLink {
  readonly path: string;
  readonly relativePath: string;
  readonly title: string;
}

export interface NoteNeighbourhood {
  readonly outgoing: readonly IndexedLink[];
  readonly backlinks: readonly IndexedLink[];
}

export interface ExplicitLinkIndex {
  readonly titles: ReadonlyMap<string, string>;
  readonly outgoing: ReadonlyMap<string, readonly string[]>;
  readonly incoming: ReadonlyMap<string, readonly string[]>;
}

export interface LinkIndexOptions {
  readonly read?: (path: string) => Promise<string>;
}

async function pool<T>(items: readonly T[], limit: number, work: (item: T) => Promise<void>): Promise<void> {
  let next = 0;
  const runners = Array.from({ length: Math.min(limit, items.length) }, async () => {
    for (;;) {
      const index = next;
      next += 1;
      if (index >= items.length) return;
      await work(items[index]);
    }
  });
  await Promise.all(runners);
}

const asCandidate = (entry: WorkspaceIndexEntryV1): WikiCandidate => ({
  path: entry.path,
  relativePath: entry.relativePath,
  name: entry.name,
});

/**
 * Resolve every written target in every note, and invert the edges.
 *
 * Unresolved targets are dropped rather than shown as dead rows: a click that
 * cannot open anything is worse than a missing neighbour.
 */
export async function buildLinkIndex(
  entries: readonly WorkspaceIndexEntryV1[],
  options: LinkIndexOptions = {},
): Promise<ExplicitLinkIndex> {
  const read = options.read ?? ((path: string) => readFile(path, 'utf8'));
  const candidates = entries.map(asCandidate);
  const lookup = buildWikiLookup(candidates);
  const titles = new Map<string, string>();
  const outgoing = new Map<string, string[]>();
  const incoming = new Map<string, string[]>();

  for (const entry of entries) {
    titles.set(entry.relativePath, entry.name.replace(/\.(?:md|markdown)$/i, ''));
    outgoing.set(entry.relativePath, []);
    incoming.set(entry.relativePath, []);
  }

  await pool(entries, CONCURRENCY, async (entry) => {
    let text: string;
    try {
      text = await read(entry.path);
    } catch {
      return;
    }
    titles.set(entry.relativePath, titleFromNote(text, entry.name));
    const seen = new Set<string>();
    const targets: string[] = [];
    for (const link of extractLinkTargets(text)) {
      const resolved = wikiCandidatesFromLookup(link.target, entry.relativePath, lookup)[0];
      if (!resolved || resolved.relativePath === entry.relativePath) continue;
      if (seen.has(resolved.relativePath)) continue;
      seen.add(resolved.relativePath);
      targets.push(resolved.relativePath);
    }
    outgoing.set(entry.relativePath, targets);
  });

  for (const [from, targets] of outgoing) {
    for (const to of targets) {
      const inbound = incoming.get(to);
      if (inbound) inbound.push(from);
    }
  }

  return { titles, outgoing, incoming };
}

const resolveOutgoing = (
  text: string,
  relativePath: string,
  lookup: ReturnType<typeof buildWikiLookup>,
): string[] => {
  const seen = new Set<string>();
  const targets: string[] = [];
  for (const link of extractLinkTargets(text)) {
    const resolved = wikiCandidatesFromLookup(link.target, relativePath, lookup)[0];
    if (!resolved || resolved.relativePath === relativePath) continue;
    if (seen.has(resolved.relativePath)) continue;
    seen.add(resolved.relativePath);
    targets.push(resolved.relativePath);
  }
  return targets;
};

/**
 * Refresh one note's outgoing edges and repair others' incoming lists.
 *
 * Does not reread any file except the patched note's text. Unresolved targets
 * are dropped, same as {@link buildLinkIndex}.
 */
export function patchLinkIndex(
  index: ExplicitLinkIndex,
  relativePath: string,
  text: string,
  entries: readonly WorkspaceIndexEntryV1[],
): void {
  const lookup = buildWikiLookup(entries.map(asCandidate));
  const titles = index.titles as Map<string, string>;
  const outgoing = index.outgoing as Map<string, string[]>;
  const incoming = index.incoming as Map<string, string[]>;
  const entry = entries.find((item) => item.relativePath === relativePath);
  titles.set(relativePath, titleFromNote(text, entry?.name ?? relativePath));

  const previous = outgoing.get(relativePath) ?? [];
  const next = resolveOutgoing(text, relativePath, lookup);
  outgoing.set(relativePath, next);

  const kept = new Set(next);
  for (const target of previous) {
    if (kept.has(target)) continue;
    const inbound = incoming.get(target);
    if (inbound === undefined) continue;
    incoming.set(target, inbound.filter((from) => from !== relativePath));
  }

  const had = new Set(previous);
  for (const target of next) {
    if (had.has(target)) continue;
    const inbound = incoming.get(target);
    if (inbound === undefined) {
      incoming.set(target, [relativePath]);
      continue;
    }
    if (!inbound.includes(relativePath)) {
      incoming.set(target, [...inbound, relativePath]);
    }
  }
}

const toLink = (index: ExplicitLinkIndex, relativePath: string, fallbackPath: string): IndexedLink => ({
  path: fallbackPath,
  relativePath,
  title: index.titles.get(relativePath) ?? relativePath.replace(/\.(?:md|markdown)$/i, ''),
});

/** The two explicit lists for one note, empty when the index has not met it. */
export function neighbourhood(
  index: ExplicitLinkIndex,
  relativePath: string,
  pathOf: (relativePath: string) => string,
): NoteNeighbourhood | null {
  const outgoing = index.outgoing.get(relativePath);
  const incoming = index.incoming.get(relativePath);
  if (outgoing === undefined || incoming === undefined) return null;
  const link = (target: string): IndexedLink => toLink(index, target, pathOf(target));
  const byTitle = (left: IndexedLink, right: IndexedLink) =>
    left.title.localeCompare(right.title) || left.relativePath.localeCompare(right.relativePath);
  return {
    outgoing: outgoing.map(link).sort(byTitle),
    backlinks: incoming.map(link).sort(byTitle),
  };
}
