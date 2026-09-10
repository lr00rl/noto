/**
 * Every note's frontmatter tags, scanned once per folder.
 *
 * The author's file-tags idea was multi-file linking by tag, "类似 obsidian
 * 的关系网". The source of truth stays the notes themselves — ordinary
 * `tags:` in YAML — so nothing under `.typora` has to exist for this to work,
 * and a note opened without Noto still carries its tags. The scan only reads
 * enough of each markdown file to reach the closing `---` of the frontmatter,
 * which is the cheap part of a vault that is otherwise tens of megabytes.
 */

import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { parseTagsFromMarkdown } from '../../shared/tags/parse';
import type { WorkspaceIndexEntryV1 } from '../../shared/workspace/v1/contracts';

/** Reads in flight at once, matching content search's politeness. */
const CONCURRENCY = 24;

/** A scan that has run this long stops rather than hanging the box. */
export const TAG_INDEX_BUDGET_MS = 4_000;

/** How many bytes are enough to hold a frontmatter. Past this, skip. */
const FRONTMATTER_READ = 8_192;

export interface TagNoteHit {
  readonly path: string;
  readonly relativePath: string;
  readonly name: string;
  /** File stem, which is what the browser shows for an untitled note. */
  readonly title: string;
}

export interface TagIndexEntry {
  /** The spelling first seen for this tag. */
  readonly tag: string;
  readonly notes: readonly TagNoteHit[];
}

export interface TagIndex {
  readonly tags: readonly TagIndexEntry[];
  readonly truncated: boolean;
}

export interface BuildTagIndexOptions {
  readonly budgetMs?: number;
  readonly now?: () => number;
  readonly read?: (path: string) => Promise<string>;
}

const titleOf = (name: string): string => name.replace(/\.md$/i, '');

/**
 * Read only the head of a file: enough for YAML frontmatter, not the body.
 *
 * A full `readFile` on seven thousand notes is what content search does and
 * costs a second; tags only live at the top, so the head is enough.
 */
async function readHead(
  filePath: string,
  read: (path: string) => Promise<string>,
): Promise<string> {
  // `readFile` then slice is fine here: Node reads the whole file into memory
  // anyway for utf8, and FRONTMATTER_READ is only a logical bound. A future
  // open/read/close of N bytes would matter on multi-megabyte notes.
  const text = await read(filePath);
  return text.length > FRONTMATTER_READ ? text.slice(0, FRONTMATTER_READ) : text;
}

/**
 * Build the tag → notes map for every markdown entry in the file index.
 *
 * Non-markdown openables (there are none today, but the index is shared) are
 * skipped. A file with no frontmatter, or no `tags:` key, contributes nothing.
 */
export async function buildTagIndex(
  entries: readonly WorkspaceIndexEntryV1[],
  options: BuildTagIndexOptions = {},
): Promise<TagIndex> {
  const budgetMs = options.budgetMs ?? TAG_INDEX_BUDGET_MS;
  const now = options.now ?? Date.now;
  const read = options.read ?? ((file) => readFile(file, 'utf8'));
  const started = now();

  const markdown = entries.filter((entry) => entry.name.toLowerCase().endsWith('.md'));
  const byKey = new Map<string, { tag: string; notes: TagNoteHit[] }>();
  let truncated = false;
  let cursor = 0;

  const worker = async (): Promise<void> => {
    while (cursor < markdown.length) {
      if (now() - started > budgetMs) {
        truncated = true;
        return;
      }
      const at = cursor;
      cursor += 1;
      const entry = markdown[at]!;
      let head: string;
      try {
        head = await readHead(entry.path, read);
      } catch {
        continue;
      }
      const tags = parseTagsFromMarkdown(head);
      if (tags.length === 0) continue;
      const hit: TagNoteHit = {
        path: entry.path,
        relativePath: entry.relativePath,
        name: entry.name,
        title: titleOf(entry.name),
      };
      for (const tag of tags) {
        const key = tag.toLowerCase();
        const bucket = byKey.get(key);
        if (bucket) bucket.notes.push(hit);
        else byKey.set(key, { tag, notes: [hit] });
      }
    }
  };

  const pool = Array.from({ length: Math.min(CONCURRENCY, Math.max(1, markdown.length)) }, () => worker());
  await Promise.all(pool);

  const tags = [...byKey.values()]
    .map((entry) => ({ tag: entry.tag, notes: entry.notes }))
    .sort((a, b) => a.tag.localeCompare(b.tag, undefined, { sensitivity: 'base' }));

  return { tags, truncated };
}

/** Notes carrying one tag, or `[]` when the index has never met it. */
export function notesForTag(index: TagIndex, tag: string): readonly TagNoteHit[] {
  const key = tag.toLowerCase();
  return index.tags.find((entry) => entry.tag.toLowerCase() === key)?.notes ?? [];
}

/** Every tag on one note, from an already-built index. */
export function tagsForPath(index: TagIndex, filePath: string): string[] {
  const found: string[] = [];
  for (const entry of index.tags) {
    if (entry.notes.some((note) => note.path === filePath)) found.push(entry.tag);
  }
  return found;
}

/** Folder of a relative path, for the browser's secondary line. */
export function folderOf(relativePath: string): string {
  const cut = relativePath.lastIndexOf('/');
  return cut > 0 ? relativePath.slice(0, cut) : '';
}
