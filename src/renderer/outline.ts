/**
 * The document outline.
 *
 * Derived from the accepted document rather than tracked as separate state, so
 * it can never drift from what the editor holds. Pure, so it is testable
 * without a DOM.
 *
 * Open used to call `outlineOf(text)`, which ran a full micromark split on the
 * UI thread while the editor Worker was already doing the same work. The wire
 * document already carries each block's kind and offsets from main's parse, so
 * the outline can be built by slicing heading markdown only.
 */

import { splitBlocks } from '../shared/markdown/v3/blocks';
import type { NotoDocumentWire } from '../shared/markdown/v3/contracts';

export interface OutlineEntry {
  /** Index of the top level block, which is what the editor navigates by. */
  readonly blockIndex: number;
  readonly depth: number;
  readonly text: string;
}

/**
 * Strip the leading `#` run and any trailing closing run from a heading.
 *
 * A closing run only counts when whitespace precedes it, which is what keeps
 * `### C# and F#` from losing its last character.
 */
function headingText(markdown: string): string {
  return markdown
    .replace(/^\s{0,3}#{1,6}\s*/, '')
    .replace(/\s+#+\s*$/, '')
    .trim();
}

function setextDepth(markdown: string): number | null {
  const lines = markdown.split('\n');
  if (lines.length < 2) return null;
  const underline = lines[lines.length - 1].trim();
  if (/^=+$/.test(underline)) return 1;
  if (/^-+$/.test(underline)) return 2;
  return null;
}

function entryFromHeading(blockIndex: number, markdown: string): OutlineEntry {
  const atx = /^\s{0,3}(#{1,6})\s/.exec(markdown);
  const depth = atx ? atx[1].length : setextDepth(markdown) ?? 1;
  const label = atx ? headingText(markdown) : markdown.split('\n')[0].trim();
  return { blockIndex, depth, text: label || 'Untitled heading' };
}

/**
 * Build the outline from the open-path wire document.
 *
 * Main already classified every block. Reusing those kinds avoids a second
 * full-document parse on the UI thread at open (and again after each accepted
 * save), which on a half-megabyte note was hundreds of milliseconds of work
 * that competed with the editor Worker.
 */
export function outlineFromDocument(document: NotoDocumentWire): OutlineEntry[] {
  const entries: OutlineEntry[] = [];
  const { text, origins, spans } = document;
  for (let blockIndex = 0; blockIndex < origins.length; blockIndex += 1) {
    if (origins[blockIndex].kind !== 'heading') continue;
    const span = spans[blockIndex];
    entries.push(entryFromHeading(blockIndex, text.slice(span.start, span.end)));
  }
  return entries;
}

/**
 * Build the outline by parsing `text`.
 *
 * Prefer `outlineFromDocument` when a wire document is already in hand. This
 * path remains for tests and any caller that only has source bytes.
 */
export function outlineOf(text: string): OutlineEntry[] {
  const entries: OutlineEntry[] = [];
  splitBlocks(text).spans.forEach((span, blockIndex) => {
    if (span.kind !== 'heading') return;
    entries.push(entryFromHeading(blockIndex, span.markdown));
  });
  return entries;
}

/** A heading with the headings nested under it. */
export interface OutlineNode extends OutlineEntry {
  readonly children: readonly OutlineNode[];
}

/**
 * Turn the flat heading list into the tree it describes.
 *
 * The rail draws connector lines, and a connector needs to know whether a
 * heading is the last of its siblings; a flat list with a depth number cannot
 * answer that without walking forward, so the walk happens once here instead of
 * in the view.
 *
 * Documents skip levels. An `h4` directly under an `h2` is nested under it
 * rather than given two empty ancestors, because the reader wants the shape of
 * their document, not a lesson about the levels they did not use. A heading
 * that starts deeper than everything after it still nests correctly, since the
 * stack pops on depth rather than on an assumed sequence.
 */
export function nestOutline(entries: readonly OutlineEntry[]): OutlineNode[] {
  const roots: OutlineNode[] = [];
  const stack: { depth: number; children: OutlineNode[] }[] = [{ depth: 0, children: roots }];
  for (const entry of entries) {
    while (stack.length > 1 && stack[stack.length - 1].depth >= entry.depth) stack.pop();
    const children: OutlineNode[] = [];
    stack[stack.length - 1].children.push({ ...entry, children });
    stack.push({ depth: entry.depth, children });
  }
  return roots;
}
