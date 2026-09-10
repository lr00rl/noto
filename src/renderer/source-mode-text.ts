/**
 * What Source Code Mode shows, and how an edit lands back in the editor.
 *
 * A clean note can show the accepted file text (LF-normalised for the
 * textarea), so blank lines between blocks are what the file holds rather
 * than a re-join of every block with `\n\n`. A dirty note has to reconstruct
 * from the editor, which is the same shape a transform plugin reads — unless
 * a pending full-source escape is holding the buffer that blocks cannot.
 *
 * Applying the text prefers block-wise `replaceMarkdown`, so every block
 * whose markdown did not change keeps its provenance and still saves byte
 * for byte. When the buffer's gaps (or leading / extra trailing whitespace)
 * differ from the accepted file in a way the block model cannot express,
 * settle falls through to an explicit `mode: 'source'` escape for the next
 * save. Ordinary block edits never take that path. See docs/design/typora-gap.md.
 */

import { splitBlocks } from '../shared/markdown/v3/blocks';
import type { NotoLineEnding } from '../shared/markdown/v3/contracts';
import { fromLf, toLf } from '../shared/markdown/v3/line-endings';

const UTF8_BOM = Uint8Array.from([0xef, 0xbb, 0xbf]);
const encoder = new TextEncoder();

export function sourceModeText(options: {
  readonly dirty: boolean;
  readonly fileText: string;
  readonly reconstructed: string;
  readonly hasFinalNewline: boolean;
  /** Buffer held for a full-source save; wins over reconstruction. */
  readonly pendingSource?: string | null;
}): string {
  if (options.pendingSource != null) return options.pendingSource;
  if (!options.dirty) return toLf(options.fileText);
  return options.reconstructed + (options.hasFinalNewline ? '\n' : '');
}

/** Whether the source buffer ends the file with a newline. */
export function sourceHasFinalNewline(markdown: string): boolean {
  return markdown.endsWith('\n');
}

/**
 * Trailing whitespace with at most one final newline removed.
 *
 * The envelope already carries "file ends with a newline"; stripping one lets
 * gap / blank-line-at-end differences stay visible without treating a
 * final-newline toggle as a full-source escape.
 */
export function sourceTrailingBody(trailing: string): string {
  return trailing.endsWith('\n') ? trailing.slice(0, -1) : trailing;
}

/**
 * True when the buffer's inter-block gaps, leading text, or trailing body
 * differ from the accepted document in a way blocks+envelope cannot save.
 */
export function sourceStructureDiffers(buffer: string, currentDocumentText: string): boolean {
  const next = splitBlocks(toLf(buffer));
  const current = splitBlocks(toLf(currentDocumentText));
  if (next.leading !== current.leading) return true;
  if (next.gaps.length !== current.gaps.length) return true;
  if (next.gaps.some((gap, index) => gap !== current.gaps[index])) return true;
  return sourceTrailingBody(next.trailing) !== sourceTrailingBody(current.trailing);
}

export type SourceSettleKind = 'noop' | 'blocks' | 'source';

/**
 * Which settle path Source Mode should take after trying block-wise replace.
 *
 * - `source` — gaps / leading / extra trailing differ; save via mode: 'source'
 * - `blocks` — only block markdown changed; keep provenance on neighbours
 * - `noop` — nothing the block model missed (final newline is envelope-only)
 */
export function sourceSettleKind(options: {
  readonly blockReplaced: boolean;
  readonly buffer: string;
  readonly currentDocumentText: string;
}): SourceSettleKind {
  if (sourceStructureDiffers(options.buffer, options.currentDocumentText)) return 'source';
  if (options.blockReplaced) return 'blocks';
  return 'noop';
}

/**
 * Bytes for a `mode: 'source'` transaction from an LF source buffer.
 *
 * Bakes the target line ending and BOM the same way a blocks save would via
 * the envelope, so a full-source escape still restores endings on write.
 */
export function encodeSourceBuffer(options: {
  readonly markdown: string;
  readonly lineEnding: NotoLineEnding;
  readonly bom: 'utf8' | 'none';
}): Uint8Array {
  const body = encoder.encode(fromLf(toLf(options.markdown), options.lineEnding));
  if (options.bom === 'none') return body;
  const output = new Uint8Array(UTF8_BOM.length + body.length);
  output.set(UTF8_BOM, 0);
  output.set(body, UTF8_BOM.length);
  return output;
}
