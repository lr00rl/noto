/**
 * What Source Code Mode shows, and how an edit lands back in the editor.
 *
 * A clean note can show the accepted file text (LF-normalised for the
 * textarea), so blank lines between blocks are what the file holds rather
 * than a re-join of every block with `\n\n`. A dirty note has to reconstruct
 * from the editor, which is the same shape a transform plugin reads.
 *
 * Applying the text still goes through block-wise `replaceMarkdown`, so every
 * block whose markdown did not change keeps its provenance and still saves
 * byte for byte. Edits that only change a gap between two blocks are the
 * honest hole: the block model has nowhere to put them, and they do not stick
 * until a full-source save exists for this path. See docs/design/typora-gap.md.
 */

import { toLf } from '../shared/markdown/v3/line-endings';

export function sourceModeText(options: {
  readonly dirty: boolean;
  readonly fileText: string;
  readonly reconstructed: string;
  readonly hasFinalNewline: boolean;
}): string {
  if (!options.dirty) return toLf(options.fileText);
  return options.reconstructed + (options.hasFinalNewline ? '\n' : '');
}

/** Whether the source buffer ends the file with a newline. */
export function sourceHasFinalNewline(markdown: string): boolean {
  return markdown.endsWith('\n');
}
