/**
 * Visible markers for tab characters inside a code fence.
 *
 * The author's `fence-enhance` plugin paints each `\t` the way CodeMirror's
 * visible-tabs demo does: a quiet arrow at the right of the character's own
 * advance, so a Makefile or a Go file that still uses tabs is readable as
 * structure rather than as empty gaps. Indent guides mark the steps of a
 * line's indentation; these mark the tabs themselves, wherever they sit.
 *
 * Drawn as a class on each tab's own span. The character stays in the file
 * and in the selection; only the paint changes. Pure, so the offsets are
 * tested without a document.
 */

export interface TabRange {
  /** Offset of the tab within the block's text. */
  readonly from: number;
  /** One past the tab. Always `from + 1`. */
  readonly to: number;
}

/** Every tab character in one code block's text. */
export function tabRanges(text: string): TabRange[] {
  const ranges: TabRange[] = [];
  for (let index = 0; index < text.length; index += 1) {
    if (text[index] === '\t') ranges.push({ from: index, to: index + 1 });
  }
  return ranges;
}

/** Class name carried by each tab span. The stylesheet paints the arrow. */
export const TAB_MARKER_CLASS = 'noto-code-tab';
