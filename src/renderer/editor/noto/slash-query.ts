/**
 * Typing `/` at the start of a block asks what to insert.
 *
 * Markdown already turns `# ` and `- ` into the thing they name. Slash is
 * the same idea for the constructs that have no short marker, or whose
 * marker is easy to forget: a table, a callout, a math block. It only
 * fires when the `/` is the first thing in the block, so a path in a
 * sentence stays a path. A second `/` in the token means a path too:
 * `/usr/bin` is not a command, but `/table` still is.
 */

const BLOCKED = new Set([
  'code_block',
  'source_block',
  'math_block',
  'html_block',
  'table_cell',
  'table_header',
]);

export interface SlashToken {
  /** The letters after the slash, which is what the menu filters on. */
  readonly query: string;
  /** Offset of the slash inside the textblock, always 0 today. */
  readonly start: number;
  /** Offset of the caret inside the textblock. */
  readonly end: number;
}

/**
 * The slash token the caret is sitting in, or null when this is not a slash
 * command at all.
 *
 * `text` is the textblock's text and `offset` is the caret inside it. The
 * token has to run from the start of the block to the caret with no space,
 * or a `/` in the middle of a sentence would open the menu.
 */
export function slashToken(parentType: string, text: string, offset: number): SlashToken | null {
  if (BLOCKED.has(parentType)) return null;
  if (offset < 1) return null;
  const before = text.slice(0, offset);
  const match = /^\/(\S*)$/.exec(before);
  if (!match) return null;
  const query = match[1] ?? '';
  if (query.includes('/')) return null;
  return { query, start: 0, end: offset };
}
