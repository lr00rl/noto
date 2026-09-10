/**
 * Every command the palette can run.
 *
 * The application menu already names these, but a menu is a poor place to
 * look: you have to know which one holds the thing. The palette is the other
 * door, the one you type into. Titles are sentences a person would search
 * for, not the ids the menu uses internally.
 *
 * Ranking is the same subsequence scorer quick open uses, so a vault of notes
 * and a list of commands feel like one product.
 */

import { fuzzyScore, NO_MATCH } from '../shared/search/v1/fuzzy';
import type { WorkspaceMenuCommandV1 } from '../shared/workspace/v1/contracts';

export type CommandGroup = 'Go' | 'File' | 'Edit' | 'Insert' | 'Format' | 'Paragraph' | 'View' | 'Plugins';

export interface CatalogCommand {
  readonly id: WorkspaceMenuCommandV1;
  readonly title: string;
  readonly group: CommandGroup;
  readonly keys?: string;
  readonly keywords: string;
  /** Shown when the box is empty. Everything else waits to be asked for. */
  readonly pinned?: boolean;
}

export interface PaletteRow {
  readonly key: string;
  readonly title: string;
  readonly group: CommandGroup;
  readonly keys?: string;
  readonly source?: string;
  readonly kind: 'workspace' | 'plugin';
  readonly command?: WorkspaceMenuCommandV1;
  readonly pluginId?: string;
  readonly commandId?: string;
}

/**
 * The chord as a label, not as an accelerator.
 *
 * Electron writes `CmdOrCtrl+Shift+P`. What a person reads is `⇧⌘P` on a Mac
 * (Shift before Command, Apple's modifier order) and `Ctrl+Shift+P`
 * everywhere else. The palette is not a menu and cannot borrow the menu's
 * renderer.
 */
const MAC_MOD: Readonly<Record<string, { glyph: string; rank: number }>> = {
  Control: { glyph: '⌃', rank: 0 },
  Ctrl: { glyph: '⌃', rank: 0 },
  Alt: { glyph: '⌥', rank: 1 },
  Option: { glyph: '⌥', rank: 1 },
  Shift: { glyph: '⇧', rank: 2 },
  CmdOrCtrl: { glyph: '⌘', rank: 3 },
  Command: { glyph: '⌘', rank: 3 },
  Cmd: { glyph: '⌘', rank: 3 },
};

const KEY_GLYPH: Readonly<Record<string, string>> = {
  Up: '↑', Down: '↓', Left: '←', Right: '→',
};

export function shortcutLabel(accelerator: string, mac: boolean): string {
  const tokens = accelerator.split('+');
  const keyToken = tokens.pop() ?? '';
  const key = KEY_GLYPH[keyToken] ?? keyToken;
  if (!mac) {
    const mods = tokens.map((token) => {
      if (token === 'CmdOrCtrl' || token === 'Command' || token === 'Cmd' || token === 'Control' || token === 'Ctrl') {
        return 'Ctrl';
      }
      if (token === 'Alt' || token === 'Option') return 'Alt';
      return token;
    });
    return [...mods, key].join('+');
  }
  const mods = tokens
    .map((token) => MAC_MOD[token])
    .filter((item): item is { glyph: string; rank: number } => item !== undefined)
    .sort((left, right) => left.rank - right.rank)
    .map((item) => item.glyph);
  return `${mods.join('')}${key}`;
}

/**
 * The commands a person reaches for without thinking, in the order they
 * belong on an empty palette: finding a note, making one, then shaping the
 * page. The rest of the catalog is there the moment they type.
 */
export const CATALOG: readonly CatalogCommand[] = [
  { id: 'quick-open', title: 'Quick Open', group: 'Go', keys: 'CmdOrCtrl+P', keywords: 'file search jump palette notes', pinned: true },
  { id: 'search-content', title: 'Find in notes', group: 'Go', keys: 'CmdOrCtrl+Shift+F', keywords: 'vault search content grep', pinned: true },
  { id: 'new-file', title: 'New note', group: 'File', keys: 'CmdOrCtrl+N', keywords: 'create file untitled', pinned: true },
  { id: 'find', title: 'Find in this note', group: 'Edit', keys: 'CmdOrCtrl+F', keywords: 'search replace', pinned: true },
  { id: 'insert-link', title: 'Insert link', group: 'Insert', keys: 'CmdOrCtrl+K', keywords: 'hyperlink url', pinned: true },
  { id: 'block-heading-1', title: 'Heading 1', group: 'Paragraph', keys: 'CmdOrCtrl+1', keywords: 'h1 title', pinned: true },
  { id: 'block-bullet-list', title: 'Bullet list', group: 'Paragraph', keys: 'CmdOrCtrl+Alt+U', keywords: 'ul unordered', pinned: true },
  { id: 'block-code', title: 'Code block', group: 'Paragraph', keys: 'CmdOrCtrl+Alt+C', keywords: 'fence pre', pinned: true },
  { id: 'table-insert', title: 'Insert table', group: 'Insert', keys: 'CmdOrCtrl+Alt+T', keywords: 'grid rows columns', pinned: true },
  { id: 'toggle-focus-mode', title: 'Toggle focus mode', group: 'View', keywords: 'dim writing zen', pinned: true },
  { id: 'toggle-sidebar', title: 'Toggle the file rail', group: 'View', keys: 'CmdOrCtrl+Shift+L', keywords: 'sidebar tree files', pinned: true },
  { id: 'settings', title: 'Preferences', group: 'View', keys: 'CmdOrCtrl+,', keywords: 'settings theme appearance', pinned: true },

  { id: 'save', title: 'Save', group: 'File', keys: 'CmdOrCtrl+S', keywords: 'write disk' },
  { id: 'save-as', title: 'Save a copy', group: 'File', keys: 'CmdOrCtrl+Shift+S', keywords: 'duplicate' },
  { id: 'reload-from-disk', title: 'Reload from disk', group: 'File', keys: 'CmdOrCtrl+R', keywords: 'revert refresh' },
  { id: 'reveal-document', title: 'Reveal the note in the file manager', group: 'File', keys: 'CmdOrCtrl+Shift+R', keywords: 'finder explorer folder' },
  { id: 'insert-image', title: 'Insert image', group: 'Insert', keywords: 'picture photo paste' },

  { id: 'undo', title: 'Undo', group: 'Edit', keys: 'CmdOrCtrl+Z', keywords: '' },
  { id: 'redo', title: 'Redo', group: 'Edit', keys: 'CmdOrCtrl+Shift+Z', keywords: '' },
  { id: 'find-replace', title: 'Find and replace', group: 'Edit', keys: 'CmdOrCtrl+Alt+F', keywords: 'substitute' },
  { id: 'find-next', title: 'Find next', group: 'Edit', keys: 'CmdOrCtrl+G', keywords: '' },
  { id: 'find-previous', title: 'Find previous', group: 'Edit', keys: 'CmdOrCtrl+Shift+G', keywords: '' },
  { id: 'select-word', title: 'Select word', group: 'Edit', keys: 'CmdOrCtrl+D', keywords: '' },
  { id: 'select-line', title: 'Select line', group: 'Edit', keys: 'CmdOrCtrl+L', keywords: '' },
  { id: 'select-scope', title: 'Select the styled run', group: 'Edit', keys: 'CmdOrCtrl+E', keywords: 'mark bold italic' },
  { id: 'jump-to-selection', title: 'Jump to selection', group: 'Edit', keys: 'CmdOrCtrl+J', keywords: 'scroll' },
  { id: 'copy-as-markdown', title: 'Copy as markdown', group: 'Edit', keys: 'CmdOrCtrl+Shift+C', keywords: '' },
  { id: 'copy-as-html', title: 'Copy as HTML', group: 'Edit', keywords: '' },
  { id: 'copy-as-plain', title: 'Copy as plain text', group: 'Edit', keywords: '' },

  { id: 'block-heading-2', title: 'Heading 2', group: 'Paragraph', keys: 'CmdOrCtrl+2', keywords: 'h2' },
  { id: 'block-heading-3', title: 'Heading 3', group: 'Paragraph', keys: 'CmdOrCtrl+3', keywords: 'h3' },
  { id: 'block-heading-4', title: 'Heading 4', group: 'Paragraph', keys: 'CmdOrCtrl+4', keywords: 'h4' },
  { id: 'block-heading-5', title: 'Heading 5', group: 'Paragraph', keys: 'CmdOrCtrl+5', keywords: 'h5' },
  { id: 'block-heading-6', title: 'Heading 6', group: 'Paragraph', keys: 'CmdOrCtrl+6', keywords: 'h6' },
  { id: 'block-paragraph', title: 'Paragraph', group: 'Paragraph', keys: 'CmdOrCtrl+0', keywords: 'body text' },
  { id: 'block-heading-up', title: 'Increase heading level', group: 'Paragraph', keys: 'CmdOrCtrl+=', keywords: 'promote' },
  { id: 'block-heading-down', title: 'Decrease heading level', group: 'Paragraph', keys: 'CmdOrCtrl+-', keywords: 'demote' },
  { id: 'block-ordered-list', title: 'Numbered list', group: 'Paragraph', keys: 'CmdOrCtrl+Alt+O', keywords: 'ol ordered' },
  { id: 'block-task-list', title: 'Task list', group: 'Paragraph', keys: 'CmdOrCtrl+Alt+X', keywords: 'todo checkbox' },
  { id: 'block-quote', title: 'Quote', group: 'Paragraph', keys: 'CmdOrCtrl+Alt+Q', keywords: 'blockquote' },
  { id: 'block-math', title: 'Math block', group: 'Paragraph', keys: 'CmdOrCtrl+Alt+B', keywords: 'latex katex equation' },
  { id: 'block-rule', title: 'Horizontal rule', group: 'Paragraph', keys: 'CmdOrCtrl+Alt+-', keywords: 'hr divider' },
  { id: 'block-alert-note', title: 'Note callout', group: 'Paragraph', keywords: 'alert github admonition' },
  { id: 'block-alert-tip', title: 'Tip callout', group: 'Paragraph', keywords: 'alert github' },
  { id: 'block-alert-important', title: 'Important callout', group: 'Paragraph', keywords: 'alert github' },
  { id: 'block-alert-warning', title: 'Warning callout', group: 'Paragraph', keywords: 'alert github' },
  { id: 'block-alert-caution', title: 'Caution callout', group: 'Paragraph', keywords: 'alert github' },
  { id: 'move-up', title: 'Move block up', group: 'Paragraph', keys: 'Alt+Up', keywords: 'line row' },
  { id: 'move-down', title: 'Move block down', group: 'Paragraph', keys: 'Alt+Down', keywords: 'line row' },
  { id: 'indent-more', title: 'Increase indent', group: 'Paragraph', keywords: 'list nest' },
  { id: 'indent-less', title: 'Decrease indent', group: 'Paragraph', keywords: 'list outdent' },

  { id: 'mark-strong', title: 'Bold', group: 'Format', keys: 'CmdOrCtrl+B', keywords: 'strong' },
  { id: 'mark-emphasis', title: 'Italic', group: 'Format', keys: 'CmdOrCtrl+I', keywords: 'emphasis' },
  { id: 'mark-strike', title: 'Strikethrough', group: 'Format', keywords: 'del' },
  { id: 'mark-code', title: 'Inline code', group: 'Format', keywords: 'monospace backtick' },
  { id: 'mark-highlight', title: 'Highlight', group: 'Format', keys: 'CmdOrCtrl+Shift+H', keywords: 'mark' },
  { id: 'mark-underline', title: 'Underline', group: 'Format', keys: 'CmdOrCtrl+U', keywords: '' },
  { id: 'mark-math', title: 'Inline math', group: 'Format', keywords: 'latex katex' },
  { id: 'clear-format', title: 'Clear formatting', group: 'Format', keywords: 'remove marks' },

  { id: 'insert-footnote', title: 'Insert footnote', group: 'Insert', keywords: 'reference' },
  { id: 'insert-toc', title: 'Insert table of contents', group: 'Insert', keywords: 'outline toc' },
  { id: 'insert-frontmatter', title: 'Insert frontmatter', group: 'Insert', keywords: 'yaml metadata' },
  { id: 'insert-link-reference', title: 'Insert link reference', group: 'Insert', keywords: 'footnote style' },
  { id: 'insert-comment', title: 'Insert comment', group: 'Insert', keywords: 'html hidden' },

  { id: 'toggle-outline', title: 'Show the outline', group: 'View', keys: 'CmdOrCtrl+Shift+O', keywords: 'headings toc' },
  { id: 'toggle-typewriter', title: 'Toggle typewriter mode', group: 'View', keywords: 'center line' },
  { id: 'toggle-source', title: 'Toggle source for this block', group: 'View', keys: 'CmdOrCtrl+Alt+/', keywords: 'markdown raw' },
  { id: 'source-code-mode', title: 'Source Code Mode', group: 'View', keys: 'CmdOrCtrl+/', keywords: 'whole note text' },
  { id: 'widen', title: 'Widen the page', group: 'View', keywords: 'measure width' },
  { id: 'narrow', title: 'Narrow the page', group: 'View', keywords: 'measure width' },
  { id: 'toggle-read-only', title: 'Toggle read-only', group: 'View', keywords: 'lock' },
  { id: 'toggle-always-on-top', title: 'Float the window on top', group: 'View', keywords: 'pin' },
  { id: 'shortcuts', title: 'What Noto can do', group: 'View', keywords: 'help keys cheatsheet' },
  { id: 'navigate-back', title: 'Back', group: 'Go', keys: 'CmdOrCtrl+Alt+Left', keywords: 'trail history' },
  { id: 'navigate-forward', title: 'Forward', group: 'Go', keys: 'CmdOrCtrl+Alt+Right', keywords: 'trail history' },
  { id: 'tree-collapse-all', title: 'Collapse the file tree', group: 'View', keywords: 'folders' },

  { id: 'export-html', title: 'Export HTML', group: 'File', keywords: 'print' },
  { id: 'export-pdf', title: 'Export PDF', group: 'File', keywords: 'print' },
];

export interface SlashItem {
  readonly id: WorkspaceMenuCommandV1;
  readonly title: string;
  readonly hint: string;
  readonly aliases: readonly string[];
}

/**
 * What typing `/` at the start of a block offers.
 *
 * Aliases are the words a person actually types after the slash. Prefix
 * match on those is the whole ranking: `/h1` has to land on Heading 1, not
 * on something that merely contains an h.
 */
export const SLASH_ITEMS: readonly SlashItem[] = [
  { id: 'block-heading-1', title: 'Heading 1', hint: '#', aliases: ['h1', 'heading', 'title'] },
  { id: 'block-heading-2', title: 'Heading 2', hint: '##', aliases: ['h2', 'heading2'] },
  { id: 'block-heading-3', title: 'Heading 3', hint: '###', aliases: ['h3', 'heading3'] },
  { id: 'block-bullet-list', title: 'Bullet list', hint: '-', aliases: ['bullet', 'ul', 'list'] },
  { id: 'block-ordered-list', title: 'Numbered list', hint: '1.', aliases: ['numbered', 'ol', 'ordered'] },
  { id: 'table-insert', title: 'Table', hint: '|', aliases: ['table', 'grid'] },
  { id: 'block-task-list', title: 'Task list', hint: '- [ ]', aliases: ['task', 'todo', 'checkbox'] },
  { id: 'block-quote', title: 'Quote', hint: '>', aliases: ['quote', 'blockquote'] },
  { id: 'block-code', title: 'Code block', hint: '```', aliases: ['code', 'fence', 'pre'] },
  { id: 'block-math', title: 'Math block', hint: '$$', aliases: ['math', 'latex', 'equation'] },
  { id: 'block-rule', title: 'Horizontal rule', hint: '---', aliases: ['rule', 'hr', 'divider'] },
  { id: 'block-alert-note', title: 'Note', hint: '[!NOTE]', aliases: ['note', 'callout', 'alert'] },
  { id: 'block-alert-tip', title: 'Tip', hint: '[!TIP]', aliases: ['tip'] },
  { id: 'block-alert-warning', title: 'Warning', hint: '[!WARNING]', aliases: ['warning', 'warn'] },
  { id: 'block-alert-important', title: 'Important', hint: '[!IMPORTANT]', aliases: ['important'] },
  { id: 'block-alert-caution', title: 'Caution', hint: '[!CAUTION]', aliases: ['caution'] },
  { id: 'insert-image', title: 'Image', hint: '![]', aliases: ['image', 'img', 'picture', 'photo'] },
  { id: 'insert-link', title: 'Link', hint: '[]()', aliases: ['link', 'url', 'href'] },
  { id: 'insert-toc', title: 'Table of contents', hint: 'toc', aliases: ['toc', 'contents', 'outline'] },
  { id: 'insert-frontmatter', title: 'Frontmatter', hint: '---', aliases: ['frontmatter', 'yaml', 'meta'] },
  { id: 'insert-footnote', title: 'Footnote', hint: '[^]', aliases: ['footnote', 'fn'] },
];

export function rankCommands(query: string, extra: readonly PaletteRow[] = []): PaletteRow[] {
  const workspace: PaletteRow[] = CATALOG.map((entry) => ({
    key: entry.id,
    title: entry.title,
    group: entry.group,
    keys: entry.keys,
    kind: 'workspace',
    command: entry.id,
  }));
  const all = [...workspace, ...extra];
  const trimmed = query.trim();
  if (trimmed.length === 0) {
    const pinned = new Set(CATALOG.filter((entry) => entry.pinned).map((entry) => entry.id));
    return all.filter((row) => row.command !== undefined && pinned.has(row.command));
  }

  const scored = all.map((row) => {
    const haystack = `${row.title} ${row.group} ${row.source ?? ''} ${row.command ?? ''} ${row.commandId ?? ''}`;
    const catalog = CATALOG.find((entry) => entry.id === row.command);
    const extraTerms = catalog?.keywords ?? '';
    const score = Math.max(fuzzyScore(row.title, trimmed), fuzzyScore(`${haystack} ${extraTerms}`, trimmed));
    return { row, score };
  }).filter((entry) => entry.score !== NO_MATCH);
  scored.sort((left, right) => right.score - left.score);
  return scored.map((entry) => entry.row);
}

export function rankSlash(query: string): SlashItem[] {
  const trimmed = query.trim().toLowerCase();
  if (trimmed.length === 0) return [...SLASH_ITEMS];
  const scored = SLASH_ITEMS.map((item, index) => {
    const alias = item.aliases.find((name) => name.startsWith(trimmed) || name === trimmed);
    const titleHit = item.title.toLowerCase().startsWith(trimmed)
      || item.title.toLowerCase().includes(trimmed);
    if (!alias && !titleHit) return null;
    const score = alias === trimmed ? 300
      : alias?.startsWith(trimmed) ? 200
        : titleHit ? 100 : 0;
    return { item, score, index };
  }).filter((entry): entry is { item: SlashItem; score: number; index: number } => entry !== null);
  scored.sort((left, right) => right.score - left.score || left.index - right.index);
  return scored.map((entry) => entry.item);
}
