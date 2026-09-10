/**
 * What exporting a note can produce, and how each one is made.
 *
 * Shared so the File menu, the command palette and the IPC contract all name
 * the same things. The writers that are null are rendered by Noto from the
 * screen; the rest are conversions Pandoc does from the file on disk.
 */

import { EXPORT_KINDS, type WorkspaceExportKindV1 } from '../workspace/v1/contracts';

export interface ExportTargetShape {
  /** What the menu and the palette call it. */
  readonly label: string;
  /** The extension the file is given, without its dot. */
  readonly extension: string;
  /**
   * The Pandoc writer, or null when Noto renders it itself.
   *
   * PDF and HTML are rendered rather than converted because what they are for
   * is the document as it looks, which Pandoc has never seen.
   */
  readonly writer: string | null;
}

export const EXPORT_TARGET_SHAPES: Readonly<Record<WorkspaceExportKindV1, ExportTargetShape>> = {
  pdf: { label: 'PDF', extension: 'pdf', writer: null },
  html: { label: 'HTML', extension: 'html', writer: null },
  'html-plain': { label: 'HTML without styles', extension: 'html', writer: null },
  docx: { label: 'Word (.docx)', extension: 'docx', writer: 'docx' },
  odt: { label: 'OpenDocument', extension: 'odt', writer: 'odt' },
  rtf: { label: 'RTF', extension: 'rtf', writer: 'rtf' },
  epub: { label: 'EPUB', extension: 'epub', writer: 'epub3' },
  latex: { label: 'LaTeX', extension: 'tex', writer: 'latex' },
  mediawiki: { label: 'MediaWiki', extension: 'wiki', writer: 'mediawiki' },
  rst: { label: 'reStructuredText', extension: 'rst', writer: 'rst' },
  textile: { label: 'Textile', extension: 'textile', writer: 'textile' },
  opml: { label: 'OPML', extension: 'opml', writer: 'opml' },
};

/** What the File menu and the palette offer, in the order Typora lists its own. */
export const EXPORT_TARGETS = EXPORT_KINDS;

export const exportShape = (target: WorkspaceExportKindV1): ExportTargetShape =>
  EXPORT_TARGET_SHAPES[target];

/** Whether Pandoc does this one, as opposed to Noto rendering it. */
export const needsPandoc = (target: WorkspaceExportKindV1): boolean =>
  EXPORT_TARGET_SHAPES[target].writer !== null;

/**
 * Commands the palette offers for export, in the same order as the File menu.
 *
 * Titles match the menu so a hand that has learned one finds the other. The
 * source label is what the palette shows beside a command that is not from a
 * plugin.
 */
export const EXPORT_PALETTE_COMMANDS: readonly {
  readonly target: WorkspaceExportKindV1;
  readonly title: string;
  readonly source: 'File';
}[] = EXPORT_TARGETS.map((target) => ({
  target,
  title: `Export ${exportShape(target).label}…`,
  source: 'File' as const,
}));
