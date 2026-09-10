import { describe, expect, it } from 'vitest';
import {
  encodeSourceBuffer,
  sourceHasFinalNewline,
  sourceModeText,
  sourceSettleKind,
  sourceStructureDiffers,
} from '../../src/renderer/source-mode-text';
import { EditorState } from 'prosemirror-state';
import { splitBlocks } from '../../src/shared/markdown/v3/blocks';
import { NOTO_MARKDOWN_VERSION } from '../../src/shared/markdown/v3/contracts';
import { parseDocument, toWire } from '../../src/shared/markdown/v3/document';
import { toLf } from '../../src/shared/markdown/v3/line-endings';
import { serializeDocument } from '../../src/shared/markdown/v3/serialize';
import { captureMarkdown, captureTransaction, type PristineBlock } from '../../src/renderer/editor/noto/capture';
import { docFromSpans } from '../../src/shared/markdown/v3/pm/from-mdast';
import { createOriginPlugin, getBlockOrigins } from '../../src/renderer/editor/noto/origin-plugin';

describe('Source Code Mode text', () => {
  it('shows the accepted file text, LF-normalised, while the note is clean', () => {
    expect(sourceModeText({
      dirty: false,
      fileText: '# A\r\n\r\n\r\nBody with a wide gap.\r\n',
      reconstructed: '# A\n\nBody with a wide gap.',
      hasFinalNewline: true,
    })).toBe('# A\n\n\nBody with a wide gap.\n');
  });

  it('reconstructs from the editor once the note is dirty', () => {
    expect(sourceModeText({
      dirty: true,
      fileText: '# A\n\n\nBody.\n',
      reconstructed: '# A\n\nBody changed.',
      hasFinalNewline: true,
    })).toBe('# A\n\nBody changed.\n');

    expect(sourceModeText({
      dirty: true,
      fileText: '# A\n\nBody.\n',
      reconstructed: '# A\n\nBody.',
      hasFinalNewline: false,
    })).toBe('# A\n\nBody.');
  });

  it('prefers a pending full-source buffer over reconstruction', () => {
    expect(sourceModeText({
      dirty: true,
      fileText: '# A\n\nBody.\n',
      reconstructed: '# A\n\nBody.',
      hasFinalNewline: true,
      pendingSource: '# A\n\n\nBody.\n',
    })).toBe('# A\n\n\nBody.\n');
  });

  it('reads the trailing newline from the buffer for the envelope', () => {
    expect(sourceHasFinalNewline('# A\n')).toBe(true);
    expect(sourceHasFinalNewline('# A')).toBe(false);
    expect(sourceHasFinalNewline('')).toBe(false);
  });
});

describe('Source Code Mode settle gate', () => {
  const file = '# Title\n\n\nBody text.\n';

  it('takes the source escape for a gap-only change', () => {
    expect(sourceStructureDiffers('# Title\n\n\n\nBody text.\n', file)).toBe(true);
    expect(sourceSettleKind({
      blockReplaced: false,
      buffer: '# Title\n\n\n\nBody text.\n',
      currentDocumentText: file,
    })).toBe('source');
  });

  it('is a no-op when the buffer matches the file (aside from final newline)', () => {
    expect(sourceSettleKind({
      blockReplaced: false,
      buffer: '# Title\n\n\nBody text.\n',
      currentDocumentText: file,
    })).toBe('noop');
    expect(sourceSettleKind({
      blockReplaced: false,
      buffer: '# Title\n\n\nBody text.',
      currentDocumentText: file,
    })).toBe('noop');
  });

  it('prefers the blocks path for a single-block edit with the same gaps', () => {
    expect(sourceStructureDiffers('# Title\n\n\nBody edited.\n', file)).toBe(false);
    expect(sourceSettleKind({
      blockReplaced: true,
      buffer: '# Title\n\n\nBody edited.\n',
      currentDocumentText: file,
    })).toBe('blocks');
  });

  it('takes source when a block edit also changes gaps', () => {
    expect(sourceSettleKind({
      blockReplaced: true,
      buffer: '# Title\n\nBody edited.\n',
      currentDocumentText: file,
    })).toBe('source');
  });
});

describe('Source Code Mode full-source escape', () => {
  const encoder = new TextEncoder();

  function parsed(source: string) {
    const result = parseDocument(encoder.encode(source));
    if (result.status !== 'parsed') throw new Error(result.message);
    return result.document;
  }

  it('gap-only change sticks through mode: source and marks a real write', () => {
    const source = '# Title\n\n\nBody text.\n';
    const document = parsed(source);
    const next = '# Title\n\n\n\nBody text.\n';
    expect(sourceSettleKind({
      blockReplaced: false,
      buffer: next,
      currentDocumentText: source,
    })).toBe('source');

    const sourceBytes = encodeSourceBuffer({
      markdown: next,
      lineEnding: document.envelope.lineEnding,
      bom: document.envelope.bom,
    });
    const result = serializeDocument(document, {
      version: NOTO_MARKDOWN_VERSION,
      mode: 'source',
      documentId: document.documentId,
      revisionId: document.revisionId,
      expectedSourceSha256: document.envelope.sourceSha256,
      sourceBytes,
    });
    expect(result.status).toBe('serialized');
    if (result.status !== 'serialized') return;
    expect(Buffer.from(result.outputBytes).toString('utf8')).toBe(next);
    expect(result.document.gaps.map((gap) => gap.text)).toEqual(['\n\n\n\n']);
  });

  it('cancel / no-op when identical does not need a source transaction', () => {
    const source = '# Title\n\nBody.\n';
    expect(sourceSettleKind({
      blockReplaced: false,
      buffer: source,
      currentDocumentText: source,
    })).toBe('noop');
  });

  it('normal single-block edit keeps neighbour provenance on the blocks path', () => {
    const source = '# Title\n\n\nBody text.\n\n- item\n';
    const document = parsed(source);
    const wire = toWire(document);
    const spans = splitBlocks(wire.text).spans;
    const doc = docFromSpans(spans);
    const pristine = new Map<string, PristineBlock>();
    doc.forEach((node, _offset, index) => {
      const origin = wire.origins[index];
      const span = spans[index];
      if (origin && span) pristine.set(origin.blockId, { node, markdown: toLf(span.markdown) });
    });
    const state = EditorState.create({ doc, plugins: [createOriginPlugin(wire.origins)] });

    // Simulate replaceMarkdown succeeding for the middle block only: capture
    // after rewriting that block's markdown in the pristine map sense by
    // building units the way a real edit would — edited middle, reused sides.
    const current = captureMarkdown({
      doc: state.doc,
      origins: getBlockOrigins(state),
      document: wire,
      pristine,
    });
    expect(current).toEqual(['# Title', 'Body text.', '- item']);

    const buffer = '# Title\n\n\nBody edited.\n\n- item\n';
    expect(sourceSettleKind({
      blockReplaced: true,
      buffer,
      currentDocumentText: source,
    })).toBe('blocks');

    const { transaction, stats } = captureTransaction({
      doc: state.doc,
      origins: getBlockOrigins(state),
      document: wire,
      pristine,
    });
    // Untouched open state: all reused. A real middle-block edit would
    // serialize one unit; the settle kind above is what Source Mode uses to
    // stay on this path instead of mode: 'source'.
    expect(transaction.mode).toBe('blocks');
    expect(stats.reused).toBe(3);

    // And a blocks save with only the middle unit dirty still preserves the
    // wide gap beside a pristine neighbour.
    if (transaction.mode !== 'blocks') throw new Error('expected blocks');
    const edited = {
      ...transaction,
      units: [
        { origin: transaction.units[0].origin, markdown: null },
        { origin: transaction.units[1].origin, markdown: 'Body edited.' },
        { origin: transaction.units[2].origin, markdown: null },
      ],
    };
    const result = serializeDocument(document, edited);
    expect(result.status).toBe('serialized');
    if (result.status !== 'serialized') return;
    expect(Buffer.from(result.outputBytes).toString('utf8')).toBe('# Title\n\n\nBody edited.\n\n- item\n');
    expect(result.preserved.some((range) => range.role === 'block')).toBe(true);
    expect(result.preserved.some((range) => range.role === 'gap')).toBe(true);
  });

  it('encodeSourceBuffer restores CRLF and BOM the way the envelope asks', () => {
    const withBom = encodeSourceBuffer({
      markdown: '# A\n\nB\n',
      lineEnding: 'crlf',
      bom: 'utf8',
    });
    expect(withBom.slice(0, 3)).toEqual(Uint8Array.from([0xef, 0xbb, 0xbf]));
    expect(Buffer.from(withBom.slice(3)).toString('utf8')).toBe('# A\r\n\r\nB\r\n');
  });
});
