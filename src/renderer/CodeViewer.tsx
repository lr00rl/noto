/**
 * Read-only view of a non-Markdown text or code file.
 *
 * The author's typora-plugin-lite `code-viewer` paints source with line numbers
 * and syntax colour, and never writes the file. Noto does the same: the pane
 * is contentEditable=false, tokens are built with textContent, and the gutter
 * is a CSS ::before so a copy never carries line numbers.
 *
 * Draw.io: `.drawio` opens as highlighted XML with a short note; `.drawio.svg`
 * shows a rendered SVG preview (via `<img>` + blob URL so scripts do not run)
 * and keeps the source behind a toggle — the export XML is not what a reader
 * wants first.
 */

import { useEffect, useMemo, useState } from 'react';
import type { WorkspaceCodeViewV1 } from '../shared/workspace/v1/contracts';
import { isDrawioFileName, isDrawioSvgFileName } from '../shared/code-viewer/languages';
import { highlightCodeLines } from './code-viewer-highlight';

export interface CodeViewerProps {
  readonly view: WorkspaceCodeViewV1;
}

/** True when the body looks like SVG enough to risk a preview. */
export function looksLikeSvgContent(content: string): boolean {
  if (content.length === 0) return false;
  const head = content.slice(0, 2048);
  return /<svg[\s/>]/i.test(head);
}

export function CodeViewer({ view }: CodeViewerProps) {
  const drawioSvg = isDrawioSvgFileName(view.name);
  const drawioXml = isDrawioFileName(view.name);
  const canPreview = drawioSvg && looksLikeSvgContent(view.content) && !view.notice;
  const [showSource, setShowSource] = useState(!canPreview);

  // Reset the source toggle when opening a different file.
  useEffect(() => {
    setShowSource(!canPreview);
  }, [view.path, canPreview]);

  const previewUrl = useMemo(() => {
    if (!canPreview) return null;
    return URL.createObjectURL(new Blob([view.content], { type: 'image/svg+xml' }));
  }, [canPreview, view.content]);

  useEffect(() => {
    if (!previewUrl) return undefined;
    return () => {
      URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const lines = useMemo(
    () => (view.content === '' && view.notice ? [] : highlightCodeLines(view.content, view.language)),
    [view.content, view.language, view.notice],
  );
  const gutterCh = Math.max(3, String(Math.max(lines.length, 1)).length + 1);
  const langLabel = drawioXml ? 'drawio' : drawioSvg ? 'drawio.svg' : view.language;

  return (
    <div
      className="code-viewer"
      data-testid="code-viewer"
      data-path={view.path}
      data-drawio={drawioXml || drawioSvg ? '1' : undefined}
      data-drawio-preview={canPreview ? '1' : undefined}
    >
      <header className="code-viewer-head">
        <span className="code-viewer-dot" aria-hidden />
        <span className="code-viewer-name">{view.name}</span>
        {langLabel ? <span className="code-viewer-lang">{langLabel}</span> : null}
        {canPreview ? (
          <button
            type="button"
            className="code-viewer-toggle"
            data-testid="code-viewer-source-toggle"
            aria-pressed={showSource}
            onClick={() => setShowSource((value) => !value)}
          >
            {showSource ? 'Hide source' : 'Show source'}
          </button>
        ) : null}
        <span className="code-viewer-ro">Read only</span>
      </header>
      {drawioXml && lines.length > 0 ? (
        <div className="code-viewer-notice" role="status">
          Draw.io diagram (XML). Edit it in Draw.io / diagrams.net; Noto shows the source read-only.
        </div>
      ) : null}
      {canPreview && previewUrl ? (
        <div className="code-viewer-preview" data-testid="code-viewer-drawio-preview">
          <img className="code-viewer-preview-img" src={previewUrl} alt={view.name} />
        </div>
      ) : null}
      {view.notice && lines.length === 0 ? (
        <div className="code-viewer-notice" role="status">{view.notice}</div>
      ) : showSource ? (
        <>
          <div
            className="code-viewer-code"
            style={{ ['--code-viewer-gutter-ch' as string]: `${gutterCh}ch` }}
            spellCheck={false}
          >
            {lines.map((tokens, index) => (
              <div className="code-viewer-row" data-ln={String(index + 1)} key={index}>
                <span className="code-viewer-src">
                  {tokens.length === 0
                    ? null
                    : tokens.map((token, tokenIndex) => (
                      token.cls
                        ? <span className={`token ${token.cls}`} key={tokenIndex}>{token.text}</span>
                        : <span key={tokenIndex}>{token.text}</span>
                    ))}
                </span>
              </div>
            ))}
          </div>
          {view.notice ? (
            <div className="code-viewer-notice" role="status">{view.notice}</div>
          ) : null}
        </>
      ) : null}
    </div>
  );
}
