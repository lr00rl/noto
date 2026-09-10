/**
 * Read-only view of a non-Markdown text or code file.
 *
 * The author's typora-plugin-lite `code-viewer` paints source with line numbers
 * and syntax colour, and never writes the file. Noto does the same: the pane
 * is contentEditable=false, tokens are built with textContent, and the gutter
 * is a CSS ::before so a copy never carries line numbers.
 */

import { useMemo } from 'react';
import type { WorkspaceCodeViewV1 } from '../shared/workspace/v1/contracts';
import { highlightCodeLines } from './code-viewer-highlight';

export interface CodeViewerProps {
  readonly view: WorkspaceCodeViewV1;
}

export function CodeViewer({ view }: CodeViewerProps) {
  const lines = useMemo(
    () => (view.content === '' && view.notice ? [] : highlightCodeLines(view.content, view.language)),
    [view.content, view.language, view.notice],
  );
  const gutterCh = Math.max(3, String(Math.max(lines.length, 1)).length + 1);

  return (
    <div className="code-viewer" data-testid="code-viewer" data-path={view.path}>
      <header className="code-viewer-head">
        <span className="code-viewer-dot" aria-hidden />
        <span className="code-viewer-name">{view.name}</span>
        {view.language ? <span className="code-viewer-lang">{view.language}</span> : null}
        <span className="code-viewer-ro">Read only</span>
      </header>
      {view.notice && lines.length === 0 ? (
        <div className="code-viewer-notice" role="status">{view.notice}</div>
      ) : (
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
      )}
    </div>
  );
}
