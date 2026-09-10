/**
 * Neighbours of the note in front: what links here, what it links to, and
 * (when note-assistant has ranked them) what is probably about the same thing.
 *
 * Explicit lists are computed from the notes themselves. Related notes are
 * the plugin's ranking and only appear when `.note-assistant/graph.json` is
 * there. Mixing the two into one "graph" was why this rail sat empty in a
 * vault full of wiki links.
 */

import { useEffect, useState } from 'react';
import type { WorkspaceLinkV1, WorkspaceLinksReplyV1 } from '../shared/workspace/v1/contracts';

/** The reply, or the reason there is none, which the view says rather than guessing. */
export type LinksOutcome = { readonly reply: WorkspaceLinksReplyV1 } | { readonly error: string };

export interface RailLinksProps {
  readonly currentPath: string | null;
  readonly onLinks: (path: string) => Promise<LinksOutcome>;
  readonly onOpen: (path: string) => void;
  /** Bumped when a file event says the neighbourhood may have changed. */
  readonly refreshToken?: number;
}

function Section({ title, items, onOpen, testId }: {
  title: string;
  items: readonly WorkspaceLinkV1[];
  onOpen: (path: string) => void;
  testId: string;
}) {
  if (items.length === 0) return null;
  return (
    <section className="rail-links-section" data-testid={testId}>
      <h3 className="rail-links-kicker">{title} <span className="rail-links-count">{items.length}</span></h3>
      {items.map((item) => {
        const folder = item.relativePath.slice(0, Math.max(0, item.relativePath.lastIndexOf('/')));
        return (
          <button type="button" key={item.path} className="rail-hit-file rail-link" title={item.relativePath}
            data-testid="rail-link" onClick={() => onOpen(item.path)}>
            <span className="rail-hit-name">{item.title}</span>
            {folder.length > 0 && <span className="rail-hit-folder">{folder}</span>}
          </button>
        );
      })}
    </section>
  );
}

export function RailLinks({ currentPath, onLinks, onOpen, refreshToken = 0 }: RailLinksProps) {
  const [reply, setReply] = useState<WorkspaceLinksReplyV1 | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (currentPath === null) { setReply(null); setError(null); return; }
    let live = true;
    setLoading(true);
    void onLinks(currentPath).then((found) => {
      if (!live) return;
      if ('reply' in found) { setReply(found.reply); setError(null); } else { setReply(null); setError(found.error); }
      setLoading(false);
    });
    return () => { live = false; };
  }, [currentPath, onLinks, refreshToken]);

  if (currentPath === null) return <p className="rail-empty">Open a note to see what it is linked to.</p>;
  if (reply === null && error === null) return <p className="rail-empty">Reading links…</p>;
  void loading;
  if (error !== null) {
    return <p className="rail-empty" data-testid="links-status">The links could not be read: {error}</p>;
  }
  if (reply === null || !reply.available) {
    return (
      <p className="rail-empty" data-testid="links-status">
        Open a folder to see what notes link to each other.
      </p>
    );
  }
  if (!reply.known) {
    return <p className="rail-empty" data-testid="links-status">This note is not in the open folder.</p>;
  }
  const nothing = reply.backlinks.length + reply.links.length + reply.related.length === 0;
  return (
    <div className="rail-links" data-testid="links-panel">
      {nothing && (
        <p className="rail-empty" data-testid="links-status">
          Nothing links here yet.
        </p>
      )}
      <Section title="Linked from" items={reply.backlinks} onOpen={onOpen} testId="links-backlinks" />
      <Section title="Links to" items={reply.links} onOpen={onOpen} testId="links-out" />
      <Section title="Related" items={reply.related} onOpen={onOpen} testId="links-related" />
    </div>
  );
}
