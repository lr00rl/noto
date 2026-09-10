/**
 * Confirm before a dirty buffer is replaced by the file on disk.
 *
 * Reloading is the right answer when another program wrote the note, but it
 * throws away unsaved edits. Asking once, with a way to keep those edits as a
 * copy first, is the tradeoff that makes the action safe to offer.
 */

import { useEffect, useRef } from 'react';

export interface ReloadConfirmDialogProps {
  readonly offerSaveCopy: boolean;
  readonly onReload: () => void;
  readonly onSaveCopy: () => void;
  readonly onCancel: () => void;
}

export function ReloadConfirmDialog({
  offerSaveCopy,
  onReload,
  onSaveCopy,
  onCancel,
}: ReloadConfirmDialogProps) {
  const cancelRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    cancelRef.current?.focus();
  }, []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      event.preventDefault();
      onCancel();
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [onCancel]);

  return (
    <div
      className="reload-confirm-scrim"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onCancel();
      }}
    >
      <div
        className="reload-confirm"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="reload-confirm-title"
        aria-describedby="reload-confirm-body"
        data-testid="reload-confirm"
      >
        <h2 id="reload-confirm-title" className="reload-confirm-title">Reload from disk?</h2>
        <p id="reload-confirm-body" className="reload-confirm-body">
          Reloading replaces what is on screen with the file on disk and discards
          your unsaved edits.
        </p>
        <div className="reload-confirm-actions">
          <button
            type="button"
            ref={cancelRef}
            data-testid="reload-confirm-cancel"
            onClick={onCancel}
          >
            Cancel
          </button>
          {offerSaveCopy && (
            <button
              type="button"
              data-testid="reload-confirm-save-copy"
              onClick={onSaveCopy}
            >
              Save a Copy first…
            </button>
          )}
          <button
            type="button"
            className="is-danger"
            data-testid="reload-confirm-reload"
            onClick={onReload}
          >
            Reload
          </button>
        </div>
      </div>
    </div>
  );
}
