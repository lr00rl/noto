/**
 * Record an open only after the host has confirmed it, and only once per
 * transition.
 *
 * Ported in spirit from typora-plugin-lite's ConfirmedOpenRecorder: frecency
 * and "recent" are rankings of what actually came to the front, not of what
 * was attempted. A failed open must not teach the ranking to offer the path
 * again, and two callers agreeing that the same path is now in front must not
 * count as two opens.
 *
 * Pure about the store it is given; the only mutable state is the last path
 * this recorder has already accepted, so a second call for the same transition
 * is a no-op.
 */

import { pruneStore, recordOpen, type FrecencyStoreV1 } from './frecency';

export interface ConfirmedOpenResult {
  readonly store: FrecencyStoreV1;
  /** False when the path was already the front document, or never recorded. */
  readonly recorded: boolean;
}

/**
 * Advances frecency only after a successful open of a *different* path.
 *
 * Call `recordAfterSuccessfulOpen` from the one place that has already seen
 * the host confirm the open. Call `noteWithoutRecording` when the front
 * document is known without that open having just happened (startup adopt,
 * trail replay), so the next real transition still counts once.
 */
export class ConfirmedOpenRecorder {
  private lastPath: string | null = null;

  /** The path most recently accepted, or null before the first one. */
  current(): string | null {
    return this.lastPath;
  }

  /**
   * Learn the front document without bumping counts.
   *
   * Used when the shell is catching up with a document that is already open
   * (startup, trail replay) so the next hand-driven transition is still the
   * first count for that move.
   */
  noteWithoutRecording(path: string): void {
    this.lastPath = path;
  }

  /**
   * Record a successful open. No-op when `path` is already the front document.
   */
  recordAfterSuccessfulOpen(
    store: FrecencyStoreV1,
    path: string,
    now: number,
  ): ConfirmedOpenResult {
    if (path === this.lastPath) return { store, recorded: false };
    this.lastPath = path;
    return { store: pruneStore(recordOpen(store, path, now), now), recorded: true };
  }
}
