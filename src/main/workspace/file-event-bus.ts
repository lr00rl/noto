/**
 * In-process fan-out for `FileEventV1`.
 *
 * The folder watcher reports what the disk did. App code reports what it just
 * wrote, renamed, or removed. Those two streams overlap on every save: the
 * write is a rename over the path, the watcher sees it, and a subscriber that
 * treated both as truth would run twice. `suppress` is the mute for that echo,
 * keyed by absolute path, and it only drops `origin: 'disk'`. An app event on
 * a muted path still goes through, because that is the announcement.
 *
 * A move has two names. Suppressing the destination is not enough: the watcher
 * will also fire for the old path. `emit` of `moved` copies the remaining mute
 * window onto `from` so the caller names the destination once.
 *
 * `recent` is a ring of the last 64 accepted events, for a subscriber that
 * attached after the folder was already busy.
 */

import path from 'node:path';
import type { FileEventV1 } from '../../shared/workspace/v1/file-events';

export type FileEventListener = (event: FileEventV1) => void;

export interface FileEventBusOptions {
  /** Injected so a test does not have to wait in real time. */
  readonly now?: () => number;
}

const RING = 64;

export class FileEventBus {
  private readonly listeners = new Set<FileEventListener>();

  private readonly ring: FileEventV1[] = [];

  /** Absolute path -> epoch ms when disk events for it may fire again. */
  private readonly muted = new Map<string, number>();

  private readonly now: () => number;

  constructor(options: FileEventBusOptions = {}) {
    this.now = options.now ?? Date.now;
  }

  subscribe(listener: FileEventListener): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  /**
   * Ignore subsequent disk-origin events for `path` until `now+ms`.
   *
   * Our own atomic save is already an app event. The watcher will see the
   * rename and would otherwise double-fire. Call this on the path about to be
   * written; a `moved` emit copies the same window onto `from`.
   */
  suppress(pathToMute: string, ms: number): void {
    this.muted.set(key(pathToMute), this.now() + ms);
  }

  emit(event: FileEventV1): void {
    if (event.origin === 'disk' && this.silenced(event.path)) return;
    if (event.origin === 'disk' && event.kind === 'moved' && this.silenced(event.from)) return;

    if (event.kind === 'moved') {
      const until = this.remaining(event.path);
      if (until > this.now()) this.muted.set(key(event.from), until);
    }

    this.ring.push(event);
    if (this.ring.length > RING) this.ring.shift();
    for (const listener of [...this.listeners]) listener(event);
  }

  recent(): readonly FileEventV1[] {
    return this.ring.slice();
  }

  /** Drop the ring and the mute map. Listeners stay. Used when the folder changes. */
  reset(): void {
    this.ring.length = 0;
    this.muted.clear();
  }

  private silenced(filePath: string): boolean {
    return this.remaining(filePath) > this.now();
  }

  private remaining(filePath: string): number {
    const until = this.muted.get(key(filePath));
    if (until === undefined) return 0;
    if (until <= this.now()) {
      this.muted.delete(key(filePath));
      return 0;
    }
    return until;
  }
}

function key(filePath: string): string {
  return path.normalize(filePath);
}
