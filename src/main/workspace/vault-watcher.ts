/**
 * Watching the open folder for changes this process did not make.
 *
 * `DocumentWatcher` holds one file and has to re-arm after every save, because
 * a rename-over leaves `fs.watch` holding a file nothing points at. This one
 * holds the folder, so a rename inside it is just another event. The noise is
 * the same: an atomic save is a burst of `rename` and `change` for one logical
 * write, and reading between two of those would invent a half-finished file.
 * Reports are therefore debounced, 300 ms after the last event, with a 2 s
 * ceiling so a continuous writer cannot push the report back forever.
 *
 * Classification is existence against a set of paths already seen: gone is
 * deleted, new is created, still there is saved. Directories are skipped.
 * Only the extensions the file index already treats as openable are reported,
 * so a sidecar or a `.tmp` from our own save does not become a note event.
 *
 * yagni: disk `moved` would need both paths, which Node does not give. App
 * code that renames should emit `moved` itself. A Finder rename shows up as
 * deleted plus created.
 */

import { watch, realpathSync, type FSWatcher } from 'node:fs';
import { readdir, realpath, stat } from 'node:fs/promises';
import path from 'node:path';
import { isEditableFile, isInside } from './file-tree';
import type { FileEventBus } from './file-event-bus';
import type { FileEventKindV1 } from '../../shared/workspace/v1/file-events';

const SETTLE_MS = 300;
const CEILING_MS = 2_000;

/** Path segments whose children are never notes, even if one is markdown. */
const IGNORED = new Set(['node_modules', '.git', '.noto']);

export interface VaultWatcherOptions {
  readonly bus: FileEventBus;
  /** Injected so a test does not have to wait in real time. */
  readonly now?: () => number;
}

type Pending = {
  ceilingAt: number;
  timer: NodeJS.Timeout | null;
};

export class VaultWatcher {
  private watcher: FSWatcher | null = null;

  private root: string | null = null;

  private generation = 0;

  private readonly seen = new Set<string>();

  private readonly pending = new Map<string, Pending>();

  private readonly bus: FileEventBus;

  private readonly now: () => number;

  constructor(options: VaultWatcherOptions) {
    this.bus = options.bus;
    this.now = options.now ?? Date.now;
  }

  /** Watch `root` recursively, dropping whatever folder was watched before. */
  arm(root: string): void {
    this.close();
    this.generation += 1;
    const generation = this.generation;
    let realRoot: string;
    try {
      realRoot = realpathSync(root);
    } catch {
      return;
    }
    this.root = realRoot;
    try {
      const watcher = watch(realRoot, { recursive: true, persistent: false }, (_kind, filename) => {
        if (generation !== this.generation) return;
        const filePath = this.resolveEvent(filename);
        if (filePath === null) return;
        this.schedule(filePath);
      });
      watcher.on('error', () => {
        if (generation !== this.generation) return;
        this.closeHandles();
      });
      this.watcher = watcher;
    } catch {
      this.root = null;
      return;
    }
    void this.seed(realRoot, generation);
  }

  close(): void {
    this.generation += 1;
    this.closeHandles();
    this.root = null;
    this.seen.clear();
  }

  private closeHandles(): void {
    this.watcher?.close();
    this.watcher = null;
    for (const pending of this.pending.values()) {
      if (pending.timer) clearTimeout(pending.timer);
    }
    this.pending.clear();
  }

  /**
   * Report once the burst settles, and at the ceiling regardless.
   *
   * A file being written continuously would otherwise push the report back
   * forever and the reader would never be told anything changed.
   */
  private schedule(filePath: string): void {
    const now = this.now();
    let pending = this.pending.get(filePath);
    if (!pending) {
      pending = { ceilingAt: now + CEILING_MS, timer: null };
      this.pending.set(filePath, pending);
    }
    if (pending.timer) clearTimeout(pending.timer);
    const wait = Math.max(0, Math.min(SETTLE_MS, pending.ceilingAt - now));
    const generation = this.generation;
    pending.timer = setTimeout(() => {
      this.pending.delete(filePath);
      if (generation !== this.generation) return;
      void this.flush(filePath, generation);
    }, wait);
  }

  private async flush(filePath: string, generation: number): Promise<void> {
    if (generation !== this.generation || this.root === null) return;

    let exists = false;
    let directory = false;
    try {
      const info = await stat(filePath);
      exists = true;
      directory = info.isDirectory();
    } catch {
      exists = false;
    }
    if (generation !== this.generation) return;

    if (!exists) {
      this.emitGone(filePath);
      return;
    }
    if (directory) return;

    let real: string;
    try {
      real = await realpath(filePath);
    } catch {
      return;
    }
    if (generation !== this.generation || this.root === null) return;
    if (!this.confined(real) || ignored(this.root, real) || !isEditableFile(real)) return;

    const kind: Exclude<FileEventKindV1, 'moved'> = this.seen.has(real) ? 'saved' : 'created';
    this.seen.add(real);
    this.bus.emit({ version: 1, kind, path: real, origin: 'disk', at: this.now() });
  }

  /**
   * A path that no longer exists. If it was a folder, every note we had seen
   * under it is gone too: the watcher often reports the directory and not each
   * child, and leaving those paths in `seen` would turn the next create at the
   * same name into a save.
   */
  private emitGone(filePath: string): void {
    const victims = [...this.seen].filter((known) => known === filePath || isInside(filePath, known));
    if (victims.length === 0) return;
    const at = this.now();
    for (const victim of victims) {
      this.seen.delete(victim);
      this.bus.emit({ version: 1, kind: 'deleted', path: victim, origin: 'disk', at });
    }
  }

  private resolveEvent(filename: string | Buffer | null): string | null {
    const root = this.root;
    if (root === null || filename == null) return null;
    const name = typeof filename === 'string' ? filename : filename.toString();
    if (name.length === 0) return null;
    const absolute = path.resolve(root, name);
    if (!this.confined(absolute) || ignored(root, absolute)) return null;
    if (!isEditableFile(absolute) && !this.seen.has(absolute) && !this.underSeen(absolute)) return null;
    return absolute;
  }

  private confined(target: string): boolean {
    const root = this.root;
    if (root === null) return false;
    return target === root || isInside(root, target);
  }

  private underSeen(directory: string): boolean {
    for (const known of this.seen) {
      if (isInside(directory, known)) return true;
    }
    return false;
  }

  private async seed(root: string, generation: number): Promise<void> {
    const files = await collectOpenable(root);
    if (generation !== this.generation) return;
    for (const file of files) this.seen.add(file);
  }
}

function ignored(root: string, target: string): boolean {
  const relative = path.relative(root, target);
  if (relative.length === 0) return false;
  return relative.split(path.sep).some((segment) => IGNORED.has(segment));
}

async function collectOpenable(root: string): Promise<string[]> {
  const files: string[] = [];
  const queue = [root];
  while (queue.length > 0) {
    const directory = queue.pop();
    if (directory === undefined) break;
    let listing;
    try {
      listing = await readdir(directory, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const entry of listing) {
      if (IGNORED.has(entry.name)) continue;
      const full = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        queue.push(full);
        continue;
      }
      if (!entry.isFile() || !isEditableFile(full)) continue;
      const real = await realpath(full).catch(() => null);
      if (!real || !(real === root || isInside(root, real))) continue;
      files.push(real);
    }
  }
  return files;
}
