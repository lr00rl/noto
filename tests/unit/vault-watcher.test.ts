import { mkdir, mkdtemp, realpath, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { FileEventBus } from '../../src/main/workspace/file-event-bus';
import { VaultWatcher } from '../../src/main/workspace/vault-watcher';
import type { FileEventV1 } from '../../src/shared/workspace/v1/file-events';

const roots: string[] = [];
const watchers: VaultWatcher[] = [];

afterEach(async () => {
  watchers.splice(0).forEach((watcher) => watcher.close());
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

async function fixture(): Promise<string> {
  const root = await realpath(await mkdtemp(path.join(os.tmpdir(), 'noto-vault-')));
  roots.push(root);
  return root;
}

/** Long enough for the trailing report, which exists so a burst is one report. */
const settle = () => new Promise((resolve) => setTimeout(resolve, 700));

function armed(root: string): { bus: FileEventBus; events: FileEventV1[] } {
  const bus = new FileEventBus();
  const events: FileEventV1[] = [];
  bus.subscribe((event) => events.push(event));
  const watcher = new VaultWatcher({ bus });
  watchers.push(watcher);
  watcher.arm(root);
  return { bus, events };
}

describe('VaultWatcher', () => {
  it('reports a new file as created, a write as saved, and an unlink as deleted', async () => {
    const root = await fixture();
    const { events } = armed(root);
    const file = path.join(root, 'note.md');

    await writeFile(file, '# Note\n');
    await settle();
    expect(events.map((event) => event.kind)).toEqual(['created']);
    expect(events[0]?.path).toBe(file);
    expect(events[0]?.origin).toBe('disk');

    await writeFile(file, '# Changed\n');
    await settle();
    expect(events.map((event) => event.kind)).toEqual(['created', 'saved']);
    expect(events[1]?.path).toBe(file);

    await rm(file);
    await settle();
    expect(events.map((event) => event.kind)).toEqual(['created', 'saved', 'deleted']);
    expect(events[2]?.path).toBe(file);
    expect(events.every((event) => event.origin === 'disk')).toBe(true);
  });

  it('says nothing after it is closed', async () => {
    const root = await fixture();
    const { events } = armed(root);
    watchers[0]?.close();
    await writeFile(path.join(root, 'note.md'), '# Note\n');
    await settle();
    expect(events).toEqual([]);
  });

  it('ignores a markdown file under node_modules', async () => {
    const root = await fixture();
    const { events } = armed(root);
    const nested = path.join(root, 'node_modules', 'pkg');
    await mkdir(nested, { recursive: true });
    await writeFile(path.join(nested, 'readme.md'), '# Dep\n');
    await settle();
    expect(events).toEqual([]);
  });
});
