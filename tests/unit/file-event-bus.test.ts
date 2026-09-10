import { describe, expect, it } from 'vitest';
import { FileEventBus } from '../../src/main/workspace/file-event-bus';
import { isFileEventV1, type FileEventV1 } from '../../src/shared/workspace/v1/file-events';

function event(
  kind: Exclude<FileEventV1['kind'], 'moved'>,
  filePath: string,
  origin: FileEventV1['origin'],
  at = 1_000,
): FileEventV1 {
  return { version: 1, kind, path: filePath, origin, at };
}

function moved(to: string, from: string, origin: FileEventV1['origin'], at = 1_000): FileEventV1 {
  return { version: 1, kind: 'moved', path: to, from, origin, at };
}

describe('FileEventBus', () => {
  it('delivers to subscribers and stops after unsubscribe', () => {
    const bus = new FileEventBus({ now: () => 1_000 });
    const first: FileEventV1[] = [];
    const second: FileEventV1[] = [];
    const stopFirst = bus.subscribe((item) => first.push(item));
    bus.subscribe((item) => second.push(item));

    bus.emit(event('saved', '/vault/a.md', 'app'));
    stopFirst();
    bus.emit(event('saved', '/vault/b.md', 'app'));

    expect(first.map((item) => item.path)).toEqual(['/vault/a.md']);
    expect(second.map((item) => item.path)).toEqual(['/vault/a.md', '/vault/b.md']);
  });

  it('keeps only the last 64 accepted events', () => {
    const bus = new FileEventBus({ now: () => 1_000 });
    for (let index = 0; index < 70; index += 1) {
      bus.emit(event('saved', `/vault/${index}.md`, 'app', 1_000 + index));
    }
    const recent = bus.recent();
    expect(recent).toHaveLength(64);
    expect(recent[0]?.path).toBe('/vault/6.md');
    expect(recent[63]?.path).toBe('/vault/69.md');
  });

  it('drops a disk echo on a suppressed path, and still delivers an app event', () => {
    let now = 1_000;
    const bus = new FileEventBus({ now: () => now });
    const received: FileEventV1[] = [];
    bus.subscribe((item) => received.push(item));

    bus.suppress('/vault/note.md', 500);
    bus.emit(event('saved', '/vault/note.md', 'disk'));
    bus.emit(event('saved', '/vault/note.md', 'app'));
    expect(received).toEqual([event('saved', '/vault/note.md', 'app')]);
    expect(bus.recent()).toHaveLength(1);

    now = 1_500;
    bus.emit(event('saved', '/vault/note.md', 'disk', 1_500));
    expect(received).toHaveLength(2);
    expect(received[1]?.origin).toBe('disk');
  });

  it('reset clears the ring and mutes without dropping listeners', () => {
    const bus = new FileEventBus({ now: () => 1_000 });
    const received: FileEventV1[] = [];
    bus.subscribe((item) => received.push(item));

    bus.suppress('/vault/a.md', 2_000);
    bus.emit(event('saved', '/vault/a.md', 'app'));
    expect(bus.recent()).toHaveLength(1);

    bus.reset();
    expect(bus.recent()).toEqual([]);

    bus.emit(event('saved', '/vault/a.md', 'disk'));
    expect(received).toEqual([
      event('saved', '/vault/a.md', 'app'),
      event('saved', '/vault/a.md', 'disk'),
    ]);
  });

  it('copies a move mute onto from, so the old path is not echoed from disk', () => {
    const bus = new FileEventBus({ now: () => 1_000 });
    const received: FileEventV1[] = [];
    bus.subscribe((item) => received.push(item));

    bus.suppress('/vault/b.md', 400);
    bus.emit(moved('/vault/b.md', '/vault/a.md', 'app'));
    bus.emit(event('deleted', '/vault/a.md', 'disk'));
    bus.emit(event('created', '/vault/b.md', 'disk'));

    expect(received).toEqual([moved('/vault/b.md', '/vault/a.md', 'app')]);
    expect(isFileEventV1(received[0])).toBe(true);
  });
});

describe('isFileEventV1', () => {
  it('accepts created without from, and moved only with from', () => {
    expect(isFileEventV1(event('created', '/vault/a.md', 'disk'))).toBe(true);
    expect(isFileEventV1(moved('/vault/b.md', '/vault/a.md', 'app'))).toBe(true);
    expect(isFileEventV1({ version: 1, kind: 'moved', path: '/vault/b.md', origin: 'app', at: 1 })).toBe(false);
    expect(isFileEventV1({ ...event('saved', '/vault/a.md', 'app'), from: '/vault/old.md' })).toBe(false);
  });
});
