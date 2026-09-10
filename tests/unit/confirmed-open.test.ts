/**
 * Confirmed-open recording: success advances frecency once per transition;
 * the same front document does not double-count; failures never call it.
 */

import { describe, expect, it } from 'vitest';
import { ConfirmedOpenRecorder } from '../../src/shared/search/v1/confirmed-open';
import type { FrecencyStoreV1 } from '../../src/shared/search/v1/frecency';

const now = 1_700_000_000_000;

describe('ConfirmedOpenRecorder', () => {
  it('records after a successful open of a new path', () => {
    const recorder = new ConfirmedOpenRecorder();
    const first = recorder.recordAfterSuccessfulOpen({}, '/a.md', now);
    expect(first.recorded).toBe(true);
    expect(first.store['/a.md']).toEqual({ path: '/a.md', count: 1, lastOpenedAt: now });
    expect(recorder.current()).toBe('/a.md');
  });

  it('does not double-count the same transition', () => {
    const recorder = new ConfirmedOpenRecorder();
    const once = recorder.recordAfterSuccessfulOpen({}, '/a.md', now);
    const again = recorder.recordAfterSuccessfulOpen(once.store, '/a.md', now + 1);
    expect(again.recorded).toBe(false);
    expect(again.store).toBe(once.store);
    expect(again.store['/a.md'].count).toBe(1);
  });

  it('counts a move to a different path as a new open', () => {
    const recorder = new ConfirmedOpenRecorder();
    const a = recorder.recordAfterSuccessfulOpen({}, '/a.md', now);
    const b = recorder.recordAfterSuccessfulOpen(a.store, '/b.md', now + 5);
    expect(b.recorded).toBe(true);
    expect(b.store['/b.md'].count).toBe(1);
    const back = recorder.recordAfterSuccessfulOpen(b.store, '/a.md', now + 10);
    expect(back.recorded).toBe(true);
    expect(back.store['/a.md'].count).toBe(2);
    expect(back.store['/a.md'].lastOpenedAt).toBe(now + 10);
  });

  it('noteWithoutRecording suppresses the next identical path without a bump', () => {
    const recorder = new ConfirmedOpenRecorder();
    recorder.noteWithoutRecording('/a.md');
    const result = recorder.recordAfterSuccessfulOpen({}, '/a.md', now);
    expect(result.recorded).toBe(false);
    expect(result.store).toEqual({});
  });

  it('never mutates the store it was given', () => {
    const recorder = new ConfirmedOpenRecorder();
    const before: FrecencyStoreV1 = {};
    const after = recorder.recordAfterSuccessfulOpen(before, '/a.md', now);
    expect(before).toEqual({});
    expect(after.store).not.toBe(before);
  });
});
