import { describe, expect, it } from 'vitest';
import { EditorState, TextSelection } from 'prosemirror-state';
import { splitBlocks } from '../../src/shared/markdown/v3/blocks';
import { docFromSpans } from '../../src/shared/markdown/v3/pm/from-mdast';
import { blockToMarkdown } from '../../src/shared/markdown/v3/pm/to-mdast';
import {
  CHECK_STAMP_RE,
  compareTaskSortKeys,
  formatCheckStamp,
  parseCheckStamp,
  rewriteTaskText,
  sortedTaskOrder,
  sortTasks,
  stripCheckStamp,
  toggleTaskStatus,
} from '../../src/renderer/editor/noto/todo-manager';
import type { Command } from 'prosemirror-state';

const DAY = new Date(2026, 8, 10); // 2026-09-10 local

function stateFor(markdown: string, from: number, to = from): EditorState {
  const state = EditorState.create({ doc: docFromSpans(splitBlocks(markdown).spans) });
  return state.apply(state.tr.setSelection(TextSelection.create(state.doc, from, to)));
}

function run(state: EditorState, command: Command): string | null {
  let next: EditorState | null = null;
  const handled = command(state, (tr) => { next = state.apply(tr); });
  if (!handled || next === null) return null;
  const parts: string[] = [];
  (next as EditorState).doc.forEach((block) => {
    parts.push(blockToMarkdown(block));
  });
  return parts.join('\n\n');
}

describe('the check stamp', () => {
  it('formats, strips and rewrites a line', () => {
    expect(formatCheckStamp(DAY)).toBe(' ✅ 2026-09-10');
    expect(stripCheckStamp('buy milk ✅ 2026-01-01')).toBe('buy milk');
    expect(stripCheckStamp('buy milk')).toBe('buy milk');
    expect(parseCheckStamp('done ✅ 2026-09-10')).toBe('2026-09-10');
    expect(parseCheckStamp('open')).toBeNull();
    expect(rewriteTaskText('buy milk', true, DAY)).toBe('buy milk ✅ 2026-09-10');
    expect(rewriteTaskText('buy milk ✅ 2026-01-01', true, DAY)).toBe('buy milk ✅ 2026-09-10');
    expect(rewriteTaskText('buy milk ✅ 2026-01-01', false, DAY)).toBe('buy milk');
    expect(CHECK_STAMP_RE.test('x ✅ 2026-09-10')).toBe(true);
  });
});

describe('sorting keys', () => {
  it('puts open tasks first, then done by stamp', () => {
    const keys = [
      { checked: true, stamp: '2026-02-01', index: 0 },
      { checked: false, stamp: null, index: 1 },
      { checked: true, stamp: '2026-01-01', index: 2 },
      { checked: false, stamp: null, index: 3 },
    ];
    expect(sortedTaskOrder(keys)).toEqual([1, 3, 2, 0]);
    expect(compareTaskSortKeys(keys[1]!, keys[0]!)).toBeLessThan(0);
  });
});

describe('toggling with a stamp', () => {
  it('writes today when checking and clears it when unchecking', () => {
    const stamp = { enabled: () => true, now: () => DAY };
    expect(run(stateFor('- [ ] milk', 4), toggleTaskStatus(true, stamp)))
      .toBe('- [x] milk ✅ 2026-09-10');
    expect(run(stateFor('- [x] milk ✅ 2026-09-10', 4), toggleTaskStatus(false, stamp)))
      .toBe('- [ ] milk');
  });

  it('leaves the line alone when the stamp is switched off', () => {
    const stamp = { enabled: () => false, now: () => DAY };
    expect(run(stateFor('- [ ] milk', 4), toggleTaskStatus(true, stamp)))
      .toBe('- [x] milk');
  });
});

describe('sorting the list', () => {
  it('floats open tasks above done ones', () => {
    const source = [
      '- [x] old ✅ 2026-01-01',
      '- [ ] open',
      '- [x] newer ✅ 2026-02-01',
    ].join('\n');
    const state = EditorState.create({ doc: docFromSpans(splitBlocks(source).spans) });
    const markdown = run(state, sortTasks);
    expect(markdown).toBe([
      '- [ ] open',
      '- [x] old ✅ 2026-01-01',
      '- [x] newer ✅ 2026-02-01',
    ].join('\n'));
  });

  it('does nothing when the list is already ordered', () => {
    const source = '- [ ] a\n- [x] b';
    const state = EditorState.create({ doc: docFromSpans(splitBlocks(source).spans) });
    expect(sortTasks(state, undefined)).toBe(false);
  });
});
