/**
 * The author's todo-manager: a check gets a date, and open tasks float up.
 *
 * The Typora plugin's brief was "按 todo 事项自动排序，自动添加 check 时间" and
 * "无插件情况下友好". So the stamp is ordinary characters in the file —
 * ` ✅ 2026-09-10` — and sorting only rearranges list items. Nothing about the
 * drawing needs a plugin, and a note opened elsewhere still reads.
 */

import type { Command, Transaction } from 'prosemirror-state';
import type { Node as ProseNode } from 'prosemirror-model';
import { notoSchema } from '../../../shared/markdown/v3/pm/schema';

/** Trailing stamp a check writes: space, mark, space, ISO date. */
export const CHECK_STAMP_RE = /\s*✅\s+(\d{4}-\d{2}-\d{2})\s*$/u;

/** The characters appended when a task is checked. */
export function formatCheckStamp(date: Date): string {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, '0');
  const d = String(date.getDate()).padStart(2, '0');
  return ` ✅ ${y}-${m}-${d}`;
}

/** Drop a trailing check stamp, leaving the words alone. */
export function stripCheckStamp(text: string): string {
  return text.replace(CHECK_STAMP_RE, '');
}

/** The ISO date a line already carries, or null when it has none. */
export function parseCheckStamp(text: string): string | null {
  const match = CHECK_STAMP_RE.exec(text);
  return match ? match[1]! : null;
}

/**
 * The text a check or uncheck should leave on the line.
 *
 * Checking replaces any older stamp so a re-tick records today. Unchecking
 * strips the stamp: an open task does not keep a finished date.
 */
export function rewriteTaskText(text: string, checked: boolean, date: Date): string {
  const base = stripCheckStamp(text);
  return checked ? `${base}${formatCheckStamp(date)}` : base;
}

/** One task in a list, enough to decide where it sorts. */
export interface TaskSortKey {
  readonly checked: boolean;
  readonly stamp: string | null;
  readonly index: number;
}

/**
 * Open tasks first (stable), then done ones by stamp ascending, then by the
 * order they had. A missing stamp sorts after a dated one among the done.
 */
export function compareTaskSortKeys(a: TaskSortKey, b: TaskSortKey): number {
  if (a.checked !== b.checked) return a.checked ? 1 : -1;
  if (a.checked && b.checked) {
    if (a.stamp && b.stamp && a.stamp !== b.stamp) return a.stamp < b.stamp ? -1 : 1;
    if (a.stamp && !b.stamp) return -1;
    if (!a.stamp && b.stamp) return 1;
  }
  return a.index - b.index;
}

/**
 * New order of indices for one contiguous run of tasks.
 *
 * Pure so the ordering rule is tested without a document.
 */
export function sortedTaskOrder(keys: readonly TaskSortKey[]): number[] {
  return keys
    .slice()
    .sort(compareTaskSortKeys)
    .map((key) => key.index);
}

/** Map a character offset into a paragraph's textContent to a document position. */
export function positionAtTextOffset(
  parent: ProseNode,
  contentStart: number,
  offset: number,
): number {
  let remaining = offset;
  let pos = contentStart;
  for (let i = 0; i < parent.childCount; i += 1) {
    const child = parent.child(i);
    if (child.isText) {
      if (remaining <= child.nodeSize) return pos + remaining;
      remaining -= child.nodeSize;
      pos += child.nodeSize;
      continue;
    }
    const size = child.textContent.length;
    if (remaining <= size) return positionAtTextOffset(child, pos + 1, remaining);
    remaining -= size;
    pos += child.nodeSize;
  }
  return pos;
}

/**
 * Write or clear the trailing stamp on a list item's first paragraph.
 *
 * The rest of the paragraph's marks stay: only the stamp at the end is
 * inserted, replaced or deleted. An item with no paragraph is left alone.
 */
export function applyCheckStamp(
  tr: Transaction,
  itemPos: number,
  item: ProseNode,
  checked: boolean,
  date: Date,
): Transaction {
  const para = item.firstChild;
  if (!para || para.type !== notoSchema.nodes.paragraph) return tr;
  const text = para.textContent;
  const next = rewriteTaskText(text, checked, date);
  if (next === text) return tr;

  const contentStart = itemPos + 1 + 1;
  const contentEnd = itemPos + 1 + para.nodeSize - 1;
  const match = CHECK_STAMP_RE.exec(text);

  if (!checked) {
    if (!match) return tr;
    const stampStart = positionAtTextOffset(para, contentStart, text.length - match[0].length);
    return tr.delete(stampStart, contentEnd);
  }

  const stamp = formatCheckStamp(date);
  if (match) {
    const stampStart = positionAtTextOffset(para, contentStart, text.length - match[0].length);
    return tr.replaceWith(stampStart, contentEnd, notoSchema.text(stamp));
  }
  return tr.insert(contentEnd, notoSchema.text(stamp));
}

export interface TaskStampOptions {
  /** Whether a check writes a stamp. Defaults to on. */
  readonly enabled?: () => boolean;
  /** The clock a check reads. Injected so a test can pin the day. */
  readonly now?: () => Date;
}

/**
 * Tick or untick the task the caret is in, and optionally stamp the line.
 *
 * Same shape as the older toggle: only an item that is already a task has a
 * state to flip, and setting it to what it already is is a no-op.
 */
export function toggleTaskStatus(
  to?: boolean,
  options: TaskStampOptions = {},
): Command {
  return (state, dispatch) => {
    const { $from } = state.selection;
    for (let depth = $from.depth; depth > 0; depth -= 1) {
      const node = $from.node(depth);
      if (node.type !== notoSchema.nodes.list_item) continue;
      if (node.attrs.checked === null) return false;
      const checked = to ?? !node.attrs.checked;
      if (checked === node.attrs.checked) return true;
      if (!dispatch) return true;
      const itemPos = $from.before(depth);
      let tr = state.tr.setNodeMarkup(itemPos, undefined, { ...node.attrs, checked });
      if (options.enabled?.() ?? true) {
        // Positions after setNodeMarkup are unchanged: attrs do not shift offsets.
        tr = applyCheckStamp(tr, itemPos, node, checked, (options.now ?? (() => new Date()))());
      }
      dispatch(tr);
      return true;
    }
    return false;
  };
}

/**
 * Reorder every contiguous run of tasks in the document: open first, then done
 * by check date. Ordinary bullets and nested lists stay where they are and
 * break a run, so a list that mixes notes with tasks is not scrambled.
 */
export const sortTasks: Command = (state, dispatch) => {
  type Run = { listPos: number; from: number; to: number; keys: TaskSortKey[] };
  const runs: Run[] = [];

  state.doc.descendants((node, pos) => {
    if (node.type !== notoSchema.nodes.bullet_list
      && node.type !== notoSchema.nodes.ordered_list) {
      return true;
    }
    let runStart = -1;
    const keys: TaskSortKey[] = [];
    const flush = (from: number, to: number, runKeys: TaskSortKey[]) => {
      if (runKeys.length < 2) return;
      const order = sortedTaskOrder(runKeys);
      if (order.every((index, at) => index === runKeys[at]!.index)) return;
      runs.push({ listPos: pos, from, to, keys: runKeys.slice() });
    };

    for (let i = 0; i < node.childCount; i += 1) {
      const child = node.child(i);
      const isTask = child.type === notoSchema.nodes.list_item && child.attrs.checked !== null;
      if (isTask) {
        if (runStart < 0) runStart = i;
        keys.push({
          checked: Boolean(child.attrs.checked),
          stamp: parseCheckStamp(child.textContent),
          index: i,
        });
      } else if (runStart >= 0) {
        flush(runStart, i, keys);
        runStart = -1;
        keys.length = 0;
      }
    }
    if (runStart >= 0) flush(runStart, node.childCount, keys);
    return true;
  });

  if (runs.length === 0) return false;
  if (!dispatch) return true;

  let tr = state.tr;
  // Deepest / later first so earlier list positions stay valid as we rewrite.
  runs.sort((a, b) => b.listPos - a.listPos);
  for (const run of runs) {
    const list = tr.doc.nodeAt(run.listPos);
    if (!list) continue;
    const order = sortedTaskOrder(run.keys);
    const rebuilt: ProseNode[] = [];
    for (let i = 0; i < list.childCount; i += 1) {
      if (i === run.from) {
        for (const index of order) rebuilt.push(list.child(index));
        i = run.to - 1;
        continue;
      }
      rebuilt.push(list.child(i));
    }
    tr = tr.replaceWith(
      run.listPos + 1,
      run.listPos + list.nodeSize - 1,
      rebuilt,
    );
  }
  dispatch(tr.scrollIntoView());
  return true;
};
