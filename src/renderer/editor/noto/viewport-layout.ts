/**
 * Long-document painting without breaking markdown input rules.
 *
 * Layout dominates a keystroke on a multi-megabyte file (see
 * `docs/performance/measurements.md`). Blanket `content-visibility: auto` on
 * every top-level block cut that cost by about forty percent on the two
 * megabyte corpus, and then broke input rules: the property implies style
 * containment, and with that set on the block being typed into ProseMirror can
 * no longer read the DOM back after a keystroke. Heading and task markers stop
 * converting.
 *
 * The fix is the same paint deferral, applied only to blocks that are not
 * being edited. Off-screen blocks keep `content-visibility: auto` from the
 * stylesheet; the blocks around the selection carry `noto-layout-live`, which
 * forces them fully painted. The active-block class already marks the selection
 * for other reasons; this plugin widens that to a one-block neighbourhood so a
 * split or a join still has a fully painted neighbour for the DOM read.
 *
 * This is not a virtual scroller. The DOM still holds every block. What it
 * removes is the engine's obligation to lay out and paint the ones nobody can
 * see. A stubbing scroller that drops off-screen content from the tree is the
 * next step if this is not enough; it is deliberately not this change.
 */

import { Plugin, PluginKey, type EditorState, type Selection } from 'prosemirror-state';
import type { Node as ProseNode } from 'prosemirror-model';
import { Decoration, DecorationSet } from 'prosemirror-view';

export const viewportLayoutKey = new PluginKey<DecorationSet>('noto-viewport-layout');

/** Forced fully painted; paired with the rule in `noto-editor.scss`. */
export const LAYOUT_LIVE_CLASS = 'noto-layout-live';

/** How many top-level neighbours of the selection stay fully painted. */
export const LAYOUT_LIVE_RADIUS = 1;

/**
 * Top-level block indices that must stay fully painted.
 *
 * The selection's own blocks, plus `radius` neighbours on each side. Indices
 * rather than positions, so a later decoration pass can walk the doc once.
 */
export function liveTopLevelIndices(
  doc: ProseNode,
  selection: Selection,
  radius: number = LAYOUT_LIVE_RADIUS,
): number[] {
  const last = doc.childCount - 1;
  if (last < 0) return [];

  let fromIndex = selection.$from.index(0);
  let toIndex = selection.$to.index(0);
  if (fromIndex > toIndex) {
    const swap = fromIndex;
    fromIndex = toIndex;
    toIndex = swap;
  }

  const start = Math.max(0, fromIndex - radius);
  const end = Math.min(last, toIndex + radius);
  const indices: number[] = [];
  for (let index = start; index <= end; index += 1) indices.push(index);
  return indices;
}

function liveDecorations(state: EditorState): DecorationSet {
  const indices = liveTopLevelIndices(state.doc, state.selection);
  if (indices.length === 0) return DecorationSet.empty;

  const decorations: Decoration[] = [];
  let position = 0;
  let index = 0;
  const wanted = new Set(indices);
  state.doc.forEach((child) => {
    if (wanted.has(index)) {
      decorations.push(Decoration.node(position, position + child.nodeSize, {
        class: LAYOUT_LIVE_CLASS,
      }));
    }
    position += child.nodeSize;
    index += 1;
  });
  return DecorationSet.create(state.doc, decorations);
}

export function viewportLayoutPlugin(): Plugin<DecorationSet> {
  return new Plugin<DecorationSet>({
    key: viewportLayoutKey,
    state: {
      init: (_config, state) => liveDecorations(state),
      apply: (transaction, previous, _oldState, newState) =>
        (transaction.docChanged || transaction.selectionSet ? liveDecorations(newState) : previous),
    },
    props: {
      decorations: (state) => viewportLayoutKey.getState(state),
    },
  });
}
