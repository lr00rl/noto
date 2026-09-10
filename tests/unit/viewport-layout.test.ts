import { describe, expect, it } from 'vitest';
import { EditorState, TextSelection } from 'prosemirror-state';
import type { Decoration } from 'prosemirror-view';
import { splitBlocks } from '../../src/shared/markdown/v3/blocks';
import { docFromSpans } from '../../src/shared/markdown/v3/pm/from-mdast';
import {
  LAYOUT_LIVE_CLASS,
  liveTopLevelIndices,
  viewportLayoutKey,
  viewportLayoutPlugin,
} from '../../src/renderer/editor/noto/viewport-layout';

function stateFor(markdown: string, caretInBlock = 0): EditorState {
  const doc = docFromSpans(splitBlocks(markdown).spans);
  let position = 0;
  for (let index = 0; index < caretInBlock; index += 1) position += doc.child(index).nodeSize;
  const state = EditorState.create({ doc, plugins: [viewportLayoutPlugin()] });
  return state.apply(state.tr.setSelection(TextSelection.create(doc, Math.min(position + 1, doc.content.size))));
}

function liveClasses(state: EditorState): string[] {
  return viewportLayoutKey.getState(state)!.find()
    .map((decoration) => (decoration as Decoration & { type: { attrs?: { class?: string } } }).type.attrs?.class ?? '');
}

const FIVE = ['One.', '', 'Two.', '', 'Three.', '', 'Four.', '', 'Five.'].join('\n');

describe('which top level blocks stay fully painted', () => {
  it('keeps the selection and its neighbours', () => {
    const state = stateFor(FIVE, 2);
    expect(liveTopLevelIndices(state.doc, state.selection)).toEqual([1, 2, 3]);
    expect(liveClasses(state)).toEqual([
      LAYOUT_LIVE_CLASS, LAYOUT_LIVE_CLASS, LAYOUT_LIVE_CLASS,
    ]);
  });

  it('does not walk past the ends of the document', () => {
    const atStart = stateFor(FIVE, 0);
    expect(liveTopLevelIndices(atStart.doc, atStart.selection)).toEqual([0, 1]);
    const atEnd = stateFor(FIVE, 4);
    expect(liveTopLevelIndices(atEnd.doc, atEnd.selection)).toEqual([3, 4]);
  });

  it('widens to cover a selection that spans several blocks', () => {
    let state = stateFor(FIVE, 1);
    const from = 1;
    let position = 0;
    for (let index = 0; index < 3; index += 1) position += state.doc.child(index).nodeSize;
    const to = position - 1;
    state = state.apply(state.tr.setSelection(TextSelection.create(state.doc, from, to)));
    expect(liveTopLevelIndices(state.doc, state.selection)).toEqual([0, 1, 2, 3]);
  });

  it('moves the live window when the caret does', () => {
    let state = stateFor(FIVE, 0);
    expect(liveTopLevelIndices(state.doc, state.selection)).toEqual([0, 1]);
    let position = 0;
    for (let index = 0; index < 3; index += 1) position += state.doc.child(index).nodeSize;
    state = state.apply(state.tr.setSelection(TextSelection.create(state.doc, position + 1)));
    expect(liveTopLevelIndices(state.doc, state.selection)).toEqual([2, 3, 4]);
  });

  it('stays cheap on a long document: only the window is decorated', () => {
    const source = `${Array.from({ length: 500 }, (_, index) => `Paragraph ${index}.`).join('\n\n')}\n`;
    const state = stateFor(source, 250);
    expect(liveTopLevelIndices(state.doc, state.selection)).toEqual([249, 250, 251]);
    expect(liveClasses(state)).toHaveLength(3);
  });
});
