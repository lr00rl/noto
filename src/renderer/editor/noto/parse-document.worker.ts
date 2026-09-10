/**
 * Parse a full markdown document into block spans on a worker thread.
 *
 * The renderer used to run this on the UI thread, which froze the window for
 * the whole micromark pass on a large open. The spans, including their mdast
 * nodes, travel back by structured clone; building the ProseMirror document
 * from them stays on the UI thread and is cheap.
 */

import { splitBlocks } from '../../../shared/markdown/v3/blocks';

interface ParseRequest {
  readonly id: number;
  readonly text: string;
}

self.onmessage = (event: MessageEvent<ParseRequest>) => {
  const { id, text } = event.data;
  const spans = splitBlocks(text).spans;
  self.postMessage({ id, spans });
};
