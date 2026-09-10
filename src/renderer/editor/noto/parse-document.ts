/**
 * Open-path document parse, off the UI thread when a Worker is available.
 *
 * Overlapping main's parse by sending the text early was tried and reverted:
 * the renderer still blocked its own thread for the whole parse, so the window
 * never painted a preview. Moving the parse into a worker is the step that
 * actually stops the freeze; the spans come back ready for `docFromSpans`.
 */

import { splitBlocks, type BlockSpan } from '../../../shared/markdown/v3/blocks';

interface ParseRequest {
  readonly id: number;
  readonly text: string;
}

interface ParseResponse {
  readonly id: number;
  readonly spans: readonly BlockSpan[];
}

type Pending = {
  readonly resolve: (spans: readonly BlockSpan[]) => void;
  readonly reject: (error: unknown) => void;
};

let worker: Worker | null = null;
let workerFailed = false;
let nextId = 1;
const pending = new Map<number, Pending>();

/** The same split the worker runs, for tests and environments without workers. */
export function parseDocumentSpansSync(text: string): readonly BlockSpan[] {
  return splitBlocks(text).spans;
}

/**
 * Round-trip spans through the structured clone algorithm.
 *
 * The worker posts mdast trees this way. A clone that still builds the same
 * ProseMirror document is the contract the open path depends on.
 */
export function cloneParsedSpans(spans: readonly BlockSpan[]): readonly BlockSpan[] {
  return structuredClone(spans);
}

function failAll(error: unknown): void {
  workerFailed = true;
  const waiting = [...pending.values()];
  pending.clear();
  if (worker) {
    worker.terminate();
    worker = null;
  }
  for (const entry of waiting) entry.reject(error);
}

function ensureWorker(): Worker | null {
  if (workerFailed) return null;
  if (typeof Worker === 'undefined') {
    workerFailed = true;
    return null;
  }
  if (worker) return worker;
  try {
    const next = new Worker(new URL('./parse-document.worker.ts', import.meta.url), {
      type: 'module',
    });
    next.onmessage = (event: MessageEvent<ParseResponse>) => {
      const entry = pending.get(event.data.id);
      if (!entry) return;
      pending.delete(event.data.id);
      entry.resolve(event.data.spans);
    };
    next.onerror = (event) => {
      failAll(event.error ?? new Error(event.message || 'Document parse worker failed'));
    };
    next.onmessageerror = () => {
      failAll(new Error('Document parse worker could not read a reply'));
    };
    worker = next;
    return worker;
  } catch {
    workerFailed = true;
    return null;
  }
}

/**
 * Split `text` into block spans, preferring a dedicated worker so the UI
 * thread is free to paint while micromark runs.
 *
 * Falls back to the synchronous split when workers are unavailable (unit
 * tests, or a worker that failed to start), so open still succeeds.
 */
export async function parseDocumentSpans(text: string): Promise<readonly BlockSpan[]> {
  const active = ensureWorker();
  if (!active) return parseDocumentSpansSync(text);

  const id = nextId;
  nextId += 1;
  return new Promise<readonly BlockSpan[]>((resolve, reject) => {
    pending.set(id, {
      resolve,
      reject: (error) => {
        // A dead worker must not strand an open: finish on this thread instead.
        try {
          resolve(parseDocumentSpansSync(text));
        } catch (fallbackError) {
          reject(fallbackError ?? error);
        }
      },
    });
    try {
      active.postMessage({ id, text } satisfies ParseRequest);
    } catch (error) {
      pending.delete(id);
      workerFailed = true;
      try {
        active.terminate();
      } catch {
        // Already gone.
      }
      worker = null;
      resolve(parseDocumentSpansSync(text));
    }
  });
}
