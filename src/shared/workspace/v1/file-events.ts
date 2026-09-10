/**
 * A file in the open folder appeared, changed, moved, or vanished.
 *
 * Later automation subscribes to these. The events are a built-in bus in main,
 * not user scripts: nothing here is evaluated, and a listener cannot reach
 * beyond what the workspace already does with files. `origin` says who did it,
 * because an atomic save this process just performed will also show up as a
 * disk event a moment later, and subscribers should not treat that echo as a
 * second change.
 *
 * `from` is only on `moved`. Node's folder watcher does not give both paths,
 * so a disk rename is not synthesized into a move; app code that renamed the
 * file is the one that can fill `from`.
 */

export type FileEventKindV1 = 'created' | 'saved' | 'moved' | 'deleted';

export type FileEventOriginV1 = 'app' | 'disk';

type FileEventBaseV1 = {
  readonly version: 1;
  readonly path: string;
  readonly origin: FileEventOriginV1;
  /** Unix epoch milliseconds when the bus accepted the event. */
  readonly at: number;
};

export type FileEventV1 =
  | (FileEventBaseV1 & { readonly kind: Exclude<FileEventKindV1, 'moved'> })
  | (FileEventBaseV1 & { readonly kind: 'moved'; readonly from: string });

/**
 * Channel name for a push of `FileEventV1`.
 *
 * `WORKSPACE_CHANNELS.fileEvent` re-exports this. Main-process automation
 * hangs on `WorkspaceSession.onFileEvent`; the renderer hangs on
 * `notoWorkspace.onFileEvent`. Neither evaluates a script.
 */
export const FILE_EVENT_CHANNEL = 'noto:v1:workspace:file-event';

const KINDS: ReadonlySet<string> = new Set(['created', 'saved', 'moved', 'deleted']);
const ORIGINS: ReadonlySet<string> = new Set(['app', 'disk']);

const record = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const exact = (value: Record<string, unknown>, keys: readonly string[]) =>
  Object.keys(value).length === keys.length && Object.keys(value).every((key) => keys.includes(key));

const isPath = (value: unknown): value is string => typeof value === 'string' && value.length > 0;

export function isFileEventV1(value: unknown): value is FileEventV1 {
  if (!record(value) || value.version !== 1) return false;
  if (!KINDS.has(String(value.kind)) || !ORIGINS.has(String(value.origin))) return false;
  if (!isPath(value.path) || !Number.isSafeInteger(value.at) || Number(value.at) < 0) return false;
  if (value.kind === 'moved') {
    return exact(value, ['version', 'kind', 'path', 'from', 'origin', 'at']) && isPath(value.from);
  }
  return exact(value, ['version', 'kind', 'path', 'origin', 'at']);
}
