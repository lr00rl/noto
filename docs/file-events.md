# File events

A file in the open folder was created, saved, moved, or deleted. Later
automation hangs here. The events are a built-in bus in the main process, not
user scripts: nothing is evaluated, and a listener cannot reach past what the
workspace already does with files.

## Who to subscribe as

In main, on the open `WorkspaceSession`:

```ts
const stop = session.onFileEvent((event) => {
  // event.kind is created | saved | moved | deleted
  // event.origin is app | disk
  // event.path is absolute
  // event.from is set only on moved
});
session.recentFileEvents(); // last 64 accepted events, for a late attach
```

In the renderer:

```ts
const stop = window.notoWorkspace.onFileEvent((event) => {
  // same FileEventV1 shape
});
```

`WORKSPACE_CHANNELS.fileEvent` is `noto:v1:workspace:file-event`. The preload
guards with `isFileEventV1` before the listener runs.

## What an event is

`origin` says who did it. An atomic save this process just performed also shows
up as a disk event a moment later. The bus mutes disk echoes on a path for two
seconds after an app event on that path, and a `moved` copies the mute onto
`from` and onto every note the file index currently lists under either path, so
a folder rename is not a burst of child deletes and creates. A subscriber that
treats an app save and its disk echo as two changes will still be wrong if the
mute misses (a symlink, or `/tmp` versus `/private/tmp` on macOS); prefer
`origin === 'app'` for work this process caused, and `origin === 'disk'` for
work someone else caused.

`from` is only on `moved`. Node's folder watcher does not give both paths, so a
Finder rename is not synthesized into a move: it arrives as `deleted` then
`created`, both `origin: 'disk'`. App code that renamed the file is the one
that can fill `from`.

Directories themselves are not events. Only extensions the file index already
treats as openable are reported. `.git`, `node_modules` and `.noto` are
ignored.

## What this is not

It is not a task runner. It does not read a hooks file from the vault. It does
not `eval`. A plugin does not activate on these events today; `editor.ready` is
still the only activation event. The remote control has no `/v1/file-events`
route: hang in process, not over HTTP.

Copy-as-save (`save a copy`) does not emit `created`. The tree is unchanged
from the session's point of view until something in the open folder is
rewritten under a new name through rename, duplicate, or new note.

## Where the events are born

App events come from the workspace session: save (after file-truth accepts
the write), new note, duplicate, new folder, rename, drag-move, trash. Disk
events come from a recursive `fs.watch` on the open folder, debounced 300 ms
after the last burst with a 2 s ceiling so a continuous writer cannot push
the report back forever. Classification is existence against a set of paths
already seen: gone is deleted, new is created, still there is saved.
