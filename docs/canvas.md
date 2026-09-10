# Canvases

A canvas is a React document: one `.canvas.tsx` file that default-exports a
component and draws with `@roobli/canvas`. It is for findings that want
tables, stats and a short argument, the way a Cursor canvas does, not for
a Markdown note.

It is not a plugin. A plugin is a manifest, a lease and a capability
boundary. Opening a file type is editor behaviour. The kit lives in its
own repository so something that is not Noto can import it; the package is
MIT, Noto is AGPL-3.0.

https://github.com/roobli/canvas

Noto does not compile a vault `.canvas.tsx` yet. Doing that is running the
author's code inside the app. The unused experimental plugin origin
(`src/main/protocol/register-experimental-plugin-protocol.ts`) is the
place that belongs, when it opens. Until then, do not add `.canvas.tsx` to
the editable extensions, and do not `eval` a note in the editor renderer.

The document column is already called a canvas in the chrome (`#document-canvas`).
That name stays. The file type, when it arrives, is `.canvas.tsx`.
