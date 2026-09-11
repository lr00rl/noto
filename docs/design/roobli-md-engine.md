# `@roobli/md` — future parse backend

Noto’s markdown v3 stack (`src/shared/markdown/v3/`) still parses with micromark
today. The long-term engine that should own that hot path is the public MIT
package **[@roobli/md](https://github.com/roobli/md)** (“WYSIWYG-first markdown
engine for Noto”).

## Why

Open cost on mid/large files is dominated by one full dialect parse (Linux
`parseDocument` ≈ 570 ms medium / 2.3 s large after wire `nodes` removed the
duplicate renderer pass). The editor already wants block spans, gaps, and
byte-exact untouched regions — a WYSIWYG-oriented engine, not only a correct
mdast dump.

## Bridge

See the engine’s own docs:

- Vision: https://github.com/roobli/md/blob/main/docs/design/vision.md
- Noto bridge: https://github.com/roobli/md/blob/main/docs/design/noto-bridge.md
- Contract v0: https://github.com/roobli/md/blob/main/docs/design/contract-v0.md
- Typora study notes: https://github.com/roobli/md/blob/main/docs/design/typora-notes.md

Integration shape: swap `splitBlocks` / dialect parse for `parseBlocks` from
`@roobli/md` while keeping branded IDs, hashing, serialize, and
`NotoDocumentWire` in Noto.

## Status

Phase 0 scaffold only (micromark behind an explicit replace boundary). Do not
depend on it in product code until parity and bench gates land.
