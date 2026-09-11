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
- Roadmap: https://github.com/roobli/md/blob/main/docs/design/roadmap.md
- Noto bridge: https://github.com/roobli/md/blob/main/docs/design/noto-bridge.md
- Contract v0: https://github.com/roobli/md/blob/main/docs/design/contract-v0.md
- Typora study notes: https://github.com/roobli/md/blob/main/docs/design/typora-notes.md

Integration shape: swap `splitBlocks` / dialect parse for `parseBlocks` from
`@roobli/md` while keeping branded IDs, hashing, serialize, and
`NotoDocumentWire` in Noto.

## Status

**Phase 1 in progress** on `@roobli/md` main: native block scanner for
heading / paragraph / list / fenced code (exact offsets; beats blank-line-naive
fence splits), with micromark fallback for GFM tables, tasks, math,
frontmatter, etc. Phase 0 API scaffold is done.

Do **not** depend on it in Noto product code until parity and bench gates land
(Phase 2+ tables/tasks natively, then bridge adapter).
