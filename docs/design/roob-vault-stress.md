# RooB vault stress checks against Noto assumptions

Read-mostly pass against shared `/workspace/RooB` (≈7185 markdown files).
Deny dirs were not opened. Preferred probes only: `Z900_MOCs/*` hubs,
`A000_Theoretical_Knowledge/A404_Linux/*` samples, `A300_AI/.../vllm_RTFS.md`,
`P000_Public/*`, plus `vault.yaml` / `.tools` READMEs.

Gold standard named for this pass: **related panel** / **read-only index** /
**byte-for-byte unchanged**.

## 1. Graph build

Command (from vault root):

```bash
node .tools/note-assistant/build-graph.mjs --root /workspace/RooB
```

| | |
|---|---|
| Result | Wrote `.note-assistant/graph.json` + `report.md` (gitignored; not committed) |
| Size | ≈17 MB (`17136883` bytes) |
| Wall time | ≈33 s on this box (Node 22.13 after nvm; first run also paid shell/nvm setup) |
| Scanned | 7183 notes |
| Target notes | 4898 |
| Graph notes | 4664 (tech-eligible only) |
| Sparse related blocks | 0 (`heuristicBlocksEnabled: false`, no AI cache) |
| Schema | `schemaVersion: 2` |

`Z900_MOCs` is in `vault.yaml` targets with `graph: true`, but notes with
`moc: true` frontmatter are dropped from `graph.notes` by
`isTechEligiblePath(...) && !moc:true` in note-assistant `lib.mjs`. Hub MOCs
therefore never appear as graph rows even when the directory is targeted.

## 2. Wiki-link density and resolve rate

Full-vault `vault.mjs lint` was **not** run (it walks deny dirs). Resolve rate
was measured on preferred files only, using the same order Noto/`vault.mjs`
document: note-relative → root-relative → unique basename via graph paths.

| Path | Wiki-links | Resolve | Notes |
|---|---:|---:|---|
| `Z900_MOCs/代理与隧道.md` | **119** | **100%** (all note-relative) | Entirely inside `<!-- note-assistant:index:* -->` |
| `Z900_MOCs/LLM推理部署.md` | 51 | 100% note-relative | Same; index-shaped MOC |
| `A000_Theoretical_Knowledge/A404_Linux/00_Linux总览.md` | 102 | 100% note-relative | `moc: true` + index markers |
| `.../跨境高RTT链路的TCP调优.md` | 4 | 100% | In graph: 3 explicit / 2 back / 2 related |
| `.../vllm_RTFS/vllm_RTFS.md` | 1 “link” | 0% | False positive: `[[100, 101, ...]]` array text, not a note |
| `P000_Public/首页.md` | 0 | — | In graph (backlink only) |
| `P000_Public/P001_DailyNews/00_索引.md` | 1 | 0% | Placeholder target `路径`, not a real note |

Allow-dir marker census (paths only): ≈176 files with
`note-assistant:index:start` under A/B/D/P/V/Z; **0** files with issued
`<!-- note-assistant:start -->` related blocks right now (matches graph
`notesWithBlocks: 0`). ≈159 `moc: true` notes under the allow tops sampled.

## 3. Noto vs tpl note-assistant / RooB habits — concrete gaps

Surfaces checked on current `main`: `wiki-link-plugin`, `wiki-target`,
`QuickOpen` Alt+Enter, `RailLinks` + `note-graph` graph consumer,
`index-block` + HTML comment quieting in `html-view`.

| Habit (RooB / typora-plugin-lite) | Noto today | Gap |
|---|---|---|
| Follow `[[target]]` / `[[target\|label]]` without rewriting bytes | Decoration in `wiki-link-plugin.ts`; Cmd/Ctrl-click | **OK** — right risk model |
| Resolve note-relative then root then name | `wikiCandidates` in `wiki-target.ts` | **OK** for MOC `../A000/...` links |
| Related panel from `.note-assistant/graph.json` | `note-graph.ts` + `RailLinks` (backlinks / links / related) | **Partial** — works for eligible notes; **silent miss on every `moc: true` hub** (“graph has not met this note”) |
| Issued in-note `<!-- note-assistant:start -->` Related Notes blocks | `index-block.ts` treats **both** `index:*` and bare `note-assistant:*` marker families as one “index” widget | **Latent** — no related blocks in vault now; if apply-graph returns, Related Notes would render as an index list, not a distinct related panel |
| Directory / MOC indexes as compact read-only UI | `index-block` widget + click `onFollow`; caret restores source | **Mostly OK** — stress MOCs are exactly this shape (100+ links) |
| Quiet HTML comment markers | `isHtmlComment` in `html-view.ts` | **OK** |
| Quick open → insert wiki link (Alt+Enter / `[[` trigger) | Inserts `[[basename]]` or vault-root `relativePath` when ambiguous; rarely `\|title`; **not** note-relative `wikiTargetFor` (`../…`) | **Gap** vs MOC/apply-graph link style |
| Rebuild graph from editor | Typora plugin rebuild shortcut | **Not in Noto** (out of scope unless productized) |
| Tags line inside related blocks | Rail uses graph titles only | N/A while sparse blocks = 0 |

`feat/note-assistant` on this clone points at the same tip as `main`; Links /
index / wiki work already landed via earlier merges. No duplicate
implementation PR from this pass.

## 4. Three scenarios — pass / fail / risk

### A. Related panel

- **Pass (narrow):** Opening an eligible content note that exists in
  `graph.notes` (e.g. the TCP-tuning sample) can populate Linked from / Links
  to / Related from the same `graph.json` Typora’s plugin reads.
- **Fail (hub stress):** Preferred MOCs (`Z900_MOCs/代理与隧道.md`,
  `LLM推理部署.md`, `00_Linux总览.md`) are `moc: true` → absent from
  `graph.notes` → RailLinks reports unknown note. That is the note type Dylan
  opens as a map.
- **Risk:** Product ambiguity — vault pipeline *intentionally* skips MOCs for
  block issuance; Noto still offers a Links rail with no fallback (e.g. derive
  neighbours from the index block’s own wiki-links, or index MOCs into the
  graph without emitting blocks).

### B. Read-only index

- **Pass:** Index markers are recognized (old + new families); region is
  decoration/widget only; file bytes stay untouched while caret is outside;
  119-link MOC is structurally what the widget was built for; HTML comment
  markers stay quiet.
- **Risk:** Performance/UX on very large index widgets not measured in-app
  here. Conflating future related-blocks with index rendering (see §3).
  False-positive `[[...]]` in code-heavy notes (vllm sample) still decorates
  as a wiki link.

### C. Byte-for-byte unchanged

- **Pass (display path):** Wiki links and index UI are decorations/widgets;
  they cannot rewrite saved bytes by themselves — matches the wiki-plugin
  design comment and the index-block “pipeline never sees a difference” claim.
- **Risk (edit path):** Untouched blocks stay byte-exact; **edited** blocks
  still go through the serializer dialect (documented in `typora-gap.md`).
  Stress here is “open hub → click around → save without editing” should be
  clean; “touch one list item inside an index region” re-enters source and
  can diverge if the author saves after an accidental edit.

## 5. Recommendations (no code in this PR)

1. **MOC / Links UX:** When `graph.notes` misses the current path but the
   note has an index region, seed RailLinks “Links to” from parsed index
   wiki targets (read-only), or teach note-assistant to emit lightweight
   MOC rows into `graph.json` without `shouldGenerateBlock`.
2. **Keep marker families distinct:** Render `note-assistant:index:*` as
   index UI; reserve `note-assistant:start/end` for a related-notes chrome
   (or ignore until apply-graph is used again).
3. **QuickOpen Alt+Enter parity:** Prefer note-relative targets
   (`wikiTargetFor` / `relPathFromDir`) and optional `|title`, matching
   apply-graph and hand-written MOC links — basename-only breaks once a
   second `00_索引` exists outside the current folder’s uniqueness rule
   (Noto already special-cases ambiguous basenames to root-relative; still
   not note-relative).
4. **Optional:** Ignore `[[digits, digits]]` / obvious non-path targets in
   the wiki decoration scanner to reduce noise in RTFS notes.
5. **Do not** commit RooB `.note-assistant/graph.json`; rebuild locally for
   tests as done here.

## 6. What was not done

- No deny-dir reads; no `vault.mjs lint` full-vault write.
- No RooB commit; no apply-graph / AI enrich.
- No Noto product code change in this branch (recommendations only;
  overlapping note-assistant work already on `main`).
