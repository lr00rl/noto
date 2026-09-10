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

Surfaces checked on current `main` (post #14 related-notes polish):
`wiki-link-plugin`, `wiki-target` (+ `wikiTargetFor`), `QuickOpen` Alt+Enter,
`RailLinks` + `seed-links` + `note-graph`, `index-block` (index vs related
families) + HTML comment quieting in `html-view`.

| Habit (RooB / typora-plugin-lite) | Noto today | Gap |
|---|---|---|
| Follow `[[target]]` / `[[target\|label]]` without rewriting bytes | Decorations in `wiki-link-plugin.ts`; Cmd/Ctrl-click | **OK** — right risk model |
| Resolve note-relative then root then name | `wikiCandidates` in `wiki-target.ts` | **OK** for MOC `../A000/...` links |
| Related panel from `.note-assistant/graph.json` | `note-graph.ts` + `RailLinks`; **#14** `seed-links` fills **Links to** from in-note wiki (preferring an index region) when `graph.notes` misses a MOC | **Partial — closed for hub “Links to”**; still no graph-backed Linked from / Related on `moc: true` hubs (pipeline still omits them) |
| Issued in-note `<!-- note-assistant:start -->` Related Notes blocks | **#14** `index-block.ts` keeps `index:*` vs bare `note-assistant:start/end` as distinct families; related draws `.noto-related` chrome | **Closed** (chrome distinct); vault still has **0** issued related blocks, so unexercised in RooB |
| Directory / MOC indexes as compact read-only UI | `index-block` widget + click `onFollow`; caret restores source | **Mostly OK** — 100+ link MOCs are the intended shape. #17 stub scroller helps **large documents** when far-off blocks leave the viewport; it does **not** virtualize *inside* one index widget |
| Quiet HTML comment markers | `isHtmlComment` in `html-view.ts` | **OK** |
| Quick open → insert wiki link (Alt+Enter / `[[` trigger) | **#14** inserts note-relative `wikiTargetFor` targets with `\|title` when the path is not the bare name (wiki-trigger / QuickOpen) | **Closed** |
| Rebuild graph from editor | Typora plugin rebuild shortcut | **Not in Noto** (out of scope unless productized) |
| Tags line inside related blocks | Related chrome can show tags when present; Rail graph titles otherwise | N/A while sparse blocks = 0 |

#14 (`feat/related-notes-polish`) closed the three §5 items that were product
gaps on this pass (MOC Links seed, distinct related chrome, Alt+Enter
note-relative). Remaining honesty: MOC hubs still lack graph-backed Related /
backlinks; large single index widgets are still one decoration.

## 4. Three scenarios — pass / fail / risk

### A. Related panel

- **Pass (narrow):** Opening an eligible content note that exists in
  `graph.notes` (e.g. the TCP-tuning sample) can populate Linked from / Links
  to / Related from the same `graph.json` Typora’s plugin reads.
- **Pass (hub Links to, #14):** Preferred MOCs absent from `graph.notes` now
  seed RailLinks **Links to** from in-note wiki targets (`seed-links`), so a
  map note is no longer a silent “graph has not met this note” dead end for
  outbound links.
- **Still open:** Graph-backed Linked from / Related on `moc: true` hubs —
  pipeline still omits those rows; Noto does not invent backlinks or related
  scores from the index alone.
- **Risk:** Product ambiguity remains if Dylan expects full Related parity on
  hubs without teaching note-assistant to emit lightweight MOC graph rows.

### B. Read-only index

- **Pass:** Index markers are recognized (old + new families); region is
  decoration/widget only; file bytes stay untouched while caret is outside;
  119-link MOC is structurally what the widget was built for; HTML comment
  markers stay quiet.
- **Risk:** Performance/UX on very large **single** index widgets still not
  measured in-app; #17 stub scroller does not slice inside one widget.
  Related vs index chrome is no longer conflated (#14). False-positive
  `[[...]]` in code-heavy notes (vllm sample) still decorates as a wiki link.

### C. Byte-for-byte unchanged

- **Pass (display path):** Wiki links and index UI are decorations/widgets;
  they cannot rewrite saved bytes by themselves — matches the wiki-plugin
  design comment and the index-block “pipeline never sees a difference” claim.
- **Risk (edit path):** Untouched blocks stay byte-exact; **edited** blocks
  still go through the serializer dialect (documented in `typora-gap.md`).
  Stress here is “open hub → click around → save without editing” should be
  clean; “touch one list item inside an index region” re-enters source and
  can diverge if the author saves after an accidental edit.

## 5. Recommendations

1. **MOC / Links UX:** **Done in #14** for outbound seed (`seed-links`).
   Optional follow-up: teach note-assistant to emit lightweight MOC rows into
   `graph.json` without `shouldGenerateBlock` if Linked from / Related on hubs
   matter.
2. **Keep marker families distinct:** **Done in #14** (`index` vs `related`
   chrome in `index-block.ts`).
3. **QuickOpen Alt+Enter parity:** **Done in #14** (`wikiTargetFor` + `\|title`).
4. **Optional:** Ignore `[[digits, digits]]` / obvious non-path targets in
   the wiki decoration scanner to reduce noise in RTFS notes.
5. **Do not** commit RooB `.note-assistant/graph.json`; rebuild locally for
   tests as done here.

## 6. What was not done

- No deny-dir reads; no `vault.mjs lint` full-vault write.
- No RooB commit; no apply-graph / AI enrich.
- No Noto product code change in this branch (recommendations only;
  overlapping note-assistant work already on `main`).
