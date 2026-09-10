/**
 * Tags from a note's YAML frontmatter.
 *
 * The author's file-tags brief was "文件支持 tag，多文件 tag 链接": the tags
 * live in the note as ordinary frontmatter, so a note opened elsewhere still
 * reads, and the same tags link notes to each other. This module only reads;
 * writing a tag is editing the frontmatter, which is already how the vault
 * works.
 *
 * Both shapes the vault uses are accepted:
 *
 *     tags: [agent, cloudflare]
 *     tags:
 *       - agent
 *       - cloudflare
 */

/** The YAML body between the opening and closing `---` of a note, or null. */
export function frontmatterBody(markdown: string): string | null {
  const text = markdown.replace(/^\uFEFF/, '');
  if (!text.startsWith('---')) return null;
  // A first line of only `---` (or `---\r`), then the body, then a line of
  // only `---` again. Matching against the whole file lets a note whose
  // frontmatter is the entire content still parse.
  const match = /^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/.exec(text);
  return match ? match[1]! : null;
}

/**
 * Split a flow-style YAML list (`[a, b, "c d"]`) into its items.
 *
 * Quotes are respected so a comma inside a tag stays part of the tag. Nested
 * lists and maps are refused: a vault tag is a string.
 */
export function splitFlowList(raw: string): string[] {
  const inner = raw.trim();
  if (!inner.startsWith('[') || !inner.endsWith(']')) return [];
  const body = inner.slice(1, -1).trim();
  if (body.length === 0) return [];
  const items: string[] = [];
  let current = '';
  let quote: '"' | "'" | null = null;
  for (let i = 0; i < body.length; i += 1) {
    const ch = body[i]!;
    if (quote) {
      if (ch === '\\' && i + 1 < body.length) {
        current += body[i + 1]!;
        i += 1;
        continue;
      }
      if (ch === quote) {
        quote = null;
        continue;
      }
      current += ch;
      continue;
    }
    if (ch === '"' || ch === "'") {
      quote = ch;
      continue;
    }
    if (ch === ',') {
      const trimmed = current.trim();
      if (trimmed.length > 0) items.push(trimmed);
      current = '';
      continue;
    }
    current += ch;
  }
  const trimmed = current.trim();
  if (trimmed.length > 0) items.push(trimmed);
  return items;
}

/** One bare YAML scalar, with optional matching quotes stripped. */
export function unquoteYamlScalar(raw: string): string {
  const value = raw.trim();
  if (
    (value.startsWith('"') && value.endsWith('"') && value.length >= 2)
    || (value.startsWith("'") && value.endsWith("'") && value.length >= 2)
  ) {
    return value.slice(1, -1);
  }
  // Inline comments after a bare scalar: `agent # note` → `agent`.
  const hash = value.search(/(^|\s)#/);
  return (hash >= 0 ? value.slice(0, hash) : value).trim();
}

/**
 * The tags declared on a `tags:` key inside a frontmatter body.
 *
 * Returns them in file order, de-duplicated case-insensitively so `AI` and
 * `ai` do not both appear as chips. The first spelling wins.
 */
export function parseTagsFromFrontmatter(yaml: string): string[] {
  const lines = yaml.split(/\r?\n/);
  const found: string[] = [];
  const seen = new Set<string>();
  const push = (raw: string): void => {
    const tag = unquoteYamlScalar(raw);
    if (tag.length === 0) return;
    const key = tag.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    found.push(tag);
  };

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i]!;
    const flow = /^\s*tags\s*:\s*(\[[\s\S]*\])\s*(?:#.*)?$/.exec(line);
    if (flow) {
      for (const item of splitFlowList(flow[1]!)) push(item);
      return found;
    }
    const keyOnly = /^\s*tags\s*:\s*(?:#.*)?$/.exec(line);
    if (keyOnly) {
      for (let j = i + 1; j < lines.length; j += 1) {
        const item = lines[j]!;
        const bullet = /^\s*-\s+(.+)$/.exec(item);
        if (!bullet) break;
        push(bullet[1]!);
      }
      return found;
    }
    // A single scalar on the same line is rare but legal YAML.
    const scalar = /^\s*tags\s*:\s+([^[#\s][^#]*?)\s*(?:#.*)?$/.exec(line);
    if (scalar && !scalar[1]!.startsWith('[')) {
      push(scalar[1]!);
      return found;
    }
  }
  return found;
}

/** Tags from a whole note, or `[]` when there is no frontmatter / no tags key. */
export function parseTagsFromMarkdown(markdown: string): string[] {
  const body = frontmatterBody(markdown);
  return body === null ? [] : parseTagsFromFrontmatter(body);
}

/** Compare two tag spellings the way the index does. */
export function tagsEqual(a: string, b: string): boolean {
  return a.toLowerCase() === b.toLowerCase();
}
