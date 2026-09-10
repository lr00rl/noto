/**
 * Explicit links written in a note: `[[wiki]]` and markdown `[text](path.md)`.
 *
 * Related-note ranking belongs to note-assistant and stays out of here. This
 * is only what the files themselves say, so the Links rail works in a vault
 * that has never run that plugin.
 */

/** `[[target]]` or `[[target|label]]`, no newline, no nested `[`. */
const WIKI_LINK = /\[\[([^[\]\n|]+)(?:\|([^[\]\n]*))?\]\]/g;
/** Markdown links, including the image form so it can be skipped. */
const MD_LINK = /(!?)\[([^\]]*)\]\((<[^>\n]+>|[^)\s]+)(?:\s+"[^"]*")?\)/g;

const REMOTE = /^(?:[a-z][a-z0-9+.-]*:|\/\/|#)/i;

export interface ExtractedLink {
  /** The target as written, heading stripped, for resolution. */
  readonly target: string;
}

/**
 * Fences and inline code are not places a person follows a link from, and
 * counting them as neighbours fills the rail with accidents.
 */
function stripProtected(text: string): string {
  return text
    .replace(/^```[^\n]*\n[\s\S]*?^```/gm, '')
    .replace(/^~~~[^\n]*\n[\s\S]*?^~~~/gm, '')
    .replace(/`[^`\n]+`/g, '');
}

function unwrapHref(raw: string): string {
  const trimmed = raw.trim();
  if (trimmed.startsWith('<') && trimmed.endsWith('>')) return trimmed.slice(1, -1).trim();
  return trimmed;
}

/** Every resolvable local target this note names, in file order, unique. */
export function extractLinkTargets(markdown: string): readonly ExtractedLink[] {
  const source = stripProtected(markdown);
  const found: ExtractedLink[] = [];
  const seen = new Set<string>();
  const take = (target: string) => {
    const wanted = target.split('#')[0].trim();
    if (wanted.length === 0 || REMOTE.test(wanted)) return;
    const key = wanted.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    found.push({ target: wanted });
  };

  WIKI_LINK.lastIndex = 0;
  for (;;) {
    const match = WIKI_LINK.exec(source);
    if (match === null) break;
    take(match[1]);
  }

  MD_LINK.lastIndex = 0;
  for (;;) {
    const match = MD_LINK.exec(source);
    if (match === null) break;
    if (match[1] === '!') continue;
    take(unwrapHref(match[3]));
  }

  return found;
}

/** The first ATX heading, or the file's name without its extension. */
export function titleFromNote(markdown: string, name: string): string {
  const heading = /^#\s+(.+)$/m.exec(markdown);
  if (heading) return heading[1].trim();
  return name.replace(/\.(?:md|markdown)$/i, '');
}
