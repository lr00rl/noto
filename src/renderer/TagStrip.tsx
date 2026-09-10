/**
 * The tags on the note in front, as chips.
 *
 * The author's file-tags brief wanted tags on a file and a way from one note
 * to the others that share a tag. The chips are that way in: a click opens
 * the browser for the tag. The tags themselves stay in the frontmatter, so a
 * note opened elsewhere still reads and nothing here rewrites the file.
 */

export interface TagStripProps {
  readonly tags: readonly string[];
  readonly onTag: (tag: string) => void;
}

export function TagStrip({ tags, onTag }: TagStripProps) {
  if (tags.length === 0) return null;
  return (
    <div className="tag-strip" data-testid="tag-strip" aria-label="Tags">
      {tags.map((tag) => (
        <button
          key={tag.toLowerCase()}
          type="button"
          className="tag-chip"
          data-testid="tag-chip"
          data-tag={tag}
          title={`Notes tagged ${tag}`}
          onClick={() => onTag(tag)}
        >
          {tag}
        </button>
      ))}
    </div>
  );
}
