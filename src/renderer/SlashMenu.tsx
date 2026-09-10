/**
 * The slash menu: what `/` at the start of a block becomes.
 *
 * The editor keeps the focus so typing continues to filter. Arrows, Enter
 * and Escape are taken here, with capture, because they would otherwise move
 * the caret or split the block. A press on a row uses mousedown-prevented
 * click so the editor does not lose the selection the command is about to
 * replace. The menu sits under the caret and flips above it when the window
 * runs out of room, measured after layout rather than guessed.
 */

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { rankSlash, type SlashItem } from './command-catalog';
import { placeOverlay } from './place-overlay';

export interface SlashMenuProps {
  readonly query: string;
  readonly left: number;
  readonly top: number;
  readonly bottom: number;
  readonly onPick: (item: SlashItem) => void;
  readonly onDismiss: (removeToken: boolean) => void;
}

export function SlashMenu({
  query, left, top, bottom, onPick, onDismiss,
}: SlashMenuProps) {
  const items = useMemo(() => rankSlash(query), [query]);
  const [selected, setSelected] = useState(0);
  const menuRef = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState({ left, top: bottom + 6 });

  useEffect(() => { setSelected(0); }, [query]);

  useLayoutEffect(() => {
    const node = menuRef.current;
    if (!node) return;
    setPos(placeOverlay(
      { left, top, bottom },
      { width: node.offsetWidth, height: node.offsetHeight },
      'below',
      { width: window.innerWidth, height: window.innerHeight },
    ));
  }, [left, top, bottom, items.length]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        event.stopPropagation();
        onDismiss(true);
        return;
      }
      if (event.key === 'ArrowDown' || (event.ctrlKey && event.key === 'n')) {
        event.preventDefault();
        event.stopPropagation();
        setSelected((current) => (items.length === 0 ? 0 : (current + 1) % items.length));
        return;
      }
      if (event.key === 'ArrowUp' || (event.ctrlKey && event.key === 'p')) {
        event.preventDefault();
        event.stopPropagation();
        setSelected((current) => (items.length === 0 ? 0 : (current - 1 + items.length) % items.length));
        return;
      }
      if (event.key === 'Enter' || event.key === 'Tab') {
        event.preventDefault();
        event.stopPropagation();
        const item = items[selected];
        if (item) onPick(item);
      }
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [items, selected, onDismiss, onPick]);

  if (items.length === 0) return null;

  return (
    <div
      ref={menuRef}
      className="slash-menu"
      role="listbox"
      aria-label="Insert"
      data-testid="slash-menu"
      style={{ left: pos.left, top: pos.top }}
    >
      {items.map((item, index) => (
        <button
          key={item.id}
          type="button"
          role="option"
          aria-selected={index === selected}
          data-selected={index === selected ? 'true' : undefined}
          data-testid="slash-row"
          className={index === selected ? 'slash-row is-current' : 'slash-row'}
          onMouseEnter={() => setSelected(index)}
          onMouseDown={(event) => event.preventDefault()}
          onClick={() => onPick(item)}
        >
          <span className="slash-row-title">{item.title}</span>
          {item.hint.length > 0 && <span className="slash-row-hint">{item.hint}</span>}
        </button>
      ))}
    </div>
  );
}
