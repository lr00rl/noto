/**
 * Formatting for a selection, without leaving the pointer.
 *
 * Typora keeps these on the keyboard and the Format menu. That is right for
 * someone who already knows the chords. It is silent for someone who has
 * just highlighted a phrase. The author's Typora plugin answered that with a
 * small menu above the selection; this is the same idea, for the marks a
 * note actually uses.
 *
 * It waits a beat after the selection settles so a drag does not grow a
 * toolbar under the pointer. Buttons swallow mousedown so the selection
 * they are about to format is still there when the click lands. A mark
 * that already applies is drawn as on, so the HUD is a state as well as a
 * set of actions. It sits above the first line of the selection and drops
 * below when the window has no room above.
 */

import { useLayoutEffect, useRef, useState } from 'react';
import type { WorkspaceMenuCommandV1 } from '../shared/workspace/v1/contracts';
import { placeOverlay } from './place-overlay';

export interface FormatHudProps {
  readonly left: number;
  readonly top: number;
  readonly bottom: number;
  readonly active: readonly string[];
  readonly onRun: (command: WorkspaceMenuCommandV1) => void;
}

interface HudAction {
  readonly id: WorkspaceMenuCommandV1;
  readonly mark: string;
  readonly label: string;
  /** A rule before this button, so link and clear are not the last mark. */
  readonly rule?: boolean;
}

const ACTIONS: readonly HudAction[] = [
  { id: 'mark-strong', mark: 'B', label: 'Bold' },
  { id: 'mark-emphasis', mark: 'I', label: 'Italic' },
  { id: 'mark-strike', mark: 'S', label: 'Strikethrough' },
  { id: 'mark-code', mark: '</>', label: 'Inline code' },
  { id: 'mark-highlight', mark: '==', label: 'Highlight' },
  { id: 'insert-link', mark: '[]', label: 'Link', rule: true },
  { id: 'clear-format', mark: 'Aa', label: 'Clear formatting', rule: true },
];

export function FormatHud({ left, top, bottom, active, onRun }: FormatHudProps) {
  const hudRef = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState({ left: left - 80, top: Math.max(8, top - 40) });
  const on = new Set(active);

  useLayoutEffect(() => {
    const node = hudRef.current;
    if (!node) return;
    setPos(placeOverlay(
      { left, top, bottom },
      { width: node.offsetWidth, height: node.offsetHeight },
      'above',
      { width: window.innerWidth, height: window.innerHeight },
      8,
      'center',
    ));
  }, [left, top, bottom, active]);

  return (
    <div
      ref={hudRef}
      className="format-hud"
      role="toolbar"
      aria-label="Format selection"
      data-testid="format-hud"
      style={{ left: pos.left, top: pos.top }}
    >
      {ACTIONS.map((action) => {
        const pressed = on.has(action.id);
        return (
          <button
            key={action.id}
            type="button"
            className={[
              'format-hud-button',
              pressed ? 'is-on' : '',
              action.rule ? 'is-rule' : '',
            ].filter(Boolean).join(' ')}
            data-testid={`format-hud-${action.id}`}
            title={action.label}
            aria-label={action.label}
            aria-pressed={pressed}
            onMouseDown={(event) => event.preventDefault()}
            onClick={() => onRun(action.id)}
          >
            <span aria-hidden="true">{action.mark}</span>
          </button>
        );
      })}
    </div>
  );
}
