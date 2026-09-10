/**
 * The command palette.
 *
 * Command-Shift-P, not Command-K: that chord belongs to the hyperlink, in
 * Typora and here. The previous palette listed only plugin commands behind a
 * Close button, which is why it read as unfinished. This one is the same
 * surface as quick open: a box at the top of the window, a list that grows
 * downward, Enter runs, Escape leaves.
 *
 * It does not animate. A palette opened from the keyboard hundreds of times
 * a day that fades in is a palette that feels late.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  rankCommands, shortcutLabel, type CommandGroup, type PaletteRow,
} from './command-catalog';
import type { WorkspaceMenuCommandV1 } from '../shared/workspace/v1/contracts';

export interface PluginPaletteCommand {
  readonly pluginId: string;
  readonly commandId: string;
  readonly title: string;
  readonly source: string;
}

export interface CommandPaletteProps {
  readonly open: boolean;
  readonly mac: boolean;
  readonly plugins: readonly PluginPaletteCommand[];
  readonly onRun: (command: WorkspaceMenuCommandV1) => void;
  readonly onRunPlugin: (pluginId: string, commandId: string) => void;
  readonly onClose: () => void;
}

const GROUP_ORDER: readonly CommandGroup[] = [
  'Go', 'File', 'Edit', 'Insert', 'Format', 'Paragraph', 'View', 'Plugins',
];

export function CommandPalette({
  open, mac, plugins, onRun, onRunPlugin, onClose,
}: CommandPaletteProps) {
  const [query, setQuery] = useState('');
  const [selected, setSelected] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    setQuery('');
    setSelected(0);
    queueMicrotask(() => inputRef.current?.focus({ preventScroll: true }));
  }, [open]);

  const extra: PaletteRow[] = useMemo(() => plugins.map((command) => ({
    key: `plugin:${command.pluginId}:${command.commandId}`,
    title: command.title,
    group: 'Plugins' as const,
    source: command.source,
    kind: 'plugin' as const,
    pluginId: command.pluginId,
    commandId: command.commandId,
  })), [plugins]);

  const rows = useMemo(() => rankCommands(query, extra), [query, extra]);
  /**
   * Keyboard order is what is on screen, not ranking order regrouped.
   *
   * An empty box is grouped so related commands sit together. A query is a
   * ranked list, because grouping would put the best match under a heading
   * halfway down.
   */
  const visualRows = useMemo(() => {
    if (query.trim().length > 0) return rows;
    return GROUP_ORDER.flatMap((group) => rows.filter((row) => row.group === group));
  }, [query, rows]);

  useEffect(() => {
    setSelected(0);
  }, [query]);

  useEffect(() => {
    const node = listRef.current?.querySelector('[data-selected="true"]');
    if (node instanceof HTMLElement) node.scrollIntoView({ block: 'nearest' });
  }, [selected, visualRows]);

  if (!open) return null;

  const run = (row: PaletteRow | undefined) => {
    if (!row) return;
    if (row.kind === 'plugin' && row.pluginId && row.commandId) {
      onRunPlugin(row.pluginId, row.commandId);
      return;
    }
    if (row.command) onRun(row.command);
  };

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      onClose();
      return;
    }
    if (event.key === 'ArrowDown' || (event.ctrlKey && event.key === 'n')) {
      event.preventDefault();
      setSelected((current) => (visualRows.length === 0 ? 0 : (current + 1) % visualRows.length));
      return;
    }
    if (event.key === 'ArrowUp' || (event.ctrlKey && event.key === 'p')) {
      event.preventDefault();
      setSelected((current) => (visualRows.length === 0 ? 0 : (current - 1 + visualRows.length) % visualRows.length));
      return;
    }
    if (event.key === 'Enter') {
      event.preventDefault();
      run(visualRows[selected]);
    }
  };

  const searching = query.trim().length > 0;
  const grouped = GROUP_ORDER
    .map((group) => ({ group, rows: visualRows.filter((row) => row.group === group) }))
    .filter((entry) => entry.rows.length > 0);

  const renderRow = (row: PaletteRow, index: number) => {
    const active = index === selected;
    return (
      <button
        key={row.key}
        type="button"
        id={`command-row-${index}`}
        role="option"
        aria-selected={active}
        data-selected={active ? 'true' : undefined}
        data-testid="command-row"
        className={active ? 'command-row is-current' : 'command-row'}
        onMouseEnter={() => setSelected(index)}
        onMouseDown={(event) => event.preventDefault()}
        onClick={() => run(row)}
      >
        <span className="command-row-title">{row.title}</span>
        {row.source && <span className="command-row-source">{row.source}</span>}
        {row.keys && (
          <kbd className="command-row-keys">{shortcutLabel(row.keys, mac)}</kbd>
        )}
      </button>
    );
  };

  return (
    <div className="command-scrim" data-testid="command-palette-scrim" onMouseDown={onClose}>
      <section
        className="command-palette"
        role="dialog"
        aria-modal="true"
        aria-label="Commands"
        data-testid="command-palette"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <input
          ref={inputRef}
          type="text"
          className="command-input"
          data-testid="command-input"
          spellCheck={false}
          placeholder="Run a command"
          aria-label="Run a command"
          role="combobox"
          aria-expanded
          aria-controls="command-palette-results"
          aria-activedescendant={visualRows[selected] ? `command-row-${selected}` : undefined}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={onKeyDown}
        />
        <div className="command-results" id="command-palette-results" role="listbox" ref={listRef}>
          {visualRows.length === 0
            ? <p className="command-empty">No command matches that.</p>
            : searching
              ? visualRows.map((row, index) => renderRow(row, index))
              : grouped.map((entry) => (
                <div key={entry.group} className="command-group">
                  <p className="command-group-label">{entry.group}</p>
                  {entry.rows.map((row) => renderRow(row, visualRows.indexOf(row)))}
                </div>
              ))}
        </div>
        <p className="command-hint">Type to search. Enter runs. Esc closes.</p>
      </section>
    </div>
  );
}
