import { describe, expect, it } from 'vitest';
import {
  CATALOG, rankCommands, rankSlash, shortcutLabel,
} from '../../src/renderer/command-catalog';

describe('the command palette', () => {
  it('shows the pinned commands when nothing has been typed', () => {
    const rows = rankCommands('');
    expect(rows.map((row) => row.title)).toContain('Quick Open');
    expect(rows.map((row) => row.title)).toContain('New note');
    expect(rows.length).toBeLessThan(20);
  });

  it('shows every plugin command when nothing has been typed', () => {
    const plugin = {
      key: 'plugin:title-shift:up',
      title: 'Shift headings up',
      group: 'Plugins' as const,
      source: 'Title Shift',
      kind: 'plugin' as const,
      pluginId: 'title-shift',
      commandId: 'up',
    };
    const rows = rankCommands('', [plugin]);
    expect(rows.some((row) => row.title === 'Shift headings up')).toBe(true);
    expect(rows.filter((row) => row.kind === 'plugin')).toHaveLength(1);
  });

  it('lists export-docx in the catalog', () => {
    expect(CATALOG.some((entry) => entry.id === 'export-docx')).toBe(true);
  });

  it('finds a heading by the letters a person would type', () => {
    const rows = rankCommands('head 1');
    expect(rows[0]?.title).toBe('Heading 1');
  });

  it('finds a plugin command the same way', () => {
    const rows = rankCommands('shift', [{
      key: 'plugin:title-shift:up',
      title: 'Shift headings up',
      group: 'Plugins',
      source: 'Title Shift',
      kind: 'plugin',
      pluginId: 'title-shift',
      commandId: 'up',
    }]);
    expect(rows.some((row) => row.title === 'Shift headings up')).toBe(true);
  });
});

describe('the slash menu', () => {
  it('puts Heading 1 first for h1', () => {
    expect(rankSlash('h1')[0]?.id).toBe('block-heading-1');
  });

  it('puts a table first for table', () => {
    expect(rankSlash('table')[0]?.id).toBe('table-insert');
  });

  it('puts a table first for ta, not a task list', () => {
    expect(rankSlash('ta')[0]?.id).toBe('table-insert');
  });

  it('puts a task list first for todo', () => {
    expect(rankSlash('todo')[0]?.id).toBe('block-task-list');
  });

  it('puts Heading 1 first for the Chinese alias 标题', () => {
    expect(rankSlash('标题')[0]?.id).toBe('block-heading-1');
  });

  it('puts a table first for the Chinese alias 表格', () => {
    expect(rankSlash('表格')[0]?.id).toBe('table-insert');
  });

  it('puts a task list first for the Chinese alias 待办', () => {
    expect(rankSlash('待办')[0]?.id).toBe('block-task-list');
  });
});

describe('shortcut labels', () => {
  it('writes a Mac chord the way the menus do', () => {
    expect(shortcutLabel('CmdOrCtrl+Shift+P', true)).toBe('⇧⌘P');
    expect(shortcutLabel('CmdOrCtrl+Alt+T', true)).toBe('⌥⌘T');
  });

  it('writes a Control chord everywhere else', () => {
    expect(shortcutLabel('CmdOrCtrl+Shift+P', false)).toBe('Ctrl+Shift+P');
  });
});
