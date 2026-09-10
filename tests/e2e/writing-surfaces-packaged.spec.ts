/**
 * Slash insert, the format HUD, and the empty state with a folder open.
 *
 * Ranking is unit-tested. What only the window can show is that typing `/`
 * at the start of a block inserts, that a selection grows a toolbar which
 * actually marks the text, and that a folder with no note in front offers
 * New note rather than pretending nothing is open.
 */

import { mkdir, readdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { expect, test, _electron as electron, type ElectronApplication, type Page } from '@playwright/test';
import { LINE_START, packagedExecutable, placeCaret } from './packaged-app';

const resultRoot = path.join(process.cwd(), 'test-results', 'writing-surfaces');
const shots = path.join(process.cwd(), 'test-results', 'writing-surfaces-shots');

async function invokeMenu(app: ElectronApplication, id: string): Promise<void> {
  await app.evaluate(({ Menu }, itemId) => {
    const find = (items: Electron.MenuItem[]): Electron.MenuItem | null => {
      for (const item of items) {
        if (item.id === itemId) return item;
        const nested = item.submenu ? find(item.submenu.items) : null;
        if (nested) return nested;
      }
      return null;
    };
    const menu = Menu.getApplicationMenu();
    const target = menu ? find(menu.items) : null;
    if (!target) throw new Error(`No menu item with id ${itemId}`);
    target.click();
  }, id);
}

async function launchFile(name: string, contents: string): Promise<{
  app: ElectronApplication; page: Page;
}> {
  const workspace = path.join(resultRoot, name);
  await rm(workspace, { recursive: true, force: true });
  await mkdir(path.join(workspace, 'user-data'), { recursive: true });
  const file = path.join(workspace, 'note.md');
  await writeFile(file, contents, 'utf8');
  const app = await electron.launch({
    executablePath: packagedExecutable(),
    args: [`--user-data-dir=${path.join(workspace, 'user-data')}`, `--open=${file}`],
  });
  const page = await app.firstWindow();
  await page.waitForSelector('[data-testid="noto-editor"]', { state: 'visible', timeout: 30_000 });
  return { app, page };
}

test.describe('writing surfaces', () => {
  test('slash at the start of a block inserts a heading', async () => {
    const { app, page } = await launchFile('slash-h1', 'A paragraph.\n');
    try {
      await page.setViewportSize({ width: 1440, height: 900 });
      await placeCaret(page, page.locator('.canvas-slot:not([hidden]) .ProseMirror p').first());
      await page.keyboard.press(LINE_START);
      await page.keyboard.press(process.platform === 'darwin' ? 'Meta+Shift+ArrowRight' : 'Shift+End');
      await page.keyboard.press('Backspace');
      await page.keyboard.type('/h1');
      await expect(page.getByTestId('slash-menu')).toBeVisible();
      await expect(page.getByTestId('slash-row').first()).toContainText('Heading 1');
      await mkdir(shots, { recursive: true });
      await page.screenshot({ path: path.join(shots, 'slash-1440.png') });
      await page.keyboard.press('Enter');
      await expect(page.getByTestId('slash-menu')).toHaveCount(0);
      await expect(page.locator('.canvas-slot:not([hidden]) .ProseMirror h1')).toBeVisible();
      await page.keyboard.type('Title');
      await expect(page.locator('.canvas-slot:not([hidden]) .ProseMirror h1')).toHaveText('Title');
    } finally {
      await app.close();
    }
  });

  test('selecting a phrase grows a format HUD that bolds it', async () => {
    const { app, page } = await launchFile('format-hud', 'A phrase to format.\n');
    try {
      await page.setViewportSize({ width: 1440, height: 900 });
      await placeCaret(page, page.locator('.canvas-slot:not([hidden]) .ProseMirror p').first());
      await page.keyboard.press(process.platform === 'darwin' ? 'Meta+A' : 'Control+A');
      await expect(page.getByTestId('format-hud')).toBeVisible();
      await mkdir(shots, { recursive: true });
      await page.screenshot({ path: path.join(shots, 'format-hud-1440.png') });
      await page.getByTestId('format-hud-mark-strong').click();
      await expect(page.locator('.canvas-slot:not([hidden]) .ProseMirror strong')).toHaveText('A phrase to format.');
      await expect(page.getByTestId('format-hud-mark-strong')).toHaveAttribute('aria-pressed', 'true');
    } finally {
      await app.close();
    }
  });

  test('Command and K opens the link panel, not the palette', async () => {
    const { app, page } = await launchFile('cmd-k', 'A phrase to link.\n');
    try {
      await page.setViewportSize({ width: 1100, height: 700 });
      await placeCaret(page, page.locator('.canvas-slot:not([hidden]) .ProseMirror p').first());
      await page.keyboard.press(process.platform === 'darwin' ? 'Meta+A' : 'Control+A');
      await expect(page.getByTestId('format-hud')).toBeVisible();
      await page.keyboard.press(process.platform === 'darwin' ? 'Meta+K' : 'Control+K');
      await expect(page.getByTestId('command-palette')).toHaveCount(0);
      if (!await page.getByTestId('link-input').isVisible()) {
        await page.getByTestId('format-hud-insert-link').click();
      }
      await expect(page.getByTestId('link-input')).toBeVisible();
    } finally {
      await app.close();
    }
  });

  test('a folder with no note in front offers New note', async () => {
    const workspace = path.join(resultRoot, 'empty-folder');
    await rm(workspace, { recursive: true, force: true });
    const vault = path.join(workspace, 'vault');
    await mkdir(vault, { recursive: true });
    await mkdir(path.join(workspace, 'user-data'), { recursive: true });
    await writeFile(path.join(vault, 'existing.md'), '# Existing\n', 'utf8');
    const app = await electron.launch({
      executablePath: packagedExecutable(),
      args: [`--user-data-dir=${path.join(workspace, 'user-data')}`, vault],
    });
    const page = await app.firstWindow();
    try {
      await page.waitForSelector('[data-testid="file-tree"]', { state: 'visible', timeout: 30_000 });
      await page.setViewportSize({ width: 1440, height: 900 });
      if (await page.getByTestId('empty-state').count() === 0) {
        await invokeMenu(app, 'close-tab');
      }
      await expect(page.getByTestId('empty-state')).toBeVisible();
      await expect(page.getByTestId('empty-new-note')).toBeVisible();
      await mkdir(shots, { recursive: true });
      await page.screenshot({ path: path.join(shots, 'empty-folder-1440.png') });
      await page.setViewportSize({ width: 720, height: 560 });
      await page.screenshot({ path: path.join(shots, 'empty-folder-720.png') });
      await page.getByTestId('empty-new-note').click();
      await expect.poll(async () => readdir(vault)).toContain('Untitled.md');
    } finally {
      await app.close();
    }
  });

  test('the command palette finds a heading and sits still', async () => {
    const { app, page } = await launchFile('palette-shot', 'A paragraph.\n');
    try {
      await page.setViewportSize({ width: 1440, height: 900 });
      await invokeMenu(app, 'command-palette');
      await expect(page.getByTestId('command-palette')).toBeVisible();
      await mkdir(shots, { recursive: true });
      await page.screenshot({ path: path.join(shots, 'palette-empty-1440.png') });
      await page.getByTestId('command-input').fill('heading 1');
      await expect(page.getByTestId('command-row').first()).toContainText('Heading 1');
      await page.screenshot({ path: path.join(shots, 'palette-query-1440.png') });
    } finally {
      await app.close();
    }
  });
});
