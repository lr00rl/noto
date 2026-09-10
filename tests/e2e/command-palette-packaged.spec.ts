/**
 * The command palette, against the packaged app.
 *
 * Ranking is unit-tested. What only the window can show is that the menu
 * opens the same surface, that typing finds a heading command, and that
 * Enter actually turns the paragraph into one.
 */

import { mkdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { expect, test, _electron as electron, type ElectronApplication, type Page } from '@playwright/test';
import { packagedExecutable, placeCaret } from './packaged-app';

const resultRoot = path.join(process.cwd(), 'test-results', 'command-palette');

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

test.describe('command palette', () => {
  test('finds Heading 1 and runs it on the paragraph', async () => {
    const workspace = path.join(resultRoot, 'heading');
    await rm(workspace, { recursive: true, force: true });
    const folder = path.join(workspace, 'vault');
    await mkdir(folder, { recursive: true });
    await writeFile(path.join(folder, 'note.md'), 'A paragraph.\n', 'utf8');

    const app = await electron.launch({
      executablePath: packagedExecutable(),
      args: [`--user-data-dir=${path.join(workspace, 'user-data')}`, `--open=${path.join(folder, 'note.md')}`],
    });
    const page = await app.firstWindow();
    await page.waitForSelector('[data-testid="noto-editor"]', { state: 'visible', timeout: 30_000 });
    await page.setViewportSize({ width: 1100, height: 700 });
    await placeCaret(page, page.locator('.canvas-slot:not([hidden]) .ProseMirror p').first());

    try {
      await invokeMenu(app, 'command-palette');
      await expect(page.getByTestId('command-palette')).toBeVisible();
      await page.getByTestId('command-input').fill('heading 1');
      await expect(page.getByTestId('command-row').first()).toContainText('Heading 1');
      await page.keyboard.press('Enter');
      await expect(page.getByTestId('command-palette')).toHaveCount(0);
      await expect(page.locator('.canvas-slot:not([hidden]) .ProseMirror h1')).toHaveText('A paragraph.');
    } finally {
      await app.close();
    }
  });
});
