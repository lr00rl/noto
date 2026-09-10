import { mkdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { expect, test, _electron as electron, type ElectronApplication, type Page } from '@playwright/test';
import { packagedExecutable } from './packaged-app';

const resultRoot = path.join(process.cwd(), 'test-results', 'sidenotes');

const NOTE = [
  '# Notes',
  '',
  'A claim<span class="sidenote">The source for this claim.</span> and another<span class="sidenote">A second note.</span> after.',
  '',
].join('\n');

async function launch(): Promise<{ app: ElectronApplication; page: Page }> {
  const workspace = path.join(resultRoot, 'note');
  await rm(workspace, { recursive: true, force: true });
  await mkdir(path.join(workspace, 'user-data'), { recursive: true });
  const file = path.join(workspace, 'note.md');
  await writeFile(file, NOTE, 'utf8');
  const app = await electron.launch({
    executablePath: packagedExecutable(),
    args: [`--user-data-dir=${path.join(workspace, 'user-data')}`, `--open=${file}`],
  });
  const page = await app.firstWindow();
  await page.waitForSelector('[data-testid="noto-editor"]', { state: 'visible', timeout: 30_000 });
  await page.setViewportSize({ width: 1200, height: 700 });
  return { app, page };
}

test.describe('sidenotes', () => {
  test('number the notes and hide the tags until the caret enters', async () => {
    const { app, page } = await launch();
    try {
      const markers = page.locator('.ProseMirror .noto-sidenote-num');
      await expect(markers).toHaveCount(2);
      await expect(markers.nth(0)).toHaveAttribute('data-sidenote-index', '1');
      await expect(markers.nth(1)).toHaveAttribute('data-sidenote-index', '2');

      const notes = page.locator('.ProseMirror .noto-sidenote');
      await expect(notes).toHaveCount(2);
      await expect(notes.nth(0)).toContainText('The source for this claim.');

      // Tags stay in the DOM (byte-faithful) but are display:none until edited.
      // Assert via innerText so hidden source is not mistaken for a drawing bug.
      await expect(page.locator('.ProseMirror')).not.toContainText('<span class="sidenote">', {
        useInnerText: true,
      });
      await expect(page.locator('.ProseMirror .noto-sidenote-tag').first()).toBeHidden();

      // Off: the decorations go and the tags return as source.
      await page.evaluate(() => window.notoSettings.write({
        version: 1, requestId: 'sidenotes-off', patch: { sidenotes: false },
      }));
      await expect(page.locator('html')).toHaveAttribute('data-sidenotes', 'off');
      await expect(page.locator('.ProseMirror .noto-sidenote-num')).toHaveCount(0);
      await expect(page.locator('.ProseMirror')).toContainText('<span class="sidenote">', {
        useInnerText: true,
      });
    } finally {
      await app.close();
    }
  });
});
