import { mkdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { expect, test, _electron as electron, type ElectronApplication, type Page } from '@playwright/test';
import { packagedExecutable } from './packaged-app';

const resultRoot = path.join(process.cwd(), 'test-results', 'code-viewer');

async function launch(folder: string): Promise<{ app: ElectronApplication; page: Page }> {
  const userData = path.join(folder, 'user-data');
  await mkdir(userData, { recursive: true });
  const note = path.join(folder, 'note.md');
  await writeFile(note, '# Hello\n\nA note.\n', 'utf8');
  await writeFile(path.join(folder, 'sample.py'), 'def greet(name):\n    return f"hi {name}"\n', 'utf8');
  const app = await electron.launch({
    executablePath: packagedExecutable(),
    args: [`--user-data-dir=${userData}`, `--open=${note}`, `--folder=${folder}`],
  });
  const page = await app.firstWindow();
  await page.waitForSelector('[data-testid="noto-editor"]', { state: 'visible', timeout: 30_000 });
  await page.setViewportSize({ width: 1200, height: 700 });
  return { app, page };
}

test.describe('code viewer', () => {
  test('opens a .py from the tree as a read-only highlighted pane', async () => {
    const workspace = path.join(resultRoot, 'py');
    await rm(workspace, { recursive: true, force: true });
    await mkdir(workspace, { recursive: true });
    const { app, page } = await launch(workspace);
    try {
      await page.getByTestId('file-tree').waitFor({ state: 'visible', timeout: 10_000 });
      await page.locator('[data-testid="tree-file"][data-path$="sample.py"]').click();

      const viewer = page.getByTestId('code-viewer');
      await expect(viewer).toBeVisible({ timeout: 10_000 });
      await expect(viewer.locator('.code-viewer-name')).toHaveText('sample.py');
      await expect(viewer.locator('.code-viewer-ro')).toHaveText('Read only');
      await expect(viewer.locator('.code-viewer-lang')).toHaveText('python');
      await expect(viewer.locator('.code-viewer-src').first()).toContainText('def');

      await page.evaluate(() => window.notoSettings.write({
        version: 1, requestId: 'code-viewer-off', patch: { codeViewer: false },
      }));
      await expect(viewer).toHaveCount(0);
    } finally {
      await app.close();
    }
  });
});
