import { mkdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { expect, test, _electron as electron, type ElectronApplication, type Page } from '@playwright/test';
import { packagedExecutable } from './packaged-app';

const resultRoot = path.join(process.cwd(), 'test-results', 'tab-markers');

const NOTE = [
  '# Tabs', '',
  '```makefile',
  'all:\tdeps',
  '\t@echo hi',
  '```', '',
].join('\n');

async function launch(): Promise<{ app: ElectronApplication; page: Page }> {
  const workspace = path.join(resultRoot, 'tabs');
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
  await page.setViewportSize({ width: 1000, height: 700 });
  return { app, page };
}

test.describe('tab markers', () => {
  test('paint an arrow on each tab, and go when the setting says so', async () => {
    const { app, page } = await launch();
    try {
      const markers = page.locator('.ProseMirror pre .noto-code-tab');
      // One on "all:\tdeps", one opening the recipe line.
      await expect(markers).toHaveCount(2);

      const painted = () => markers.first().evaluate((el) => {
        const image = getComputedStyle(el).backgroundImage;
        return image !== 'none' && image.includes('url');
      });
      expect(await painted()).toBe(true);

      // The tab is still the character, so the code still copies as code.
      expect(await page.locator('.ProseMirror pre code').innerText()).toContain('\t');

      await page.evaluate(() => window.notoSettings.write({
        version: 1, requestId: 'tabs-off', patch: { codeTabMarkers: false },
      }));
      await expect(page.locator('html')).toHaveAttribute('data-code-tab-markers', 'off');
      await expect.poll(painted).toBe(false);
    } finally {
      await app.close();
    }
  });
});
