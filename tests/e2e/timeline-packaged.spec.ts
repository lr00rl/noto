import { mkdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { expect, test, _electron as electron, type ElectronApplication, type Page } from '@playwright/test';
import { packagedExecutable, placeCaret } from './packaged-app';

const resultRoot = path.join(process.cwd(), 'test-results', 'timeline');

const NOTE = [
  '# Chronology',
  '',
  'Before.',
  '',
  '```timeline',
  '# Project',
  '## 2024',
  'Started.',
  '## 2025',
  'Shipped.',
  '```',
  '',
  'After.',
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

test.describe('timelines', () => {
  test('draws a timeline fence and hides the source until the caret enters', async () => {
    const { app, page } = await launch();
    try {
      const fence = page.locator('pre.noto-fence[data-lang="timeline"]');
      const frame = fence.locator('.noto-timeline-frame');
      await expect(frame).toHaveAttribute('data-state', 'rendered', { timeout: 10_000 });
      await expect(frame.locator('.noto-timeline-title')).toHaveText('Project');
      await expect(frame.locator('.noto-timeline-time')).toHaveCount(2);
      await expect(frame.locator('.noto-timeline-time').nth(0)).toHaveText('2024');

      const hidden = await fence.locator('.noto-fence-code').evaluate((element) => {
        const rect = element.getBoundingClientRect();
        return rect.width <= 1 && rect.height <= 1;
      });
      expect(hidden).toBe(true);

      await placeCaret(page, frame);
      await expect(fence).toHaveClass(/noto-active-block/);
      const shown = await fence.locator('.noto-fence-code').evaluate(
        (element) => element.getBoundingClientRect().height,
      );
      expect(shown).toBeGreaterThan(20);

      await page.evaluate(() => window.notoSettings.write({
        version: 1, requestId: 'timelines-off', patch: { timelines: false },
      }));
      await expect(page.locator('html')).toHaveAttribute('data-timelines', 'off');
      await expect(frame).toBeHidden();
    } finally {
      await app.close();
    }
  });
});
