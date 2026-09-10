import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { expect, test, _electron as electron, type ElectronApplication, type Page } from '@playwright/test';
import { packagedExecutable, placeCaret } from './packaged-app';

const resultRoot = path.join(process.cwd(), 'test-results', 'todo-manager');

async function launch(
  name: string,
  note: string,
): Promise<{ app: ElectronApplication; page: Page; file: string }> {
  const workspace = path.join(resultRoot, name);
  await rm(workspace, { recursive: true, force: true });
  await mkdir(path.join(workspace, 'user-data'), { recursive: true });
  const file = path.join(workspace, 'note.md');
  await writeFile(file, note, 'utf8');
  const app = await electron.launch({
    executablePath: packagedExecutable(),
    args: [`--user-data-dir=${path.join(workspace, 'user-data')}`, `--open=${file}`],
  });
  const page = await app.firstWindow();
  await page.waitForSelector('[data-testid="noto-editor"]', { state: 'visible', timeout: 30_000 });
  await page.setViewportSize({ width: 1100, height: 700 });
  return { app, page, file };
}

async function run(app: ElectronApplication, id: string): Promise<void> {
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
    const found = menu ? find(menu.items) : null;
    if (!found) throw new Error(`No menu item with id ${itemId}`);
    found.click();
  }, id);
}

test.describe('todo-manager', () => {
  test('writes today beside a newly checked task', async () => {
    const { app, page, file } = await launch(
      'stamp',
      '# Tasks\n\n- [ ] milk\n- [x] bread\n',
    );
    try {
      const first = page.locator('.ProseMirror li.noto-task-item').first();
      const box = await first.boundingBox();
      if (!box) throw new Error('no task');
      await page.mouse.click(box.x + 1, box.y + box.height / 2);

      await page.getByTestId('save-button').click();
      await expect.poll(() => readFile(file, 'utf8'), { timeout: 15_000 }).toMatch(
        /^# Tasks\n\n- \[x\] milk ✅ \d{4}-\d{2}-\d{2}\n- \[x\] bread\n$/,
      );
    } finally {
      await app.close();
    }
  });

  test('sorts open tasks above done ones from the menu', async () => {
    const { app, page, file } = await launch(
      'sort',
      [
        '# Tasks',
        '',
        '- [x] old ✅ 2026-01-01',
        '- [ ] open',
        '- [x] newer ✅ 2026-02-01',
        '',
      ].join('\n'),
    );
    try {
      await placeCaret(page, page.locator('.ProseMirror li.noto-task-item').first());
      await run(app, 'sort-tasks');
      await page.getByTestId('save-button').click();
      await expect.poll(() => readFile(file, 'utf8'), { timeout: 15_000 }).toBe([
        '# Tasks',
        '',
        '- [ ] open',
        '- [x] old ✅ 2026-01-01',
        '- [x] newer ✅ 2026-02-01',
        '',
      ].join('\n'));
    } finally {
      await app.close();
    }
  });
});
