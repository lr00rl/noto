import { mkdir, mkdtemp, realpath, rm, symlink, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import {
  confineRemoteOpenPath,
  resolveRemoteOpenPath,
} from '../../src/main/remote/resolve-open-path';
import { isInside } from '../../src/main/workspace/file-tree';

describe('resolveRemoteOpenPath on POSIX paths', () => {
  const posix = path.posix;
  const vault = '/Users/a/vault';

  it('accepts a note inside the vault, relative or absolute', () => {
    expect(resolveRemoteOpenPath(vault, 'second.md', posix)).toBe('/Users/a/vault/second.md');
    expect(resolveRemoteOpenPath(vault, 'notes/a.md', posix)).toBe('/Users/a/vault/notes/a.md');
    expect(resolveRemoteOpenPath(vault, '/Users/a/vault/second.md', posix)).toBe('/Users/a/vault/second.md');
    expect(resolveRemoteOpenPath(vault, '/Users/a/vault/notes/../second.md', posix))
      .toBe('/Users/a/vault/second.md');
  });

  it('refuses absolute paths outside the vault, including /etc/hosts', () => {
    expect(resolveRemoteOpenPath(vault, '/etc/hosts', posix)).toBeNull();
    expect(resolveRemoteOpenPath(vault, '/etc/passwd', posix)).toBeNull();
    expect(resolveRemoteOpenPath(vault, '/Users/a/other/note.md', posix)).toBeNull();
  });

  it('refuses relative climbs that leave the vault', () => {
    expect(resolveRemoteOpenPath(vault, '../secret.md', posix)).toBeNull();
    expect(resolveRemoteOpenPath(vault, '../../etc/hosts', posix)).toBeNull();
    expect(resolveRemoteOpenPath(vault, 'notes/../../outside.md', posix)).toBeNull();
  });

  it('refuses a sibling whose name only starts the same as the vault', () => {
    expect(resolveRemoteOpenPath(vault, '/Users/a/vaultX/note.md', posix)).toBeNull();
    expect(isInside(vault, '/Users/a/vaultX/note.md', posix)).toBe(false);
  });

  it('refuses when there is no vault, and empty or poisoned requests', () => {
    expect(resolveRemoteOpenPath(null, 'a.md', posix)).toBeNull();
    expect(resolveRemoteOpenPath('', 'a.md', posix)).toBeNull();
    expect(resolveRemoteOpenPath(vault, '', posix)).toBeNull();
    expect(resolveRemoteOpenPath(vault, 'a.md\0', posix)).toBeNull();
  });

  it('treats the vault root itself as inside (openPath will still refuse a folder)', () => {
    expect(resolveRemoteOpenPath(vault, '.', posix)).toBe(vault);
    expect(resolveRemoteOpenPath(vault, vault, posix)).toBe(vault);
  });
});

describe('resolveRemoteOpenPath on Windows paths', () => {
  const win32 = path.win32;
  const vault = 'C:\\Users\\a\\vault';

  it('accepts a note inside the vault, relative or absolute', () => {
    expect(resolveRemoteOpenPath(vault, 'second.md', win32)).toBe('C:\\Users\\a\\vault\\second.md');
    expect(resolveRemoteOpenPath(vault, 'notes\\a.md', win32)).toBe('C:\\Users\\a\\vault\\notes\\a.md');
    expect(resolveRemoteOpenPath(vault, 'C:\\Users\\a\\vault\\second.md', win32))
      .toBe('C:\\Users\\a\\vault\\second.md');
  });

  it('refuses /etc/hosts-style absolutes and other drives', () => {
    // On Windows an absolute path starting with / resolves onto the current drive.
    expect(resolveRemoteOpenPath(vault, '/etc/hosts', win32)).toBeNull();
    expect(resolveRemoteOpenPath(vault, 'C:\\etc\\hosts', win32)).toBeNull();
    expect(resolveRemoteOpenPath(vault, 'D:\\vault\\note.md', win32)).toBeNull();
  });

  it('refuses relative climbs that leave the vault', () => {
    expect(resolveRemoteOpenPath(vault, '..\\secret.md', win32)).toBeNull();
    expect(resolveRemoteOpenPath(vault, '..\\..\\Windows\\System32\\drivers\\etc\\hosts', win32))
      .toBeNull();
  });

  it('refuses a sibling whose name only starts the same as the vault', () => {
    expect(resolveRemoteOpenPath(vault, 'C:\\Users\\a\\vaultX\\note.md', win32)).toBeNull();
    expect(isInside(vault, 'C:\\Users\\a\\vaultX\\note.md', win32)).toBe(false);
  });
});

describe('confineRemoteOpenPath follows links out of the vault', () => {
  let vault: string;
  let outside: string;

  beforeAll(async () => {
    const base = await realpath(await mkdtemp(path.join(os.tmpdir(), 'noto-remote-open-')));
    vault = path.join(base, 'vault');
    outside = path.join(base, 'outside');
    await mkdir(vault, { recursive: true });
    await mkdir(outside, { recursive: true });
    await writeFile(path.join(vault, 'inside.md'), '# inside\n');
    await writeFile(path.join(outside, 'secret.md'), '# secret\n');
    await symlink(path.join(outside, 'secret.md'), path.join(vault, 'escape.md'));
  });

  afterAll(async () => {
    await rm(path.dirname(vault), { recursive: true, force: true });
  });

  it('keeps a real note inside the vault', async () => {
    const inside = path.join(vault, 'inside.md');
    await expect(confineRemoteOpenPath(vault, 'inside.md', { realpath })).resolves.toBe(inside);
    await expect(confineRemoteOpenPath(vault, inside, { realpath })).resolves.toBe(inside);
  });

  it('refuses a path outside the vault and a symlink that leads outside', async () => {
    await expect(confineRemoteOpenPath(vault, path.join(outside, 'secret.md'), { realpath }))
      .resolves.toBeNull();
    await expect(confineRemoteOpenPath(vault, '/etc/hosts', { realpath })).resolves.toBeNull();
    await expect(confineRemoteOpenPath(vault, 'escape.md', { realpath })).resolves.toBeNull();
  });

  it('still returns a missing path that is lexically inside, so open can say no-such-note', async () => {
    const missing = path.join(vault, 'gone.md');
    await expect(confineRemoteOpenPath(vault, 'gone.md', { realpath })).resolves.toBe(missing);
  });
});
