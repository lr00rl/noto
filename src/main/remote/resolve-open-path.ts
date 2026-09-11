/**
 * Where a remote `/v1/open` may land.
 *
 * Main already confines every other path to the open folder. The remote open
 * callback used to resolve a relative name against that folder and then hand
 * the result straight to `openPath`, which does not itself check containment
 * (File > Open is allowed to leave the folder and adopt a new one). Absolute
 * paths such as `/etc/hosts` therefore opened whenever the code viewer would
 * show them, which is how a caller without a browser or a bad token still
 * reached a file outside the vault.
 *
 * This helper is the check the remote path was documented to have: resolve
 * against the vault, refuse anything that does not stay inside it. Lexical
 * containment catches the absolute and `..` cases on every OS; when the path
 * exists, following symlinks catches a link inside the vault that points out.
 */

import path from 'node:path';
import { isInside } from '../workspace/file-tree';

export type PathApi = Pick<
  path.PlatformPath,
  'resolve' | 'normalize' | 'isAbsolute' | 'relative'
>;

/**
 * Resolve `requested` against `vault` and return the absolute path when it
 * stays inside the vault. Null when there is no vault, the request is empty,
 * or the resolved path would leave the folder.
 *
 * Pure string work so unit tests can pass `path.posix` or `path.win32` and
 * exercise both spellings without touching the disk.
 */
export function resolveRemoteOpenPath(
  vault: string | null,
  requested: string,
  pathApi: PathApi = path,
): string | null {
  if (vault === null || vault.length === 0) return null;
  if (requested.length === 0 || /[\0\r\n]/.test(requested)) return null;

  const root = pathApi.resolve(vault);
  // Always resolve against the vault so a Windows absolute like `/etc/hosts`
  // keeps the vault's drive (`C:\etc\hosts`) rather than a bare `\etc\hosts`.
  // On POSIX an absolute second argument still replaces, which is what we want.
  const candidate = pathApi.resolve(root, requested);

  return isInside(root, candidate, pathApi) ? candidate : null;
}

/**
 * Same as `resolveRemoteOpenPath`, then follow symlinks when the path exists.
 *
 * A missing file that is lexically inside the vault is still returned so the
 * open can answer `no-such-note` rather than looking like an outside refusal.
 * A link that resolves outside the vault is refused.
 */
export async function confineRemoteOpenPath(
  vault: string | null,
  requested: string,
  options: {
    readonly pathApi?: PathApi;
    readonly realpath: (target: string) => Promise<string>;
  },
): Promise<string | null> {
  const pathApi = options.pathApi ?? path;
  const candidate = resolveRemoteOpenPath(vault, requested, pathApi);
  if (candidate === null || vault === null) return null;

  let realRoot: string;
  try {
    realRoot = await options.realpath(pathApi.resolve(vault));
  } catch {
    return null;
  }

  try {
    const real = await options.realpath(candidate);
    return isInside(realRoot, real, pathApi) ? real : null;
  } catch {
    // Does not exist yet: lexical containment already held.
    return candidate;
  }
}
