/**
 * Seeding the Links rail from in-note wiki targets when the graph skips a MOC.
 */
import { describe, expect, it } from 'vitest';
import {
  markdownForLinkSeed, seedOutboundLinks, wikiLinksInMarkdown,
} from '../../src/renderer/seed-links';
import type { WikiCandidate } from '../../src/renderer/wiki-target';

const entry = (relativePath: string): WikiCandidate => ({
  path: `/vault/${relativePath}`,
  relativePath,
  name: relativePath.split('/').at(-1)!,
});

const VAULT = [
  entry('Z900_MOCs/代理与隧道.md'),
  entry('A000_Theoretical_Knowledge/A404_Linux/00_Linux总览.md'),
  entry('A000_Theoretical_Knowledge/A404_Linux/tcp调优.md'),
  entry('P000_Public/首页.md'),
  entry('V000_Vault/alpha.md'),
];

const MOC = [
  '# 代理与隧道',
  '',
  '<!-- note-assistant:index:start -->',
  '',
  '## 目录索引',
  '',
  '- [[../A000_Theoretical_Knowledge/A404_Linux/00_Linux总览|Linux 总览]]',
  '- [[../A000_Theoretical_Knowledge/A404_Linux/tcp调优|TCP 调优]]',
  '- [[../P000_Public/首页|首页]]',
  '',
  '<!-- note-assistant:index:end -->',
  '',
  'Prose mentions [[V000_Vault/alpha|alpha]] outside the index.',
  '',
].join('\n');

describe('wiki links written in a note', () => {
  it('prefers the index region when seeding a MOC hub', () => {
    const seedBody = markdownForLinkSeed(MOC);
    expect(seedBody).toContain('Linux 总览');
    expect(seedBody).not.toContain('V000_Vault/alpha');
    expect(wikiLinksInMarkdown(seedBody)).toHaveLength(3);
  });

  it('resolves note-relative targets against the vault index', () => {
    const links = seedOutboundLinks(MOC, 'Z900_MOCs/代理与隧道.md', VAULT);
    expect(links.map((link) => link.relativePath)).toEqual([
      'A000_Theoretical_Knowledge/A404_Linux/00_Linux总览.md',
      'A000_Theoretical_Knowledge/A404_Linux/tcp调优.md',
      'P000_Public/首页.md',
    ]);
    expect(links[0].title).toBe('Linux 总览');
  });

  it('falls back to the whole note when there is no index region', () => {
    const note = 'See [[../P000_Public/首页|首页]] and [[tcp调优]].\n';
    const links = seedOutboundLinks(note, 'A000_Theoretical_Knowledge/A404_Linux/00_Linux总览.md', VAULT);
    expect(links.map((link) => link.relativePath)).toEqual([
      'P000_Public/首页.md',
      'A000_Theoretical_Knowledge/A404_Linux/tcp调优.md',
    ]);
  });
});
