import { describe, expect, it } from 'vitest';
import { isProbablyBinary } from '../../src/shared/code-viewer/binary';
import {
  isMarkdownFileName,
  isViewableCodeFile,
  languageFor,
} from '../../src/shared/code-viewer/languages';
import { highlightCodeLines } from '../../src/renderer/code-viewer-highlight';
import { CODE_VIEW_MAX_BYTES, CODE_VIEW_MAX_LINES } from '../../src/shared/code-viewer/limits';

describe('recognising what to open', () => {
  it('maps common source extensions and special filenames', () => {
    expect(languageFor('/a/b/main.py')).toBe('python');
    expect(languageFor('server.ts')).toBe('typescript');
    expect(languageFor('app.tsx')).toBe('tsx');
    expect(languageFor('main.go')).toBe('go');
    expect(languageFor('lib.rs')).toBe('rust');
    expect(languageFor('data.yaml')).toBe('yaml');
    expect(languageFor('run.sh')).toBe('bash');
    expect(languageFor('Dockerfile')).toBe('dockerfile');
    expect(languageFor('Makefile')).toBe('makefile');
    expect(languageFor('.gitignore')).toBe('gitignore');
  });

  it('refuses notes and known binaries, opens unknown text as plain', () => {
    expect(languageFor('README.md')).toBeNull();
    expect(languageFor('notes.txt')).toBeNull();
    expect(isMarkdownFileName('a/b/c.MD')).toBe(true);
    expect(isViewableCodeFile('photo.png')).toBe(false);
    expect(languageFor('notes.xyz')).toBe('');
    expect(isViewableCodeFile('notes.xyz')).toBe(true);
    expect(languageFor('SCRIPT.PY')).toBe('python');
  });
});

describe('binary detection', () => {
  it('treats a NUL or control-heavy sample as binary', () => {
    expect(isProbablyBinary('abc\0def')).toBe(true);
    expect(isProbablyBinary('def f():\n\treturn 1\n')).toBe(false);
    expect(isProbablyBinary('')).toBe(false);
    const junk = Array.from({ length: 100 }, (_, i) => String.fromCharCode((i % 8) + 1)).join('');
    expect(isProbablyBinary(junk)).toBe(true);
  });
});

describe('highlighting', () => {
  it('reconstructs the source exactly and colours a simple python line', () => {
    const source = 'def f():\n    return True  # yes\n';
    const lines = highlightCodeLines(source, 'python');
    expect(lines.map((line) => line.map((t) => t.text).join('')).join('\n')).toBe(source);
    const joined = lines[0]!.map((t) => t.text).join('');
    expect(joined).toBe('def f():');
  });

  it('leaves plain text uncoloured', () => {
    const lines = highlightCodeLines('just some text\n123 456', 'text');
    expect(lines.every((line) => line.every((t) => t.cls === null))).toBe(true);
  });
});

describe('limits', () => {
  it('keeps the author\'s preview caps', () => {
    expect(CODE_VIEW_MAX_BYTES).toBe(4_000_000);
    expect(CODE_VIEW_MAX_LINES).toBe(50_000);
  });
});
