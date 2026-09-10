/**
 * Prism tokenisation for the read-only code viewer.
 *
 * Reuses the same Prism bundle the fence highlighter already loads. Returns
 * plain token lists so the React pane can paint with textContent, never HTML.
 */

import Prism from 'prismjs';
// Side-effect imports: the fence highlighter already registers these; repeat
// the core set so this module stays usable if loaded alone in a unit test.
import 'prismjs/components/prism-markup';
import 'prismjs/components/prism-css';
import 'prismjs/components/prism-clike';
import 'prismjs/components/prism-javascript';
import 'prismjs/components/prism-typescript';
import 'prismjs/components/prism-jsx';
import 'prismjs/components/prism-tsx';
import 'prismjs/components/prism-json';
import 'prismjs/components/prism-yaml';
import 'prismjs/components/prism-bash';
import 'prismjs/components/prism-python';
import 'prismjs/components/prism-rust';
import 'prismjs/components/prism-go';
import 'prismjs/components/prism-java';
import 'prismjs/components/prism-c';
import 'prismjs/components/prism-cpp';
import 'prismjs/components/prism-sql';
import 'prismjs/components/prism-diff';
import 'prismjs/components/prism-toml';
import 'prismjs/components/prism-docker';
import 'prismjs/components/prism-ini';
import 'prismjs/components/prism-lua';
import 'prismjs/components/prism-ruby';
import 'prismjs/components/prism-markup-templating';
import 'prismjs/components/prism-php';
import 'prismjs/components/prism-csharp';
import 'prismjs/components/prism-powershell';
import 'prismjs/components/prism-vim';
import 'prismjs/components/prism-makefile';
import 'prismjs/components/prism-swift';
import 'prismjs/components/prism-kotlin';
import 'prismjs/components/prism-scss';
import 'prismjs/components/prism-lisp';
import 'prismjs/components/prism-json5';
import 'prismjs/components/prism-properties';
import 'prismjs/components/prism-latex';
import 'prismjs/components/prism-r';
import 'prismjs/components/prism-perl';
import 'prismjs/components/prism-elixir';
import 'prismjs/components/prism-erlang';
import 'prismjs/components/prism-scala';
import 'prismjs/components/prism-dart';
import 'prismjs/components/prism-groovy';
import 'prismjs/components/prism-zig';
import 'prismjs/components/prism-protobuf';
import 'prismjs/components/prism-graphql';

export interface CodeToken {
  readonly text: string;
  readonly cls: string | null;
}

const ALIASES: Readonly<Record<string, string>> = {
  js: 'javascript',
  ts: 'typescript',
  py: 'python',
  sh: 'bash',
  shell: 'bash',
  zsh: 'bash',
  yml: 'yaml',
  html: 'markup',
  xml: 'markup',
  svg: 'markup',
  htm: 'markup',
  dockerfile: 'docker',
  csharp: 'csharp',
  'c#': 'csharp',
  cpp: 'cpp',
  'c++': 'cpp',
  text: '',
  plain: '',
  txt: '',
};

function grammarFor(language: string): Prism.Grammar | null {
  const key = ALIASES[language.toLowerCase()] ?? language.toLowerCase();
  if (!key) return null;
  return Prism.languages[key] ?? null;
}

function flatten(
  tokens: ReadonlyArray<string | Prism.Token>,
  into: CodeToken[],
): void {
  for (const token of tokens) {
    if (typeof token === 'string') {
      if (token.length > 0) into.push({ text: token, cls: null });
      continue;
    }
    const content = token.content;
    if (typeof content === 'string') {
      into.push({ text: content, cls: token.type });
    } else if (Array.isArray(content)) {
      // Nested tokens keep the outer type only when leaves have none.
      const nested: CodeToken[] = [];
      flatten(content, nested);
      for (const part of nested) {
        into.push({ text: part.text, cls: part.cls ?? token.type });
      }
    } else {
      flatten([content], into);
    }
  }
}

/** Tokenise one whole file, then split the flat stream back into lines. */
export function highlightCodeLines(source: string, language: string): CodeToken[][] {
  const lines = source.replace(/\r\n?/g, '\n').split('\n');
  // A trailing empty string from a final newline is a blank last line; keep it.
  const grammar = grammarFor(language);
  if (!grammar) {
    return lines.map((line) => (line.length === 0 ? [] : [{ text: line, cls: null }]));
  }

  const flat: CodeToken[] = [];
  flatten(Prism.tokenize(source.replace(/\r\n?/g, '\n'), grammar), flat);

  const result: CodeToken[][] = Array.from({ length: lines.length }, () => []);
  let lineIndex = 0;
  for (const token of flat) {
    const parts = token.text.split('\n');
    for (let i = 0; i < parts.length; i += 1) {
      if (i > 0) lineIndex += 1;
      if (lineIndex >= result.length) break;
      const piece = parts[i]!;
      if (piece.length > 0) result[lineIndex]!.push({ text: piece, cls: token.cls });
    }
  }
  return result;
}
