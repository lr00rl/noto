import { describe, expect, it } from 'vitest';
import { sourceHasFinalNewline, sourceModeText } from '../../src/renderer/source-mode-text';

describe('Source Code Mode text', () => {
  it('shows the accepted file text, LF-normalised, while the note is clean', () => {
    expect(sourceModeText({
      dirty: false,
      fileText: '# A\r\n\r\n\r\nBody with a wide gap.\r\n',
      reconstructed: '# A\n\nBody with a wide gap.',
      hasFinalNewline: true,
    })).toBe('# A\n\n\nBody with a wide gap.\n');
  });

  it('reconstructs from the editor once the note is dirty', () => {
    expect(sourceModeText({
      dirty: true,
      fileText: '# A\n\n\nBody.\n',
      reconstructed: '# A\n\nBody changed.',
      hasFinalNewline: true,
    })).toBe('# A\n\nBody changed.\n');

    expect(sourceModeText({
      dirty: true,
      fileText: '# A\n\nBody.\n',
      reconstructed: '# A\n\nBody.',
      hasFinalNewline: false,
    })).toBe('# A\n\nBody.');
  });

  it('reads the trailing newline from the buffer for the envelope', () => {
    expect(sourceHasFinalNewline('# A\n')).toBe(true);
    expect(sourceHasFinalNewline('# A')).toBe(false);
    expect(sourceHasFinalNewline('')).toBe(false);
  });
});
