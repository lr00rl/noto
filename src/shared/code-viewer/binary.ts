/**
 * Text/binary classification for the code viewer.
 *
 * A NUL byte is definitive; otherwise a sample dominated by other C0 controls
 * is treated as binary so the pane shows a notice instead of a giant DOM.
 */

export function isProbablyBinary(content: string, sampleSize = 8192): boolean {
  const n = Math.min(content.length, sampleSize);
  if (n === 0) return false;
  let control = 0;
  for (let i = 0; i < n; i += 1) {
    const code = content.charCodeAt(i);
    if (code === 0) return true;
    if (code < 32 && code !== 9 && code !== 10 && code !== 13) control += 1;
  }
  return control / n > 0.15;
}
