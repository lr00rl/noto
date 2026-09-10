import { describe, expect, it } from 'vitest';
import {
  TIMELINE_LANGUAGE,
  TIMELINE_TEMPLATE,
  escapeHtml,
  inlineToHtml,
  isTimelineLanguage,
  parseTimeline,
  timelineHtml,
} from '../../src/renderer/editor/noto/timeline';

describe('recognising the language', () => {
  it('accepts the author\'s spelling, any case', () => {
    expect(isTimelineLanguage('timeline')).toBe(true);
    expect(isTimelineLanguage(' Timeline ')).toBe(true);
    expect(isTimelineLanguage('mermaid')).toBe(false);
  });
});

describe('parsing a timeline fence', () => {
  it('reads a title, times and mixed content', () => {
    const source = [
      '# Voyage',
      '## 2024-01',
      'Set out.',
      '### Notes',
      '- pack',
      '- go',
      '1. first',
      '1. second',
      '> a quote',
      '- [x] done',
      '- [ ] open',
      '---',
      '## 2025',
      'Arrived **home**.',
    ].join('\n');
    const parsed = parseTimeline(source);
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    expect(parsed.data.title).toBe('Voyage');
    expect(parsed.data.buckets).toHaveLength(2);
    expect(parsed.data.buckets[0]!.time).toBe('2024-01');
    const types = parsed.data.buckets[0]!.items.map((item) => item.type);
    expect(types).toEqual(['p', 'h3', 'ul', 'ol', 'blockquote', 'task', 'task', 'hr']);
    const ul = parsed.data.buckets[0]!.items[2]!;
    expect(ul).toMatchObject({ type: 'ul', list: ['pack', 'go'] });
    const done = parsed.data.buckets[0]!.items[5]!;
    expect(done).toMatchObject({ type: 'task', checked: true, value: 'done' });
    expect(parsed.data.buckets[1]!.items[0]).toMatchObject({
      type: 'p',
      value: 'Arrived **home**.',
    });
  });

  it('refuses content before a time and a second title', () => {
    expect(parseTimeline('orphan\n## 2024\nok')).toMatchObject({
      ok: false,
      reason: expect.stringContaining('time'),
    });
    expect(parseTimeline('# A\n# B\n## 2024\nok')).toMatchObject({
      ok: false,
      reason: expect.stringContaining('one title'),
    });
    expect(parseTimeline('## 2024\nok\n# late')).toMatchObject({
      ok: false,
      reason: expect.stringContaining('before any time'),
    });
  });
});

describe('HTML drawing', () => {
  it('escapes raw markup and paints inline marks', () => {
    expect(escapeHtml('<script>"x"&')).toBe('&lt;script&gt;&quot;x&quot;&amp;');
    expect(inlineToHtml('say **hi** and `code`')).toBe(
      'say <strong>hi</strong> and <code>code</code>',
    );
    const html = timelineHtml({
      title: 'T <x>',
      buckets: [{ time: '2024', items: [{ type: 'p', value: '**go**' }] }],
    });
    expect(html).toContain('noto-timeline');
    expect(html).toContain('T &lt;x&gt;');
    expect(html).toContain('<strong>go</strong>');
    expect(html).toContain('noto-timeline-time');
  });
});

describe('the insert template', () => {
  it('is ordinary timeline markdown with two stops', () => {
    expect(TIMELINE_LANGUAGE).toBe('timeline');
    const parsed = parseTimeline(TIMELINE_TEMPLATE);
    expect(parsed.ok).toBe(true);
    if (!parsed.ok) return;
    expect(parsed.data.buckets).toHaveLength(2);
  });
});
