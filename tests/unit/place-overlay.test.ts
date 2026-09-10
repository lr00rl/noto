import { describe, expect, it } from 'vitest';
import { placeOverlay } from '../../src/renderer/place-overlay';

const viewport = { width: 1440, height: 900 };

describe('placing a floating overlay', () => {
  it('sits below the caret when there is room', () => {
    const pos = placeOverlay(
      { left: 200, top: 100, bottom: 118 },
      { width: 220, height: 160 },
      'below',
      viewport,
    );
    expect(pos.left).toBe(200);
    expect(pos.top).toBe(124);
  });

  it('flips above when the window has no room below', () => {
    const pos = placeOverlay(
      { left: 200, top: 780, bottom: 798 },
      { width: 220, height: 160 },
      'below',
      viewport,
    );
    expect(pos.top).toBe(780 - 6 - 160);
  });

  it('centres above a selection and stays on screen at the left edge', () => {
    const pos = placeOverlay(
      { left: 40, top: 80, bottom: 98 },
      { width: 220, height: 32 },
      'above',
      viewport,
      8,
      'center',
    );
    expect(pos.left).toBe(8);
    expect(pos.top).toBe(80 - 6 - 32);
  });

  it('drops below a selection that sits under the titlebar', () => {
    const pos = placeOverlay(
      { left: 400, top: 40, bottom: 58 },
      { width: 220, height: 32 },
      'above',
      viewport,
      8,
      'center',
    );
    expect(pos.top).toBe(64);
  });
});
