/**
 * Keep a floating surface on screen, next to the thing that summoned it.
 *
 * Slash sits under the caret and the format HUD sits above the selection.
 * Both have to flip when that side of the window is too small, and both
 * have to slide horizontally when they would hang off the right edge. The
 * editor reports the anchor; this decides the top-left of the overlay once
 * the overlay has a size.
 */

export interface OverlayAnchor {
  readonly left: number;
  readonly top: number;
  readonly bottom: number;
}

export interface OverlaySize {
  readonly width: number;
  readonly height: number;
}

export interface ViewportBox {
  readonly width: number;
  readonly height: number;
}

const GAP = 6;
/** The titlebar is 32px and is not a place for a menu. */
const TOP_INSET = 36;

export function placeOverlay(
  anchor: OverlayAnchor,
  size: OverlaySize,
  prefer: 'above' | 'below',
  viewport: ViewportBox,
  pad = 8,
  align: 'start' | 'center' = 'start',
): { left: number; top: number } {
  let left = align === 'center' ? anchor.left - size.width / 2 : anchor.left;
  const maxLeft = viewport.width - pad - size.width;
  if (left > maxLeft) left = Math.max(pad, maxLeft);
  if (left < pad) left = pad;

  const below = anchor.bottom + GAP;
  const above = anchor.top - GAP - size.height;
  const fitsBelow = below + size.height <= viewport.height - pad;
  const fitsAbove = above >= TOP_INSET;

  let top: number;
  if (prefer === 'below') {
    top = fitsBelow || !fitsAbove ? below : above;
  } else {
    top = fitsAbove || !fitsBelow ? above : below;
  }
  const maxTop = viewport.height - pad - size.height;
  if (top > maxTop) top = Math.max(TOP_INSET, maxTop);
  if (top < TOP_INSET) top = TOP_INSET;
  return { left, top };
}
