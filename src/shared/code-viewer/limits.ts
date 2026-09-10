/** Files larger than this (bytes) are not painted; a notice is shown. */
export const CODE_VIEW_MAX_BYTES = 4_000_000;

/** Lines beyond this are omitted so a huge file does not build a giant DOM. */
export const CODE_VIEW_MAX_LINES = 50_000;
