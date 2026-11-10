"""Core rough mosaic functions extracted from the marimo notebook.
"""

import warnings

import numpy as np
from skimage.feature import match_template

__all__ = ["rough_mosaic", "cumulative_offsets"]


def _top_peaks(cc, k):
	cc = cc.copy()
	for _ in range(k):
		idx = np.argmax(cc)
		y, x = np.unravel_index(idx, cc.shape)
		yield cc[y, x], y, x
		cc[y, x] = -np.inf


def _overlap_ratio(prev, cur, dy, dx, min_overlap):
	"""Score how well `cur` fits on `prev` when shifted by (dy, dx).

	min_overlap is an AREA in pixels, matching MATLAB's
	`sum(mskx(:)) < min_overlap` test, not a linear dimension.
	"""
	m, n = prev.shape
	mc, nc = cur.shape
	y0, y1 = max(0, dy), min(m, dy + mc)
	x0, x1 = max(0, dx), min(n, dx + nc)
	if (y1 - y0) <= 0 or (x1 - x0) <= 0:
		return np.inf
	if (y1 - y0) * (x1 - x0) < min_overlap:
		return np.inf
	a = prev[y0:y1, x0:x1]
	b = cur[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
	s1, s2 = a.std(), b.std()
	if s1 == 0.0 or s2 == 0.0:
		return np.inf
	mu = np.abs(a - b).mean()
	return (mu * mu) / (s1 * s2)


def rough_mosaic(strips, ov=32, numbest=16):
	"""Rough-mosaic phase: coarse pairwise offsets between strips.

	ov is the minimum overlap between images in pixels. MATLAB prompts for
	it before this step and defaults to 32; it sets the template band width,
	the correlation search region and the minimum-overlap area together.

	Returns
	-------
	rel_reg : (p0, 4) float array. Row i = [neighbor_index, dy, dx, ratio].
	"""
	p0 = len(strips)
	m0, n0 = strips[0].shape
	for i, s in enumerate(strips):
		if s.shape != (m0, n0):
			raise ValueError(
				f"strip {i} has shape {s.shape}, expected {(m0, n0)}; "
				"all strips must match"
			)
	# template bands, centred on the strip (MATLAB nblk1:nblk2, mblk1:mblk2)
	ncol = slice(max(0, n0 // 2 - 2 * ov), min(n0, n0 // 2 + 2 * ov))
	mrow = slice(max(0, m0 // 2 - 2 * ov), min(m0, m0 // 2 + 2 * ov))
	# correlation search region, twice as wide (MATLAB nsearch1:nsearch2)
	nsearch = slice(max(0, n0 // 2 - 4 * ov), min(n0, n0 // 2 + 4 * ov))
	msearch = slice(max(0, m0 // 2 - 4 * ov), min(m0, m0 // 2 + 4 * ov))
	# MATLAB min_overlap is an AREA: min((nblk2-nblk1)*ov, (mblk2-mblk1)*ov)
	min_overlap = min((ncol.stop - ncol.start) * ov, (mrow.stop - mrow.start) * ov)

	rel_reg = np.zeros((p0, 4))
	for inc in range(1, p0):
		prev = np.asarray(strips[inc - 1], dtype=np.float64)
		cur = np.asarray(strips[inc], dtype=np.float64)

		# each template is matched only within the restricted search region
		# of the previous strip, as MATLAB does (img2_1(:, nsearch1:nsearch2)
		# for the top/bottom bands, img2_1(msearch1:msearch2, :) for left/right)
		templates = [
			(cur[:ov, ncol], 0, ncol.start, (slice(None), nsearch)),
			(cur[-ov:, ncol], m0 - ov, ncol.start, (slice(None), nsearch)),
			(cur[mrow, :ov], mrow.start, 0, (msearch, slice(None))),
			(cur[mrow, -ov:], mrow.start, n0 - ov, (msearch, slice(None))),
		]

		best = (np.inf, 0, 0)
		for temp, ay, ax, region in templates:
			if temp.std() == 0.0:
				continue
			th, tw = temp.shape
			search = prev[region]
			if search.shape[0] < th or search.shape[1] < tw:
				continue
			oy = region[0].start or 0
			ox = region[1].start or 0
			cc = match_template(
				search, temp, pad_input=True, mode="constant", constant_values=0
			)
			for _score, ypk, xpk in _top_peaks(cc, numbest):
				py = ypk - th // 2 + oy
				px = xpk - tw // 2 + ox
				dy, dx = int(py - ay), int(px - ax)
				r = _overlap_ratio(prev, cur, dy, dx, min_overlap)
				if r < best[0]:
					best = (r, dy, dx)

		if not np.isfinite(best[0]):
			warnings.warn(
				f"rough_mosaic: no valid match for strip {inc} against "
				f"strip {inc - 1}; offset defaulted to (0, 0)"
			)
		rel_reg[inc] = [inc - 1, best[1], best[2], best[0]]
	return rel_reg


def cumulative_offsets(rel_reg):
	"""Relative offsets to absolute (row, col) placement per strip."""
	real_reg = np.cumsum(rel_reg[:, 1:3], axis=0)
	real_reg -= real_reg.min(axis=0)
	return real_reg.astype(np.int64)
