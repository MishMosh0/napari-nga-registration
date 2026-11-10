"""MATLAB-compatible `imresize`.

`skimage.transform.resize` is not a drop-in substitute for MATLAB's
`imresize(..., 'bicubic')`. Verified against a MATLAB-produced downsample of
the panel C reference canvas, skimage differs by 4.9% RMS with peak errors of
~6800 counts; this module matches to 99.97% of pixels exactly, with no pixel
off by more than one count.

Four things have to line up, and each was measured separately on that data:

1. Kernel. MATLAB uses the Keys cubic convolution kernel with a = -0.5.
   skimage `order=3` applies a cubic B-spline prefilter, a different
   interpolant. The signature difference is ringing: Keys under- and
   overshoots a step edge, the B-spline does not.

2. Scale. Given a scalar factor, MATLAB sets the output size to
   `ceil(scale * in_size)` but keeps using the *requested* scale for the
   kernel, not `out_size / in_size`. On the panel C canvas the column axis
   goes 1739 -> 435, a ratio of 0.250144 rather than 0.25, and using the
   ratio costs a factor of 15 in accuracy (228 vs 15 RMS).

3. Edges. MATLAB mirrors out-of-range indices with `[1..n, n..1]` rather than
   clamping to the edge sample. With antialiasing the kernel spans 4/scale
   samples, so this reaches well inside a downsample. Worth 15 -> 9 RMS.

4. Integer class. MATLAB resizes one dimension at a time and keeps the
   original integer class between passes, so a uint16 image is rounded back
   to uint16 after the first pass. Worth 9 -> 3 RMS, and the last of it once
   the output is rounded too.

Reference: Keys, "Cubic Convolution Interpolation for Digital Image
Processing", IEEE ASSP-29(6), 1981.
"""

import numpy as np

__all__ = ["keys_kernel", "imresize_bicubic"]


def keys_kernel(x, a=-0.5):
	"""Keys cubic convolution kernel, support [-2, 2]."""
	x = np.abs(np.asarray(x, float))
	out = np.zeros_like(x)
	m1 = x <= 1.0
	m2 = (x > 1.0) & (x < 2.0)
	x1, x2 = x[m1], x[m2]
	out[m1] = (a + 2.0) * x1**3 - (a + 3.0) * x1**2 + 1.0
	out[m2] = a * x2**3 - 5.0 * a * x2**2 + 8.0 * a * x2 - 4.0 * a
	return out


def _contributions(out_len, in_len, scale, a=-0.5, antialias=True):
	"""Indices and weights for one axis, following MATLAB's contributions().

	Out-of-range indices mirror rather than clamp, matching MATLAB's
	`aux = [1:n, n:-1:1]` trick.
	"""
	# Written exactly as MATLAB does it, 1-based and as `scale * h(scale * x)`.
	# The 0-based `(i + 0.5)/scale - 0.5` with `h(x / kscale) / kscale` is the
	# same algebra, but at sc=0.33 its last-bit differences flip a few
	# near-half pixels when rounding to an integer class (data set 2).
	x = np.arange(1, out_len + 1, dtype=float)
	u = x / scale + 0.5 * (1 - 1 / scale)
	shrink = scale < 1.0 and antialias
	width = 4.0 / scale if shrink else 4.0
	left = np.floor(u - width / 2.0)
	taps = int(np.ceil(width)) + 2

	idx = left[:, None] + np.arange(taps)[None, :]
	if shrink:
		w = scale * keys_kernel(scale * (u[:, None] - idx), a)
	else:
		w = keys_kernel(u[:, None] - idx, a)
	w = w / w.sum(axis=1, keepdims=True)

	mirror = np.concatenate([np.arange(in_len), np.arange(in_len)[::-1]])
	idx = mirror[np.mod(idx.astype(int) - 1, len(mirror))]
	return idx, w


def _resize_axis(img, out_len, axis, scale, a=-0.5, antialias=True):
	idx, w = _contributions(out_len, img.shape[axis], scale, a, antialias)
	taken = np.take(img, idx.ravel(), axis=axis)
	shape = list(img.shape)
	shape[axis : axis + 1] = [out_len, idx.shape[1]]
	taken = taken.reshape(shape)
	wshape = [1] * taken.ndim
	wshape[axis], wshape[axis + 1] = out_len, idx.shape[1]
	return (taken * w.reshape(wshape)).sum(axis=axis + 1)


def imresize_bicubic(img, scale=None, out_shape=None, a=-0.5, antialias=True, dtype=None):
	"""Resize a 2-D array the way MATLAB `imresize(img, ..., 'bicubic')` does.

	Give either `scale` (a scalar factor, as MATLAB's `imresize(A, 0.25)`) or
	`out_shape` as (rows, cols). Prefer `scale` when porting a scalar call:
	MATLAB keeps using the requested factor for the kernel even though the
	output size is rounded up, and reconstructing it from the output shape is
	measurably worse.

	`dtype` sets the class carried between dimension passes. Pass an integer
	dtype to reproduce MATLAB's behaviour on integer images, where the result
	is rounded back after each pass. Defaults to the input's dtype.
	"""
	img = np.asarray(img)
	if img.ndim != 2:
		raise ValueError(f"expected a 2-D array, got shape {img.shape}")
	if (scale is None) == (out_shape is None):
		raise ValueError("give exactly one of scale or out_shape")

	if dtype is None:
		dtype = img.dtype
	dtype = np.dtype(dtype)
	work = img.astype(float)

	if scale is not None:
		sy = sx = float(scale)
		m = int(np.ceil(sy * img.shape[0]))
		n = int(np.ceil(sx * img.shape[1]))
	else:
		m, n = out_shape
		sy, sx = m / img.shape[0], n / img.shape[1]

	def cast(arr):
		if np.issubdtype(dtype, np.integer):
			info = np.iinfo(dtype)
			# MATLAB rounds halves away from zero; np.rint rounds them to even.
			# Exact halves are common on a 1/4 downsample of integer data.
			return np.clip(np.floor(arr + 0.5), info.min, info.max)
		return arr

	work = cast(_resize_axis(work, m, 0, sy, a, antialias))
	work = cast(_resize_axis(work, n, 1, sx, a, antialias))
	return work.astype(dtype) if np.issubdtype(dtype, np.integer) else work
