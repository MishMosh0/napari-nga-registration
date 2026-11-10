"""Core registration functions extracted from the marimo notebook.
"""

import warnings
from copy import copy
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.linalg import qr, solve_triangular
from scipy.ndimage import correlate, median_filter
from scipy.spatial import Delaunay
from skimage.feature import match_template
from skimage.morphology import local_maxima

from matlab_resize import imresize_bicubic
from sibson import sibson_matrix
from . import fine_register


__all__ = ["Registration", "register_reference", "total_features"]


def poc(img):
	"""Phase-Only Correlation image: FFT2 -> keep phase -> real(IFFT2)."""
	F = np.fft.fft2(img.astype(np.float64))
	return np.real(np.fft.ifft2(np.exp(1j * np.angle(F))))


def _normxcorr2_full(template, image):
	"""MATLAB normxcorr2 'full' output, pure numpy (FFT-based).

	MATLAB's normxcorr2 passes both inputs through shiftData, which subtracts
	the minimum from any non-unsigned array that has negative values. Over
	the full-overlap interior that shift cancels out of the normalised
	correlation, but in the outer ring the template window includes the zero
	padding, and shifting the data moves it relative to those zeros. The
	phase-correlation images fed in here are signed, so without the shift the
	ring differs from MATLAB by up to 0.5, which shifts std(cc) and with it
	the cnt == 1 acceptance gate in art_regx. With it, the whole surface
	matches MATLAB to 6e-13 on every feature point tested.
	"""
	t = template.astype(np.float64)
	f = image.astype(np.float64)
	if t.min() < 0:
		t = t - t.min()
	if f.min() < 0:
		f = f - f.min()
	th, tw = t.shape
	oh, ow = f.shape[0] + th - 1, f.shape[1] + tw - 1
	t0 = t - t.mean()
	Ff = np.fft.rfft2(f, s=(oh, ow))
	num = np.fft.irfft2(Ff * np.fft.rfft2(t0[::-1, ::-1], s=(oh, ow)), s=(oh, ow))
	ones = np.fft.rfft2(np.ones((th, tw)), s=(oh, ow))
	lsum = np.fft.irfft2(Ff * ones, s=(oh, ow))
	lssq = np.fft.irfft2(np.fft.rfft2(f * f, s=(oh, ow)) * ones, s=(oh, ow))
	n = th * tw
	fvar = lssq - (lsum * lsum) / n
	fvar[fvar < 0] = 0
	denom = np.sqrt(fvar * np.sum(t0 * t0))
	cc = np.zeros((oh, ow))
	if denom.max() > 0:
		m = denom > np.finfo(np.float64).eps * denom.max()
		cc[m] = num[m] / denom[m]
		np.clip(cc, -1, 1, out=cc)
	return cc


def _entropy_u16(img):
	"""MATLAB entropy() equivalent for uint16 images (65536-bin hist)."""
	h, _ = np.histogram(img, bins=65536, range=(0, 65536))
	p = h[h > 0] / img.size
	return float(-(p * np.log2(p)).sum())


def _poly_eval(X, x, y, Ntran):
	"""(dest_x, dest_y) from coefficient vector X (legacy layout)."""
	if Ntran == 0:
		return (
			X[0] * x + X[1] * y + X[2] * x * y + X[3],
			X[4] * x + X[5] * y + X[6] * x * y + X[7],
		)
	half = sum(range(1, Ntran + 2))
	vx = np.zeros_like(np.asarray(x, float))
	vy = np.zeros_like(vx)
	pnt = 0
	for i in range(Ntran + 1):
		for j in range(Ntran + 1 - i):
			b = (np.asarray(x, float) ** i) * (np.asarray(y, float) ** j)
			vx = vx + X[pnt] * b
			vy = vy + X[pnt + half] * b
			pnt += 1
	return vx, vy


def init_pass(tiles, tile_corners, strip):
	"""Coarse POC placement of `strip` among overlapping reference tiles."""
	sp = poc(strip)
	m2s, n2s = sp.shape
	table = []
	for p, (tile, corner) in enumerate(zip(tiles, tile_corners)):
		tp_ = poc(tile)
		if tp_.shape[0] < m2s or tp_.shape[1] < n2s or tile.max() == tile.min():
			continue
		cc = match_template(tp_, sp)
		noise_floor = 4.0 * cc.std()
		pk = cc.max()
		if pk > noise_floor:
			ypk, xpk = np.unravel_index(cc.argmax(), cc.shape)
			table.append((p, pk, corner[0] + ypk, corner[1] + xpk))
	if not table:
		return None, True
	return max(table, key=lambda r: r[1]), False


def total_features(img, j, samp_ratio=3, morepts=0):
	"""Multiscale wavelet feature points at scale j."""
	img = img.astype(np.float64)
	a = 0.8
	k = (1 / (a + 1)) * np.array([[-a, a - 1, -a], [a - 1, a + 5, a - 1], [-a, a - 1, -a]])
	img = correlate(img, k, mode="constant")
	img[img < 0] = 0
	L = 2**j
	hp = np.array([-1.0] * j + [1.0] * j)
	# These kernels have even length. imfilter centres a length-L kernel at
	# element L/2 (1-based), scipy at index L//2 (0-based), one sample later,
	# so origin=-1 on the even axis. Without it every feature is one pixel
	# down and right of MATLAB's.
	LPx = correlate(img, np.ones((1, L)) / L, mode="constant", origin=(0, -1))
	LH = correlate(LPx, hp.reshape(-1, 1), mode="constant", origin=(-1, 0))
	HPy = correlate(img, hp.reshape(1, -1), mode="constant", origin=(0, -1))
	HL = correlate(HPy, np.ones((L, 1)) / L, mode="constant", origin=(-1, 0))
	M = np.sqrt(LH**2 + HL**2)

	F0 = local_maxima(M, connectivity=2).astype(np.float64)
	m0, n0 = img.shape
	hw = L + 1
	F0[:hw, :] = 0
	F0[:, :hw] = 0
	F0[m0 - hw :, :] = 0
	F0[:, n0 - hw :] = 0

	# Column-major, like MATLAB's find(). The suppression below edits F0 as it
	# goes, so the visiting order changes which points survive.
	xs, ys = np.nonzero(F0.T == 1)
	w = L if morepts == 0 else L // 2
	for r, c in zip(ys, xs):
		win = (slice(r - w, r + w + 1), slice(c - w, c + w + 1))
		tmp = F0[win] * M[win]
		mx = tmp.max()
		if mx > 0:
			idx = np.argmax(tmp.T)
			yp, xp = idx % tmp.shape[0], idx // tmp.shape[0]
			msk = np.zeros_like(tmp)
			msk[yp, xp] = 1
			F0[win] = F0[win] * msk

	xs, ys = np.nonzero(F0.T == 1)
	F = np.zeros((len(ys), 3))
	d = (samp_ratio // 2) / samp_ratio
	for n, (r, c) in enumerate(zip(ys, xs)):
		patch = M[r - 1 : r + 2, c - 1 : c + 2]
		big = imresize_bicubic(patch, scale=samp_ratio)
		axn = (c - 1 - d) + np.arange(3 * samp_ratio) / samp_ratio
		axm = (r - 1 - d) + np.arange(3 * samp_ratio) / samp_ratio
		pr, pc = np.nonzero(big == big.max())
		i = 0
		if len(pr) > 1:
			i = int(np.argmax((axn[pc] - c) ** 2 + (axm[pr] - r) ** 2))
		F[n] = [axn[pc[i]], axm[pr[i]], j]
	return F[(F[:, 0] != 0) & (F[:, 1] != 0) & (F[:, 2] != 0), :]


def _mldivide(A, b):
	"""MATLAB A\\b for a tall A: pivoted QR, basic solution.

	MATLAB fits these polynomials with lsqlin and no constraints, which it
	hands to mldivide. The cubic fits on coordinates in the thousands are
	badly conditioned, so the solver matters: numpy's SVD lstsq left strip 1
	of panel C with 259 pairs instead of MATLAB's 245. This reproduces all
	245 exactly. Rank uses MATLAB's tolerance, max(m, n) * eps * |R[0, 0]|.
	"""
	Q, R, P = qr(A, mode="economic", pivoting=True)
	tol = max(A.shape) * np.finfo(float).eps * abs(R[0, 0])
	r = int((np.abs(np.diag(R)) > tol).sum())
	x = np.zeros(A.shape[1])
	x[P[:r]] = solve_triangular(R[:r, :r], (Q.T @ b)[:r])
	return x


def xy_disp(F1, F2, mxd, minpts, Npoly=0):
	"""Iterative outlier rejection."""
	F1 = np.asarray(F1, float).copy()
	F2 = np.asarray(F2, float).copy()
	xdist = F2[:, 0] - F1[:, 0]
	ydist = F2[:, 1] - F1[:, 1]
	mxx = mxy = np.inf
	mf1 = len(F1)
	while (mxx > 0 or mxy > 0) and mf1 >= minpts:
		if Npoly == 0:
			A = np.column_stack([F1[:, 0], F1[:, 1], F1[:, 0] * F1[:, 1], np.ones(mf1)])
		else:
			cols = []
			for i in range(Npoly + 1):
				for j in range(Npoly + 1 - i):
					cols.append((F1[:, 0] ** i) * (F1[:, 1] ** j))
			A = np.column_stack(cols)
		if Npoly == 0:
			xx = np.linalg.lstsq(A, xdist, rcond=None)[0]
			xy = np.linalg.lstsq(A, ydist, rcond=None)[0]
		else:
			xx = _mldivide(A, xdist)
			xy = _mldivide(A, ydist)
		ex = np.abs(xdist - A @ xx) - mxd
		ex[ex < 0] = 0
		ey = np.abs(ydist - A @ xy) - mxd
		ey[ey < 0] = 0
		mxx, indx = ex.max(), int(ex.argmax())
		mxy, indy = ey.max(), int(ey.argmax())
		ind = indx if mxx >= mxy else indy
		F1 = np.delete(F1, ind, 0)
		F2 = np.delete(F2, ind, 0)
		xdist = np.delete(xdist, ind)
		ydist = np.delete(ydist, ind)
		mf1 = len(F1)
	return F1, F2, max(mxx, mxy), mf1


def mapping(F11, F21, Ntran=0):
	"""Least-squares polynomial transform: dest = poly(src)."""
	F11 = np.asarray(F11, float)
	F21 = np.asarray(F21, float)
	m2 = len(F21)
	x, y = F21[:, 0], F21[:, 1]
	Y = np.concatenate([F11[:, 0], F11[:, 1]])
	if Ntran == 0:
		A = np.zeros((2 * m2, 8))
		A[:m2, 0] = x
		A[:m2, 1] = y
		A[:m2, 2] = x * y
		A[:m2, 3] = 1
		A[m2:, 4] = x
		A[m2:, 5] = y
		A[m2:, 6] = x * y
		A[m2:, 7] = 1
		return np.linalg.pinv(A) @ Y
	half = sum(range(1, Ntran + 2))
	A = np.zeros((2 * m2, 2 * half))
	pnt = 0
	for i in range(Ntran + 1):
		for j in range(Ntran + 1 - i):
			b = (x**i) * (y**j)
			A[:m2, pnt] = b
			A[m2:, pnt + half] = b
			pnt += 1
	return _mldivide(A, Y)


def apply_mapping(I2, init_reg, X, Ntran, mfull, nfull, extra=()):
	"""Forward-warp I2 and optional extra arrays onto destination canvas."""
	m0, n0 = I2.shape
	x0, y0 = np.meshgrid(np.arange(n0, dtype=float), np.arange(m0, dtype=float))
	x0 += init_reg[1]
	y0 += init_reg[0]
	v2, u2 = _poly_eval(X, x0, y0, Ntran)
	m21 = max(int(np.ceil(u2.min())), 0)
	m22 = min(int(np.floor(u2.max())), mfull - 1)
	n21 = max(int(np.ceil(v2.min())), 0)
	n22 = min(int(np.floor(v2.max())), nfull - 1)
	pts = np.column_stack([v2.ravel(), u2.ravel()])
	tri = Delaunay(pts)
	V1, U1 = np.meshgrid(np.arange(n21, n22 + 1), np.arange(m21, m22 + 1))
	# natural-neighbour weights, like MATLAB's 'natural'; rows outside the hull are empty, so 0
	W, _ = sibson_matrix(tri, np.column_stack([V1.ravel(), U1.ravel()]).astype(float))
	outs = [(W @ np.asarray(arr, float).ravel()).reshape(V1.shape) for arr in (I2,) + tuple(extra)]
	return outs[0], m21, n21, outs[1:]


def mosaic1(img2_reg, zd, zd1, blknum, blk_id, img20, m21, n21):
	"""Phase-3 mosaic: nearest-centre rule (zd1 < zd0 claims pixel)."""
	m0, n0 = img20.shape
	sl = (slice(m21, m21 + m0), slice(n21, n21 + n0))
	msk = zd1 < zd[sl]
	img2_reg[sl] = np.where(msk, img20, img2_reg[sl])
	zd[sl] = np.where(msk, zd1, zd[sl])
	blknum[sl] = np.where(msk, blk_id, blknum[sl])
	return img2_reg, zd, blknum


def art_regx(image0bin_ch, img1, init_reg, cfg, m2, n2):
	"""Fine registration engine used by register_reference."""
	img11 = img1.astype(np.float64)
	if cfg["filteron"]:
		img11 = median_filter(img11, size=3, mode="constant")
	m1, n1 = img11.shape
	F1 = np.vstack(
		[total_features(img11, j, cfg["samp_ratio"], cfg["morepts"]) for j in range(cfg["j01"], cfg["j02"] + 1)]
	)
	if len(F1) == 0:
		return None, np.inf, 0
	src_poc = poc(img1)
	emb = np.zeros((m2, n2))
	emb[init_reg[0] : init_reg[0] + m1, init_reg[1] : init_reg[1] + n1] = src_poc
	F1[:, 0] += init_reg[1]
	F1[:, 1] += init_reg[0]

	samp = cfg["samp_ratio"]
	max_shift = cfg["max_shift"]
	d = (samp // 2) / samp
	keep1, keep2 = [], []
	for i in range(len(F1)):
		fx, fy, j = F1[i, 0], F1[i, 1], int(F1[i, 2])
		if fx < 0 or fy < 0 or fx > n2 - 1 or fy > m2 - 1:
			continue
		h = 2**j - 1
		lt = int(np.floor(fx)) - h
		rt = int(np.ceil(fx)) + h
		tp = int(np.floor(fy)) - h
		bt = int(np.ceil(fy)) + h
		if lt < 0 or tp < 0 or rt > n2 - 1 or bt > m2 - 1:
			continue
		tmp = emb[tp : bt + 1, lt : rt + 1]
		t1 = imresize_bicubic(tmp, scale=samp)
		axn = (lt - d) + np.arange(t1.shape[1]) / samp
		axm = (tp - d) + np.arange(t1.shape[0]) / samp
		x1 = int(np.argmin(np.abs(axn - fx)))
		y1 = int(np.argmin(np.abs(axm - fy)))
		dd = ((2 ** (j + 1) - 1) * samp - 1) // 2
		if y1 - dd < 0 or x1 - dd < 0 or y1 + dd >= t1.shape[0] or x1 + dd >= t1.shape[1]:
			continue
		t1 = t1[y1 - dd : y1 + dd + 1, x1 - dd : x1 + dd + 1]

		lt2 = lt - max_shift
		if lt2 < 0:
			lt2 = 0
		elif lt2 > n2 - 2 ** (j + 1):
			lt2 = n2 - 2 ** (j + 1)
		rt2 = min(rt + max_shift, n2 - 1)
		tp2 = tp - max_shift
		if tp2 < 0:
			tp2 = 0
		elif tp2 > m2 - 2 ** (j + 1):
			tp2 = m2 - 2 ** (j + 1)
		bt2 = min(bt + max_shift, m2 - 1)
		ref = image0bin_ch[tp2 : bt2 + 1, lt2 : rt2 + 1]
		t2 = imresize_bicubic(ref, scale=samp)
		ax2n = (lt2 - d) + np.arange(t2.shape[1]) / samp
		ax2m = (tp2 - d) + np.arange(t2.shape[0]) / samp

		mt1, nt1 = t1.shape
		mt2, nt2 = t2.shape
		cc = np.abs(_normxcorr2_full(t1, t2))
		if cfg["addsharp"] != 1:
			sig = cc.std()
			noise_floor = 4 * sig
		else:
			sig = 0.0
			noise_floor = 0.0
		cc = cc * local_maxima(cc, connectivity=2)
		cc = cc[mt1 // 2 :, nt1 // 2 :]
		max_cc = cc.max()
		ypk, xpk = np.unravel_index(np.argmax(cc), cc.shape)
		cnt = int((cc > (max_cc - sig)).sum()) if cfg["addsharp"] != 1 else 1
		if xpk < nt2 and ypk < mt2 and cnt == 1 and max_cc > noise_floor:
			keep1.append([fx, fy])
			keep2.append([ax2n[xpk], ax2m[ypk]])
	if len(keep1) == 0:
		return None, np.inf, 0
	F1c, F2c, mxd1, mf1 = xy_disp(np.array(keep1), np.array(keep2), cfg["mxd"], cfg["minpts"], cfg["Npoly"])
	X = None
	if mxd1 == 0:
		X = mapping(F2c, F1c, cfg["Ntran"])
	return X, mxd1, mf1, F1c, F2c


def incremental_registration(image0bin_ch, img2_reg, zd, blknum, blk_id, img21, init_reg, cfg, m2, n2, verbose=True):
	"""Run art_regx then merge the warped strip into the current mosaic."""
	X, mxd1, mf1, F1c, F2c = art_regx(image0bin_ch, img21, init_reg, cfg, m2, n2)
	if verbose:
		print(f"    strip {blk_id-1}: disparity={mxd1:.3g} pairs={mf1} " f"{'PASS' if mxd1 == 0 else 'FAIL'}")
	if mxd1 != 0:
		return None, 0, img2_reg, zd, blknum, mxd1, mf1, F1c, F2c
	mi, ni = img21.shape
	v1, u1 = np.meshgrid(np.arange(ni, dtype=float), np.arange(mi, dtype=float))
	zd0 = np.sqrt((u1 - (mi - 1) / 2) ** 2 + (v1 - (ni - 1) / 2) ** 2)
	img20, m21, n21, (mskw, zdw) = apply_mapping(
		img21, init_reg, X, cfg["Ntran"], m2, n2, extra=(np.full((mi, ni), 2.0**15), zd0)
	)
	msk = np.round(mskw) == 2**15
	zdw[~msk] = np.sqrt(mi**2 + ni**2)
	zd1 = np.floor(zdw + 0.5).astype(np.uint16)  # MATLAB uint16() rounds, astype truncates
	img2_reg, zd, blknum = mosaic1(img2_reg, zd, zd1, blknum, blk_id, img20, m21, n21)
	return X, 1, img2_reg, zd, blknum, mxd1, mf1, F1c, F2c


def build_reference(ref_arr, sc, m0, n0, down, seed=0, bitdepth=16):
	"""Build padded reference canvases used by the registration pipeline.

	MATLAB holds this canvas in the reference image's integer class (uint8 or
	uint16 depending on bd1) rather than in floating point, so everything
	downstream sees quantised data. `bitdepth` selects that class; it is read
	from the source file in the MATLAB and has to be supplied here because the
	array arrives as float.

	The two resizes pass a scalar scale rather than an output shape, matching
	`imresize(img, sc)` and `imresize(image0tif, down)`. That distinction
	matters: MATLAB rounds the output size up but keeps using the requested
	scale for the interpolation kernel.
	"""
	dtype = np.uint8 if bitdepth == 8 else np.uint16
	if ref_arr.ndim == 2:
		ref_arr = ref_arr[:, :, None]
	if sc != 1.0:
		ref_arr = np.stack(
			[imresize_bicubic(ref_arr[:, :, i], scale=sc, dtype=dtype) for i in range(ref_arr.shape[2])],
			axis=2,
		)
	m1, n1, p1 = ref_arr.shape
	m1x, n1x = m1 + 2 * (m0 - 1), n1 + 2 * (n0 - 1)
	# `seed` is the MATLAB rng seed. MATLAB's default generator is MT19937,
	# and rng(0) (i.e. rng('default')) initialises it with 5489, so legacy
	# RandomState reproduces its stream. rand(m,n,p) fills the whole array
	# at once in column-major order, and uint16() rounds rather than
	# truncates. With all three, the padding matches Martin's seeded
	# image0.tif pixel for pixel.
	rs = np.random.RandomState(5489 if seed == 0 else seed)
	noise = rs.random_sample(m1x * n1x * p1).reshape((m1x, n1x, p1), order="F")
	image0tif = np.empty((m1x, n1x, p1), dtype)
	info = np.iinfo(dtype)
	for i in range(p1):
		pad = np.floor(ref_arr[:, :, i].mean() * noise[:, :, i] + 0.5)
		image0tif[:, :, i] = np.clip(pad, info.min, info.max).astype(dtype)
	image0tif[m0 - 1 : m0 - 1 + m1, n0 - 1 : n0 - 1 + n1, :] = np.clip(
		np.rint(ref_arr), info.min, info.max
	).astype(dtype)
	image0bin = np.stack([poc(image0tif[:, :, i]) for i in range(p1)], axis=2)
	image1tif = np.stack(
		[imresize_bicubic(image0tif[:, :, i], scale=down, dtype=dtype) for i in range(p1)],
		axis=2,
	).astype(float)
	return dict(image0bin=image0bin, image1tif=image1tif, m1=m1, n1=n1, p1=p1, m1x=m1x, n1x=n1x)


def make_tiles(plane, m00, n00):
	"""Overlapping tile stack from one downsampled-reference channel."""
	m10, n10 = plane.shape
	tiles, corners = [], []
	r = 0
	stopm = 0
	while stopm < 2:
		r2 = min(r + 2 * m00, m10)
		if r2 == m10:
			stopm += 1
		c = 0
		stopn = 0
		while stopn < 2:
			c2 = min(c + 2 * n00, n10)
			if c2 == n10:
				stopn += 1
			t = np.zeros((2 * m00, 2 * n00))
			t[: r2 - r, : c2 - c] = plane[r:r2, c:c2]
			tiles.append(t)
			corners.append((r, c))
			c = c2 - n00 + 1
			if c + 2 * n00 >= n10:
				stopn += 1
		r = r2 - m00 + 1
		if r + 2 * m00 >= m10:
			stopm += 1
	return tiles, corners



class Registration:
	"""Stateful Phase-3 registration pipeline for one reference and strip set."""

	def __init__(self, strips, ref_arr, rel_reg, cfg, verbose=True):
		if isinstance(strips, np.ndarray) and strips.ndim == 2:
			strips = [strips]
		self.strips = [np.asarray(strip, np.float64) for strip in strips]
		# MATLAB reads bd1 from the reference file; pass the array as loaded
		# (uint8 or uint16) so the canvas is quantised to the same class.
		self.bitdepth = 8 if np.asarray(ref_arr).dtype == np.uint8 else 16
		self.ref_arr = np.asarray(ref_arr, np.float64)
		self.rel_reg = rel_reg
		self.cfg = cfg
		self.verbose = verbose
		self.anchor = None
		self.image0bin = self.image1tif = None
		self.m1x = self.n1x = self.p1 = None
		self.m10 = self.n10 = self.m00 = self.n00 = None
		self.m0 = self.n0 = None
		self.Xstar00 = self.init_reg00 = None
		self.img2_reg = self.zd = self.blknum = None
		self.results = self.F1c_all = self.F2c_all = self.IX = None
		self.dstrips = None
		self.ref = None
		self._sequential_state = None

	def prepare(self):
		"""Build and retain the reference context used by all registration steps."""
		if not self.strips:
			raise ValueError("Registration requires at least one strip")
		self.m0, self.n0 = self.strips[0].shape
		down = self.cfg["down"]
		ncoef = 8 if self.cfg["Ntran"] == 0 else 2 * sum(range(1, self.cfg["Ntran"] + 2))
		lo = min(float(strip.min()) for strip in self.strips)
		hi = max(float(strip.max()) for strip in self.strips)
		u16 = [
			np.clip((strip - lo) / max(hi - lo, 1e-12) * 65535, 0, 65535).astype(np.uint16)
			for strip in self.strips
		]
		self.IX = np.argsort([_entropy_u16(strip) for strip in u16])[::-1]
		self.ref = build_reference(self.ref_arr, self.cfg["sc"], self.m0, self.n0, down, bitdepth=self.bitdepth)
		self.image0bin = self.ref["image0bin"]
		self.image1tif = self.ref["image1tif"]
		self.m1x, self.n1x, self.p1 = self.ref["m1x"], self.ref["n1x"], self.ref["p1"]
		self.m10, self.n10 = self.image1tif.shape[:2]
		self.m00, self.n00 = int(np.ceil(self.m0 * down)), int(np.ceil(self.n0 * down))
		# MATLAB: img2 = imresize(img21, down) on the uint16 strip
		self.dstrips = [
			imresize_bicubic(strip, scale=down, dtype=np.uint16).astype(float)
			for strip in self.strips
		]
		count = len(self.strips)
		self.img2_reg = np.zeros((self.m1x, self.n1x))
		self.zd = np.full((self.m1x, self.n1x), 2**16 - 1, np.uint16)
		self.blknum = np.zeros((self.m1x, self.n1x), np.uint16)
		self.Xstar00 = np.zeros((count, ncoef))
		self.init_reg00 = np.zeros((count, 2), int)
		self.results = np.zeros((count, 5))
		self.F1c_all = np.empty(count, dtype=object)
		self.F2c_all = np.empty(count, dtype=object)
		return self

	def art_regx(self, image, init_reg, channel=0):
		"""Fine-register ``image`` using this instance's reference canvas."""
		if self.image0bin is None:
			raise RuntimeError("call prepare before art_regx")
		return art_regx(
			self.image0bin[:, :, channel], image, init_reg, self.cfg, self.m1x, self.n1x
		)

	def incremental_registration(self, strip_index, init_reg, channel=0, cfg=None):
		"""Fine-register and merge one strip into this instance's mosaic state."""
		if self.img2_reg is None:
			raise RuntimeError("call prepare before incremental_registration")
		cfg = self.cfg if cfg is None else cfg
		out = incremental_registration(
			self.image0bin[:, :, channel], self.img2_reg, self.zd, self.blknum,
			strip_index + 1, self.strips[strip_index], init_reg, cfg,
			self.m1x, self.n1x, self.verbose,
		)
		X, passed, self.img2_reg, self.zd, self.blknum, mxd, pairs, F1c, F2c = out
		return X, passed, mxd, pairs, F1c, F2c

	def _upscale_corner(self, corner):
		corner = np.asarray(corner, float) + 1
		corner = np.round(corner / self.cfg["down"] - (1 / (2 * self.cfg["down"]) + 1)) - 1
		corner = np.maximum(corner.astype(int), 0)
		return np.minimum(corner, [self.m1x - self.m0, self.n1x - self.n0])

	def find_anchor(self):
		"""Register the first strip that can be placed on the reference."""
		for index in self.IX:
			for channel in range(self.p1):
				tiles, corners = make_tiles(self.image1tif[:, :, channel], self.m00, self.n00)
				best, failed = init_pass(tiles, corners, self.dstrips[index])
				if failed:
					continue
				init_reg = self._upscale_corner(best[2:])
				if self.verbose:
					print(f"anchor try: strip {index} ch {channel} -> init_reg {init_reg}")
				X, passed, mxd, pairs, F1c, F2c = self.incremental_registration(index, init_reg, channel)
				if passed:
					self.anchor = int(index)
					self.Xstar00[index] = X
					self.init_reg00[index] = init_reg
					self.results[index] = [index, mxd, pairs, channel, self.cfg["morepts"]]
					self.F1c_all[index] = F1c
					self.F2c_all[index] = F2c
					return index
		return None

	def _register_strip(self, index, mult, dstrip, pos, Xstar):
		"""Register one strip relative to a previously registered strip."""
		if self.verbose:
			print(f"registering strip {index} with mult={mult} at pos={pos}")
		search_test = 0 if self.cfg["Ntran"] == 0 else 1
		repcnt = passed = 0
		mult0 = mult
		boost = 1 if self.cfg["morepts"] else 0
		channel = 0
		while passed == 0 and repcnt <= self.cfg["repcnt_max"]:
			ms = self.cfg["max_shift"] * (1 + repcnt)
			x0, y0 = pos[1], pos[0]
			x1, y1 = x0 + self.n0 - 1, y0 + self.m0 - 1
			vx0, vy0 = _poly_eval(Xstar, x0, y0, self.cfg["Ntran"])
			vx1, vy1 = _poly_eval(Xstar, x1, y1, self.cfg["Ntran"])
			reg_out = np.array([round(float(vy0)), round(float(vx0))])
			reg_out2 = np.array([round(float(vy1)), round(float(vx1))])
			mtmp1 = max(int(np.floor((reg_out[0] - self.m0 * (mult - 1) - ms * 2) * self.cfg["down"])), 0)
			ntmp1 = max(int(np.floor((reg_out[1] - self.n0 * (mult - 1) - ms * 2) * self.cfg["down"])), 0)
			mtmp2 = min(int(np.floor((reg_out2[0] + self.m0 * (mult - 1) + ms * 2) * self.cfg["down"])), self.m10 - 1)
			ntmp2 = min(int(np.floor((reg_out2[1] + self.n0 * (mult - 1) + ms * 2) * self.cfg["down"])), self.n10 - 1)
			if mtmp1 >= mtmp2:
				mtmp1 = max(mtmp2 - round(3 * self.m0 * self.cfg["down"]), 0)
			if ntmp1 >= ntmp2:
				ntmp1 = max(ntmp2 - round(3 * self.n0 * self.cfg["down"]), 0)
			failed = False
			if search_test == 0:
				init_reg = np.clip(reg_out, [0, 0], [self.m1x - self.m0, self.n1x - self.n0])
			else:
				window = self.image1tif[mtmp1 : mtmp2 + 1, ntmp1 : ntmp2 + 1, channel]
				best, failed = init_pass([window], [(0, 0)], dstrip)
				if not failed:
					init_reg = self._upscale_corner((best[2] + mtmp1, best[3] + ntmp1))
			if not failed:
				attempt_cfg = dict(self.cfg)
				attempt_cfg["morepts"] = boost
				X, passed, mxd, pairs, F1c, F2c = self.incremental_registration(index, init_reg, channel, attempt_cfg)
			else:
				mult = min(mult + 1, mult0 + 1)
				mxd, pairs = np.inf, 0
			if passed:
				self.init_reg00[index] = init_reg
				self.Xstar00[index] = X
				self.F1c_all[index] = F1c
				self.F2c_all[index] = F2c
				self.results[index] = [index, mxd, pairs, channel, boost]
				return 1
			if self.p1 > 1 and channel < self.p1 - 1:
				channel += 1
			elif search_test == 0:
				search_test, channel = 1, 0
			elif boost == 0:
				boost, channel = 1, 0
			else:
				repcnt += 1
				channel = 0
		self.results[index] = [index, np.inf, 0, channel, boost]
		return mult0 + 1

	def register_strips_sequentially(self):
		"""Register strips sequentially after anchor placement.

		Each strip is predicted from its immediate neighbour rather than from
		the anchor: rel_reg holds relative offsets, so the prediction has to
		accumulate as we walk outward. `previous` therefore advances to each
		strip that registers successfully. It stays put on failure, so the
		next strip is predicted from the last one that worked.
		"""
		previous, mult = self.anchor, 1
		for index in range(self.anchor - 1, -1, -1):
			Xstar = self.Xstar00[previous]
			pos = self.init_reg00[previous] - self.rel_reg[index + 1, 1:3]
			mult = self._register_strip(index, mult, self.dstrips[index], pos, Xstar)
			if mult == 1:
				previous = index
		previous, mult = self.anchor, 1
		for index in range(self.anchor + 1, len(self.strips)):
			Xstar = self.Xstar00[previous]
			pos = self.init_reg00[previous] + self.rel_reg[index, 1:3]
			mult = self._register_strip(index, mult, self.dstrips[index], pos, Xstar)
			if mult == 1:
				previous = index
		passed = [index for index in range(len(self.strips)) if not np.all(self.Xstar00[index] == 0)]
		previous, mult = min(passed), 1
		for index in range(min(passed) + 1, self.anchor):
			if index in passed:
				previous, mult = index, 1
			else:
				Xstar = self.Xstar00[previous]
				pos = self.init_reg00[previous] + self.rel_reg[index, 1:3]
				mult = self._register_strip(index, mult, self.dstrips[index], pos, Xstar)
				if mult == 1:
					previous = index
		previous, mult = max(passed), 1
		for index in range(max(passed) - 1, self.anchor, -1):
			if index in passed:
				previous, mult = index, 1
			else:
				Xstar = self.Xstar00[previous]
				pos = self.init_reg00[previous] - self.rel_reg[index + 1, 1:3]
				mult = self._register_strip(index, mult, self.dstrips[index], pos, Xstar)
				if mult == 1:
					previous = index

	def begin_sequential_anchor_registration(self):
		"""Initialize one-strip-at-a-time sequential registration based on an anchor."""
		if self.anchor is None:
			raise RuntimeError("begin_sequential_anchor_registration: no anchored strip")
		anchor = int(self.anchor)
		self._sequential_state = dict(phase=0, index=anchor + 1, previous=anchor, mult=1)
		return self

	def step_sequential_anchor_registration(self):
		"""Register or skip exactly one strip and return its progress status."""
		if self._sequential_state is None:
			self.begin_sequential_anchor_registration()
		state = self._sequential_state
		while True:
			phase = state["phase"]
			index = state["index"]
			if phase == 0 and index >= len(self.strips):
				state.update(phase=1, index=self.anchor - 1, previous=self.anchor, mult=1)
				continue
			if phase == 1 and index < 0:
				passed = {int(value) for value in np.flatnonzero(np.any(self.Xstar00 != 0, axis=1))}
				state.update(phase=2, index=min(passed) + 1, previous=min(passed), mult=1, passed=passed)
				continue
			if phase == 2 and index >= self.anchor:
				passed = state["passed"]
				state.update(phase=3, index=max(passed) - 1, previous=max(passed), mult=1)
				continue
			if phase == 3 and index <= self.anchor:
				self._sequential_state = None
				return dict(done=True, index=None, phase="complete", passed=int(np.count_nonzero(np.any(self.Xstar00 != 0, axis=1))))

			registered = False
			if phase == 0:
				pos = self.init_reg00[state["previous"]] + self.rel_reg[index, 1:3]
				state["mult"] = self._register_strip(index, state["mult"], self.dstrips[index], pos, self.Xstar00[state["previous"]])
				state["index"] += 1
				registered = True
			elif phase == 1:
				pos = self.init_reg00[state["previous"]] - self.rel_reg[index + 1, 1:3]
				state["mult"] = self._register_strip(index, state["mult"], self.dstrips[index], pos, self.Xstar00[state["previous"]])
				state["index"] -= 1
				registered = True
			elif phase == 2:
				if index in state["passed"]:
					state.update(previous=index, mult=1)
					state["index"] += 1
					continue
				else:
					pos = self.init_reg00[state["previous"]] + self.rel_reg[index, 1:3]
					state["mult"] = self._register_strip(index, state["mult"], self.dstrips[index], pos, self.Xstar00[state["previous"]])
					registered = True
				state["index"] += 1
			else:
				if index in state["passed"]:
					state.update(previous=index, mult=1)
					state["index"] -= 1
					continue
				else:
					pos = self.init_reg00[state["previous"]] - self.rel_reg[index + 1, 1:3]
					state["mult"] = self._register_strip(index, state["mult"], self.dstrips[index], pos, self.Xstar00[state["previous"]])
					registered = True
				state["index"] -= 1
			return dict(done=False, index=int(index), phase=int(phase), registered=registered, result=self.results[index].copy())

	def begin_sequential_free_registration(self):
		"""Initialize one-strip-at-a-time sequential registration without anchor."""
		self._sequential_state = dict(phase=0, index=0, mult=1)
		self.initialize_positions_and_transforms(init_F_mode="unit")
		return self

	def step_sequential_free_registration(self):
		"""Register or skip exactly one strip and return its progress status."""
		if self._sequential_state is None:
			self.begin_sequential_free_registration()
		state = self._sequential_state
		while True:
			phase = state["phase"]
			index = state["index"]
			if phase == 0 and index >= len(self.strips):
				self._sequential_state = None
				return dict(done=True, index=None, phase="complete", passed=int(np.count_nonzero(np.any(self.Xstar00 != 0, axis=1))))

			registered = False
			if phase == 0:
				pos = self.init_reg00[index]
				state["mult"] = self._register_strip(index, state["mult"], self.dstrips[index], pos, self.Xstar00[index])
				state["index"] += 1
				registered = True
			return dict(done=False, index=int(index), phase=int(phase), registered=registered, result=self.results[index].copy())

	def _register_strip_independently(self, index):
		"""Register one strip through `_register_strip` on isolated mosaic state."""
		worker = copy(self)
		worker.img2_reg = np.zeros_like(self.img2_reg)
		worker.zd = np.full_like(self.zd, 2**16 - 1)
		worker.blknum = np.zeros_like(self.blknum)
		worker.Xstar00 = self.Xstar00.copy()
		worker.init_reg00 = self.init_reg00.copy()
		worker.results = self.results.copy()
		worker.F1c_all = self.F1c_all.copy()
		worker.F2c_all = self.F2c_all.copy()
		worker._register_strip(
			index, 1, self.dstrips[index], self.init_reg00[index], self.Xstar00[index]
		)
		return index, worker

	def initialize_positions_and_transforms(self, init_F_mode="unit"):
		"""Prepare initial positions and transforms for every strip."""
		if self.anchor is not None:
			# Initialize positions for all strips based on the anchor and relative registration offsets.
			for index in range(self.anchor + 1, len(self.strips)):
				self.init_reg00[index] = self.init_reg00[index - 1] + self.rel_reg[index, 1:3]
			for index in range(self.anchor - 1, -1, -1):
				self.init_reg00[index] = self.init_reg00[index + 1] - self.rel_reg[index + 1, 1:3]
		else:
			# Initialize positions relative to the first strip
			self.init_reg00[0] = self.rel_reg[0, 1:3]
			for index in range(1, len(self.strips)):
				self.init_reg00[index] = self.init_reg00[index - 1] + self.rel_reg[index, 1:3]
		# initialize transforms for all other strips
		if self.cfg["Ntran"] == 0:
			unit_coeff = [0, 5] # unit coefficients, all others zero
			zero_coeff = [3, 7] # start with given X, set these to zero to disable translation
		else:
			half_ncoef = sum(range(1, self.cfg["Ntran"] + 2))
			unit_coeff = [4, half_ncoef + 1] # unit coefficients, all others zero
			zero_coeff = [0, half_ncoef] # start with given X, set these to zero to disable translation
		if self.anchor is not None:
			for index in range(len(self.strips)):
				if index == self.anchor:
					continue
				# set unit coefficients to 1 for all non-anchor strips (was initialized to all zeros)
				if init_F_mode == "unit":
					self.Xstar00[index][unit_coeff] = 1
				# copy anchor's transform, but set translation to zero
				else:
					self.Xstar00[index] = self.Xstar00[self.anchor]
					self.Xstar00[index][zero_coeff] = 0
		else: 
			for index in range(len(self.strips)):
				if init_F_mode == "unit":
					self.Xstar00[index][unit_coeff] = 1
				else:
					raise ValueError("Cannot initialize transforms relative to anchor when no anchor is set.")
		return self

	def register_strips_parallel(self, max_workers=None):
		"""Register initialized strips concurrently, then merge their results."""
		self.initialize_positions_and_transforms()
		indices = [index for index in range(len(self.strips)) if index != self.anchor]
		with ThreadPoolExecutor(max_workers=max_workers) as executor:
			candidates = list(executor.map(self._register_strip_independently, indices))
		for index, worker in candidates:
			self.init_reg00[index] = worker.init_reg00[index]
			self.Xstar00[index] = worker.Xstar00[index]
			self.results[index] = worker.results[index]
			self.F1c_all[index] = worker.F1c_all[index]
			self.F2c_all[index] = worker.F2c_all[index]
			mask = (worker.blknum == index + 1) & (worker.zd < self.zd)
			self.img2_reg[mask] = worker.img2_reg[mask]
			self.zd[mask] = worker.zd[mask]
			self.blknum[mask] = worker.blknum[mask]
		return self

	def register_reference(self, mode="sequential"):
		"""Register all strips and retain the resulting registration state."""
		self.prepare()
		if self.find_anchor() is None:
			raise RuntimeError("register_reference: no strip could be anchored")
		if mode == "parallel":
			self.register_strips_parallel()
		else:
			self.register_strips_sequentially()
		self.crop = (slice(self.m0 - 1, self.m0 - 1 + self.ref["m1"]), slice(self.n0 - 1, self.n0 - 1 + self.ref["n1"]))
		return dict(Xstar=self.Xstar00, init_reg=self.init_reg00, img2_reg=self.img2_reg, zd=self.zd,
			blknum=self.blknum, canvas_crop=self.crop, results=self.results, anchor=self.anchor,
			F1c=self.F1c_all, F2c=self.F2c_all)


def register_reference(strips, ref_arr, rel_reg, cfg, verbose=True, mode="sequential"):
	"""Compatibility wrapper for :class:`Registration`."""
	return Registration(strips, ref_arr, rel_reg, cfg, verbose).register_reference(mode)
