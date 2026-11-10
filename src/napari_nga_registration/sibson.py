"""Sibson (natural-neighbour) weights as a sparse matrix, + brute-force check.

For query p, the cavity is every Delaunay triangle whose circumcircle holds p.
Area p steals from vertex a = sum over cavity triangles (a,b,c) of the signed
area of (C_abc, g(p,a,b), g(p,c,a)), g = circumcentre. Vectorised over queries;
candidates are the containing triangle plus its K-ring of neighbours.
"""
import numpy as np
from scipy import sparse
from scipy.spatial import Delaunay


def _cc(a, b, c):
	"""Circumcentres of triangles (a, b, c), arrays (..., 2)."""
	bx, by = b[..., 0] - a[..., 0], b[..., 1] - a[..., 1]
	cx, cy = c[..., 0] - a[..., 0], c[..., 1] - a[..., 1]
	d = 2 * (bx * cy - by * cx)
	b2, c2 = bx * bx + by * by, cx * cx + cy * cy
	with np.errstate(divide="ignore", invalid="ignore"):  # collinear -> inf/nan, caught in sibson_matrix
		return np.stack([a[..., 0] + (cy * b2 - by * c2) / d, a[..., 1] + (bx * c2 - cx * b2) / d], -1)


def _area(p, q, r):
	return 0.5 * ((q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0]))


def _rings(tri, depth):
	"""Per triangle, itself plus neighbours up to `depth` steps, padded with -1."""
	nb = tri.neighbors
	cur = np.arange(len(nb))[:, None]
	for _ in range(depth):
		nxt = nb[np.where(cur < 0, 0, cur)].reshape(len(nb), -1)
		nxt[np.repeat(cur < 0, 3, axis=1)] = -1
		cur = np.concatenate([cur, nxt], 1)
		cur.sort(1)
		dup = np.zeros_like(cur, bool)
		dup[:, 1:] = cur[:, 1:] == cur[:, :-1]
		cur[dup] = -1
		cur.sort(1)
		cur = cur[:, (cur >= 0).any(0)]
	return cur


def _wedges(P, V, C, pp):
	"""Per cavity triangle (vertices V, circumcentre C) and query pp: area stolen per vertex."""
	A = P[V]
	return [_area(C, _cc(pp, A[:, i], A[:, (i + 1) % 3]), _cc(pp, A[:, (i + 2) % 3], A[:, i])) for i in range(3)]


def sibson_matrix(tri, q, depth=3, chunk=200_000):
	"""Sparse (len(q), npts) Sibson weights; rows outside the hull are empty.

	Cavities are searched within `depth` rings of the containing triangle,
	vectorised. Queries whose cavity reaches the outermost ring (thin
	hull-edge triangles have huge circumcircles) get an exact flood fill.
	"""
	P = tri.points
	ccs = _cc(P[tri.simplices[:, 0]], P[tri.simplices[:, 1]], P[tri.simplices[:, 2]])
	r2 = ((ccs - P[tri.simplices[:, 0]]) ** 2).sum(1)
	simp = tri.find_simplex(q)
	todo = np.nonzero(simp >= 0)[0]
	ring, inner = _rings(tri, depth), _rings(tri, depth - 1)
	rows, cols, vals, retry = [], [], [], []
	for s in range(0, len(todo), chunk):
		qi = todo[s:s + chunk]
		cand = ring[simp[qi]]                                 # (n, K)
		ok = cand >= 0
		cs = np.where(ok, cand, 0)
		inc = ok & (((q[qi][:, None, :] - ccs[cs]) ** 2).sum(-1) < r2[cs])
		bad = (inc & ~(cand[:, :, None] == inner[simp[qi]][:, None, :]).any(-1)).any(1)
		retry.append(qi[bad])
		inc[bad] = False
		n_i, k_i = np.nonzero(inc)
		t = cs[n_i, k_i]
		for i, w in enumerate(_wedges(P, tri.simplices[t], ccs[t], q[qi][n_i])):
			rows.append(qi[n_i]); cols.append(tri.simplices[t][:, i]); vals.append(w)
	retry = np.concatenate(retry)
	for k in retry:  # ponytail: per-query Python flood fill, only hull-edge stragglers hit it
		p, seen, stack, cav = q[k], {simp[k]}, [simp[k]], []
		while stack:
			t = stack.pop()
			cav.append(t)
			for u in tri.neighbors[t]:
				if u >= 0 and u not in seen:
					seen.add(u)
					if ((p - ccs[u]) ** 2).sum() < r2[u]:
						stack.append(u)
		t = np.array(cav)
		for i, w in enumerate(_wedges(P, tri.simplices[t], ccs[t], np.broadcast_to(p, (len(t), 2)))):
			rows.append(np.full(len(t), k)); cols.append(tri.simplices[t][:, i]); vals.append(w)
	rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
	tot = np.bincount(rows, vals, minlength=len(q))
	# A query sitting exactly on a data point (or on an edge of cocircular points) has an empty or
	# 0/0 cavity. Its Sibson limit is the barycentric weight, so use that rather than leave a hole.
	bad = np.nonzero((simp >= 0) & ~(np.isfinite(tot) & (tot != 0)))[0]
	keep = ~np.isin(rows, bad)
	T = tri.transform[simp[bad]]
	b = np.einsum("ijk,ik->ij", T[:, :2], q[bad] - T[:, 2])
	rows = np.concatenate([rows[keep], np.repeat(bad, 3)])
	cols = np.concatenate([cols[keep], tri.simplices[simp[bad]].ravel()])
	tot[bad] = 1
	vals = np.concatenate([vals[keep], np.column_stack([b, 1 - b.sum(1)]).ravel()]) / np.where(tot != 0, tot, 1)[rows]
	return sparse.csr_matrix((vals, (rows, cols)), shape=(len(q), len(P))), len(retry)


def _brute(P, p):
	"""Sibson weights by Voronoi area difference (shapely-free polygon areas)."""
	from scipy.spatial import Voronoi

	def areas(pts):
		v = Voronoi(pts)
		out = np.full(len(pts), np.nan)
		for k, r in enumerate(v.point_region):
			reg = v.regions[r]
			if reg and -1 not in reg:
				xy = v.vertices[reg]
				xy = xy[np.argsort(np.arctan2(*(xy - xy.mean(0)).T[::-1]))]
				out[k] = 0.5 * abs(np.dot(xy[:, 0], np.roll(xy[:, 1], -1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], -1)))
		return out
	a0, a1 = areas(P), areas(np.vstack([P, p]))
	w = np.nan_to_num(a0 - a1[:-1])
	return w / w.sum()


if __name__ == "__main__":
	rng = np.random.default_rng(0)
	# random scatter, framed so inner cells are finite
	inner = rng.random((300, 2))
	frame = np.array([[x, y] for x in np.linspace(-2, 3, 12) for y in (-2, 3)] + [[x, y] for y in np.linspace(-2, 3, 12) for x in (-2, 3)])
	P = np.vstack([inner, frame])
	tri = Delaunay(P)
	Q = 0.2 + 0.6 * rng.random((200, 2))
	W, _ = sibson_matrix(tri, Q, depth=6)
	err = max(np.abs(W[k].toarray().ravel() - _brute(P, Q[k])).max() for k in range(len(Q)))
	print("random scatter: max |ours - brute force Voronoi| =", f"{err:.1e}")
	f = 3 * P[:, 0] - 2 * P[:, 1] + 5
	print("  reproduces linear f exactly:", f"{np.abs(W @ f - (3 * Q[:, 0] - 2 * Q[:, 1] + 5)).max():.1e}")
	# near-regular warped grid, like a strip: depth 2 vs 4 must agree
	gx, gy = np.meshgrid(np.arange(60.0), np.arange(50.0))
	gx2 = 1.002 * gx + 0.01 * gy + 1e-4 * gx * gy + 0.3
	gy2 = -0.008 * gx + 0.998 * gy + 2e-5 * gx * gx
	G = np.column_stack([gx2.ravel(), gy2.ravel()])
	tri = Delaunay(G)
	Vq, Uq = np.meshgrid(np.arange(1, 58.0), np.arange(1, 48.0))
	Q = np.column_stack([Vq.ravel(), Uq.ravel()])
	W2, _ = sibson_matrix(tri, Q, depth=3)
	W4, _ = sibson_matrix(tri, Q, depth=5)
	Qr = 0.2 + 0.6 * np.random.default_rng(1).random((200, 2))
	Wd, _ = sibson_matrix(Delaunay(P), Qr, depth=2)
	print("random scatter from depth 2 (auto-deepen): max |diff| vs brute", f"{max(np.abs(Wd[k].toarray().ravel() - _brute(P, Qr[k])).max() for k in range(len(Qr))):.1e}")
	inside = tri.find_simplex(Q) >= 0
	print("warped grid: depth 3 vs 5 max diff", f"{abs(W2 - W4).max():.1e}", "| empty inside rows", int((np.asarray(W2.sum(1)).ravel()[inside] == 0).sum()))
	# brute force zeroes the unbounded hull cells, so only test well inside the grid
	interior = inside & (Q[:, 0] > 5) & (Q[:, 0] < 52) & (Q[:, 1] > 5) & (Q[:, 1] < 42)
	ks = rng.choice(np.nonzero(interior)[0], 200, replace=False)
	print("  vs brute force (200 interior pts):", f"{max(np.abs(W2[k].toarray().ravel() - _brute(G, Q[k])).max() for k in ks):.1e}")
	# exact grid, queries on the data points and on cell edges: no holes, data reproduced
	gx, gy = np.meshgrid(np.arange(20.0), np.arange(15.0))
	G = np.column_stack([gx.ravel(), gy.ravel()])
	f = rng.random(len(G))
	We, _ = sibson_matrix(Delaunay(G), G)
	assert np.abs(We @ f - f).max() < 1e-12, "on-vertex queries must return the data"
	Wm, _ = sibson_matrix(Delaunay(G), G[(gx.ravel() < 19)] + [0.5, 0])
	assert np.abs(np.asarray(Wm.sum(1)).ravel() - 1).max() < 1e-12 and np.isfinite(Wm.data).all(), "on-edge queries must be covered"
	print("exact grid: on-vertex and on-edge queries covered")
