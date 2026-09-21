"""Extract hand-drawn DR annotation curves from a screenshot and fit geometry parameters.

Usage:
    python scripts/calibrate_from_screenshot.py --hist <image>     # inspect hue histogram
    python scripts/calibrate_from_screenshot.py <image> [<image>]  # fit and report

Used to calibrate the default PD coefficients in labelmyeye/dr_geometry.py.
Annotations are expected as saturated thin strokes on a fundus photo:
red=optic, blue=macular, green=macular_area(ellipse), orange=posterior_pole_boundary,
purple=central_inner.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
from PIL import Image

# ---------------------------------------------------------------- color masks


def rgb_to_hsv_deg(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """arr: HxWx3 uint8 -> (H in [0,360), S in [0,1], V in [0,1]) float arrays."""
    rgb = arr.astype(np.float64) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    v = maxc
    delta = maxc - minc
    s = np.where(maxc > 0, delta / np.maximum(maxc, 1e-12), 0.0)
    h = np.zeros_like(maxc)
    nz = delta > 1e-12
    rc = np.where(nz, (maxc - r) / np.maximum(delta, 1e-12), 0.0)
    gc = np.where(nz, (maxc - g) / np.maximum(delta, 1e-12), 0.0)
    bc = np.where(nz, (maxc - b) / np.maximum(delta, 1e-12), 0.0)
    h = np.where(maxc == r, bc - gc, np.where(maxc == g, 2.0 + rc - bc, 4.0 + gc - rc))
    h = (h * 60.0) % 360.0
    return h, s, v


def stroke_mask(h, s, v, hue_lo, hue_hi, s_min=0.45, v_min=0.45):
    if hue_lo <= hue_hi:
        m = (h >= hue_lo) & (h <= hue_hi)
    else:
        m = (h >= hue_lo) | (h <= hue_hi)
    return m & (s >= s_min) & (v >= v_min)


def mask_points(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    return np.stack([xs, ys], axis=1).astype(np.float64)


def drop_text_components(mask: np.ndarray, min_dim: int = 60) -> np.ndarray:
    """Remove connected components whose bbox min-dimension < min_dim (text labels).

    Stroke curves span large bboxes in both dimensions; text labels are short.
    Pure-python BFS is fine at these pixel counts (~10k).
    """
    m = mask.copy()
    H, W = m.shape
    ys, xs = np.nonzero(m)
    seen = np.zeros_like(m, dtype=bool)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        comp = []
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and m[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        cy = [p[0] for p in comp]
        cx = [p[1] for p in comp]
        if (max(cx) - min(cx) + 1) < min_dim or (max(cy) - min(cy) + 1) < min_dim:
            for y, x in comp:
                m[y, x] = False
    return m


# ------------------------------------------------------------------- fitting


def fit_circle(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Kasa least-squares circle fit. Returns (center, radius)."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.stack([2 * x, 2 * y, np.ones_like(x)], axis=1)
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = math.sqrt(max(c + cx * cx + cy * cy, 1e-9))
    return np.array([cx, cy]), r


def ransac_circle(pts, iters=3000, tol=3.0, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    n = len(pts)
    if n < 3:
        return None, np.zeros(0, dtype=bool)
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        try:
            c, r = fit_circle(pts[idx])
        except Exception:
            continue
        if not (1.0 < r < 1e5):
            continue
        d = np.abs(np.linalg.norm(pts - c, axis=1) - r)
        inl = d < tol
        if best is None or inl.sum() > best[0]:
            best = (inl.sum(), inl)
    if best is None:
        return None, np.zeros(len(pts), dtype=bool)
    # refine on inliers
    c, r = fit_circle(pts[best[1]])
    d = np.abs(np.linalg.norm(pts - c, axis=1) - r)
    inl = d < max(tol, 2.0)
    c, r = fit_circle(pts[inl])
    return (c, r), inl


def fit_ellipse(pts: np.ndarray) -> dict:
    """Direct conic fit -> {'c': (cx,cy), 'a': major, 'b': minor, 'theta': rad}."""
    x, y = pts[:, 0], pts[:, 1]
    mx, my = x.mean(), y.mean()
    sx = max(x.std(), 1e-9)
    sy = max(y.std(), 1e-9)
    xn = (x - mx) / sx
    yn = (y - my) / sy
    A = np.stack([xn * xn, xn * yn, yn * yn, xn, yn, np.ones_like(xn)], axis=1)
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    a1, b1, c1, d1, e1, f1 = Vt[-1]
    # conic -> geometric params
    M = np.array([[a1, b1 / 2], [b1 / 2, c1]])
    evals, evecs = np.linalg.eigh(M)
    if np.any(evals <= 0):
        evals = np.abs(evals)
    # normalize so that conic value = 1 on the ellipse: solve for f
    # conic: [x y] M [x y]^T + [d e][x y] + f = 0 ; center = -0.5 M^-1 [d e]
    lin = np.array([d1, e1])
    center_n = -0.5 * np.linalg.solve(M, lin)
    f = f1
    const = center_n @ M @ center_n + lin @ center_n + f
    # value at center: const ; axes: sqrt(-const / eval)
    axes_n = np.sqrt(np.abs(-const / evals))
    axes = axes_n * np.array([sx, sy])
    center = center_n * np.array([sx, sy]) + np.array([mx, my])
    # evecs columns correspond to evals; axes pair in order
    order = np.argsort(-axes)  # major first
    axes = axes[order]
    vecs = evecs[:, order]
    theta = math.atan2(vecs[1, 0], vecs[0, 0])
    return {"c": center, "a": float(axes[0]), "b": float(axes[1]), "theta": theta}


def ellipse_radius_at(ell: dict, pts: np.ndarray) -> np.ndarray:
    """Elliptic radius rho of each point (rho=1 on the ellipse)."""
    ct, st = math.cos(-ell["theta"]), math.sin(-ell["theta"])
    d = pts - ell["c"]
    xr = d[:, 0] * ct - d[:, 1] * st
    yr = d[:, 0] * st + d[:, 1] * ct
    return np.sqrt((xr / ell["a"]) ** 2 + (yr / ell["b"]) ** 2)


def point_dist_to_ellipse(ell: dict, pts: np.ndarray) -> np.ndarray:
    """Approx geometric distance (rho-1)*local_scale, good for thin strokes."""
    rho = ellipse_radius_at(ell, pts)
    ct, st = math.cos(-ell["theta"]), math.sin(-ell["theta"])
    d = pts - ell["c"]
    xr = d[:, 0] * ct - d[:, 1] * st
    yr = d[:, 0] * st + d[:, 1] * ct
    grad = np.sqrt((xr / ell["a"] ** 2) ** 2 + (yr / ell["b"] ** 2) ** 2)
    local = 1.0 / np.maximum(grad, 1e-9)
    return (rho - 1.0) * local


def mvee_two_circles(c1, r1, c2, r2):
    """Minimal-area ellipse enclosing two disks.

    Axis frame: x along c1->c2, origin at segment midpoint, disk centers t1=-d/2,
    t2=+d/2. Nested minimization: golden section on center m in [-d/2, d/2];
    for each m, golden section on semi-major a of area(a) = a*max(b1(a), b2(a)),
    where b_i(a) = largest semi-minor that keeps disk i inside (bisection on
    the closed-form containment function g).
    """
    u = np.asarray(c2, dtype=float) - np.asarray(c1, dtype=float)
    d = float(np.linalg.norm(u))
    if d < 1e-9:
        r = max(r1, r2)
        return {"c": np.asarray(c1, dtype=float), "a": r, "b": r, "theta": 0.0}
    u = u / d
    t1, t2 = -d / 2.0, d / 2.0

    def g(t, r, a, b):
        # max over unit circle of ((t + r c)^2/a^2 + r^2 (1-c^2)/b^2) - 1
        den = r * (1.0 / a**2 - 1.0 / b**2)
        cs = [1.0, -1.0]
        if abs(den) > 1e-12:
            c_star = -(t / a**2) / den
            if -1.0 <= c_star <= 1.0:
                cs.append(c_star)
        return max(((t + r * c) ** 2) / a**2 + (r * r * (1.0 - c * c)) / b**2
                   for c in cs) - 1.0

    def b_fit(m, a, t, r):
        lo, hi = 1e-4, a
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if g(t - m, r, a, mid) > 0.0:
                lo = mid
            else:
                hi = mid
        return hi

    def area_at(m):
        a_min = max(abs(t1 - m) + r1, abs(t2 - m) + r2)
        a_lo, a_hi = a_min, 2.0 * a_min

        def obj(a):
            b = max(b_fit(m, a, t1, r1), b_fit(m, a, t2, r2))
            return a * b, b

        for _ in range(50):
            m1 = a_lo + (a_hi - a_lo) / 3.0
            m2 = a_hi - (a_hi - a_lo) / 3.0
            if obj(m1)[0] < obj(m2)[0]:
                a_hi = m2
            else:
                a_lo = m1
        a = 0.5 * (a_lo + a_hi)
        _, b = obj(a)
        return a * b, a, b

    lo_m, hi_m = -d / 2.0, d / 2.0
    for _ in range(50):
        m1 = lo_m + (hi_m - lo_m) / 3.0
        m2 = hi_m - (hi_m - lo_m) / 3.0
        if area_at(m1)[0] < area_at(m2)[0]:
            hi_m = m2
        else:
            lo_m = m1
    m = 0.5 * (lo_m + hi_m)
    _, a, b = area_at(m)
    center = (np.asarray(c1, dtype=float) + np.asarray(c2, dtype=float)) / 2.0 + u * m
    return {"c": center, "a": a, "b": b, "theta": math.atan2(u[1], u[0])}


# -------------------------------------------------------------------- report


def angular_gaps(angles: np.ndarray):
    """Sort angles and return them plus largest gap info."""
    srt = np.sort(angles)
    if len(srt) < 2:
        return srt, 0.0, 0.0
    gaps = np.diff(srt)
    wrap = 2 * math.pi - (srt[-1] - srt[0])
    gi = int(np.argmax(gaps))
    return srt, float(gaps[gi]), float(wrap)


def analyze(path: str) -> dict:
    img = np.asarray(Image.open(path).convert("RGB"))
    h, s, v = rgb_to_hsv_deg(img)
    H, W = h.shape

    # thresholds tuned against measured stroke colors:
    #   red (217,96,97) green (60,167,103) blue (68,130,217)
    #   purple (155,81,224) orange (242,153,74)
    red = stroke_mask(h, s, v, 350, 8, s_min=0.55, v_min=0.72)
    orange = stroke_mask(h, s, v, 20, 38, s_min=0.60, v_min=0.80)
    green = stroke_mask(h, s, v, 130, 158)
    blue = stroke_mask(h, s, v, 205, 228)
    purple = stroke_mask(h, s, v, 262, 290)

    out = {"image": path, "size": [W, H], "counts": {}}
    for name, m in (("red", red), ("orange", orange), ("green", green),
                    ("blue", blue), ("purple", purple)):
        out["counts"][name] = int(m.sum())
    if min(out["counts"][k] for k in ("red", "blue", "green", "orange", "purple")) < 50:
        print(json.dumps(out, indent=2))
        print("NOT ENOUGH STROKE PIXELS — check --hist", file=sys.stderr)
        return out

    disc_c, disc_r = fit_circle(mask_points(red))
    mac_c, mac_r = fit_circle(mask_points(blue))
    ell = fit_ellipse(mask_points(green))
    pole_c, pole_r = fit_circle(mask_points(orange))

    axis = mac_c - disc_c
    d = float(np.linalg.norm(axis))
    axis_u = axis / d
    axis_ang = math.atan2(axis_u[1], axis_u[0])
    Rd = disc_r
    mid = (disc_c + mac_c) / 2

    # green ellipse relations
    ell_ang = ell["theta"] % math.pi
    da = (ell_ang - axis_ang + math.pi / 2) % math.pi - math.pi / 2
    # mvee prediction
    mvee = mvee_two_circles(disc_c, disc_r, mac_c, mac_r)

    # purple decomposition
    pp = mask_points(purple)
    d_orange = np.abs(np.linalg.norm(pp - pole_c, axis=1) - pole_r)
    on_orange = d_orange < 4.0
    rest = pp[~on_orange]
    ang = np.arctan2((pp[on_orange, 1] - pole_c[1]), (pp[on_orange, 0] - pole_c[0]))
    srt, gap, wrap = angular_gaps(ang) if on_orange.sum() > 2 else (np.array([]), 0, 0)
    # arc = everything except the biggest wrap gap
    if len(srt):
        # rotate so that arc is contiguous: arc spans from srt[0]+gap.. srt[-1] plus wrap side
        arc_lo, arc_hi = srt[0], srt[-1]
    else:
        arc_lo = arc_hi = 0.0

    (bub_c, bub_r), bub_in = ransac_circle(rest, iters=4000, tol=3.5)
    funnel = rest[~bub_in] if bub_in.any() else rest

    # bubble params relative to disc
    bub_off = bub_c - disc_c
    off_along = float(bub_off @ axis_u)  # + = toward macula
    off_perp = float(bub_off @ np.array([-axis_u[1], axis_u[0]]))

    # funnel characterization: distance outside green ellipse in PD units
    fun_ell = point_dist_to_ellipse(ell, funnel) / Rd if len(funnel) else np.array([np.nan])
    # homothetic check: rho for scaled ellipse (a+kRd, b+kRd)
    def rho_scaled(k):
        e2 = {"c": ell["c"], "a": ell["a"] + k * Rd, "b": ell["b"] + k * Rd,
              "theta": ell["theta"]}
        return ellipse_radius_at(e2, funnel)

    out.update({
        "disc": {"c": disc_c.tolist(), "r": disc_r},
        "macula": {"c": mac_c.tolist(), "r": mac_r},
        "d_over_Rd": d / Rd,
        "Rm_over_Rd": mac_r / Rd,
        "green_ellipse": {"c": ell["c"].tolist(), "a": ell["a"], "b": ell["b"],
                          "b_over_a": ell["b"] / ell["a"],
                          "angle_vs_axis_deg": math.degrees(da),
                          "c_minus_mid_along": float((ell["c"] - mid) @ axis_u) / Rd,
                          "c_minus_mid_perp": float((ell["c"] - mid) @ np.array([-axis_u[1], axis_u[0]])) / Rd},
        "mvee_prediction": {"c": mvee["c"].tolist(), "a": mvee["a"], "b": mvee["b"],
                            "a_diff_pct": 100 * (mvee["a"] - ell["a"]) / ell["a"],
                            "b_diff_pct": 100 * (mvee["b"] - ell["b"]) / ell["b"]},
        "orange_circle": {"c": pole_c.tolist(), "r": pole_r,
                          "c_minus_ellc_along": float((pole_c - ell["c"]) @ axis_u) / Rd,
                          "c_minus_ellc_perp": float((pole_c - ell["c"]) @ np.array([-axis_u[1], axis_u[0]])) / Rd,
                          "R_minus_a_over_Rd": (pole_r - ell["a"]) / Rd,
                          "R_minus_(d/2+Rm+2Rd)_over_Rd": (pole_r - (d / 2 + mac_r + 2 * Rd)) / Rd},
        "purple_on_orange": {"n": int(on_orange.sum()),
                             "arc_from_deg": math.degrees(arc_lo) if len(srt) else None,
                             "arc_to_deg": math.degrees(arc_hi) if len(srt) else None},
        "bubble": {"c": (bub_c.tolist() if bub_c is not None else None),
                   "r": (bub_r if bub_c is not None else None),
                   "Rb_over_Rd": (bub_r / Rd if bub_c is not None else None),
                   "offset_along_over_Rd": off_along / Rd,
                   "offset_perp_over_Rd": off_perp / Rd,
                   "n_inliers": int(bub_in.sum())},
        "funnel": {"n": int(len(funnel)),
                   "dist_outside_ellipse_over_Rd_mean": float(np.mean(fun_ell)),
                   "dist_outside_ellipse_over_Rd_p10": float(np.percentile(fun_ell, 10)),
                   "dist_outside_ellipse_over_Rd_p90": float(np.percentile(fun_ell, 90))},
    })
    print(json.dumps(out, indent=2, default=float))
    return out


def hist(path: str) -> None:
    img = np.asarray(Image.open(path).convert("RGB"))
    h, s, v = rgb_to_hsv_deg(img)
    m = (s >= 0.45) & (v >= 0.45)
    hh = h[m]
    counts, edges = np.histogram(hh, bins=36, range=(0, 360))
    print(f"{path}: {m.sum()} saturated px")
    for i, c in enumerate(counts):
        if c > 50:
            print(f"  hue {edges[i]:5.0f}-{edges[i+1]:5.0f}: {c}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*")
    ap.add_argument("--hist", action="store_true")
    args = ap.parse_args()
    if not args.images:
        print(__doc__)
        sys.exit(1)
    if args.hist:
        for p in args.images:
            hist(p)
    else:
        for p in args.images:
            analyze(p)
