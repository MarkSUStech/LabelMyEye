"""Direct-fit the parametric central_inner generator to hand-drawn polygons.

For each calibration JSON:
  - fit circles for optic / macular / posterior_pole_boundary (64-gons)
  - build MVEE ellipse of disc+macula; its center E is the profile hub
  - ray-cast r_pp(phi) = distance from E to the HAND-DRAWN posterior-pole
    polygon (decouples the fit from posterior-pole rule deviations)
  - model profile: r(phi) = smoothmax(r_pp(phi), r_bubble(phi); w)
    with bubble circle center = cd - delta*Rd*axis, radius = beta*Rd
  - Nelder-Mead over (beta, delta, w/Rd) minimizing symmetric mean boundary
    distance between model curve and the hand central_inner polygon

Reports per-sample best params and aggregate medians -> defaults for
labelmyeye/dr_geometry.DRParams.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_from_screenshot import fit_circle, mvee_two_circles  # noqa: E402
from calibrate_from_jsons import fit_circle_rms  # noqa: E402

N_ANG = 720


def ray_polyline_hit(E: np.ndarray, phi: float, poly: np.ndarray) -> float:
    """Smallest t>0 with E + t*dir(phi) on polyline loop poly (nan if none)."""
    dx, dy = math.cos(phi), math.sin(phi)
    px, py = E
    best = math.inf
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        # solve E + t d = A + s e
        det = dx * (-ey) - dy * (-ex)
        if abs(det) < 1e-12:
            continue
        wx, wy = ax - px, ay - py
        t = (wx * (-ey) - wy * (-ex)) / det
        s = (dx * wy - dy * wx) / det
        if t > 1e-6 and -1e-9 <= s <= 1.0 + 1e-9:
            best = min(best, t)
    return best


def ray_circle_hit(E: np.ndarray, phi: float, c: np.ndarray, r: float) -> float:
    dx, dy = math.cos(phi), math.sin(phi)
    fx, fy = E - c
    b = fx * dx + fy * dy
    cc = fx * fx + fy * fy - r * r
    disc = b * b - cc
    if disc < 0:
        return math.nan
    t = -b + math.sqrt(disc)
    return t if t > 1e-6 else math.nan


def smooth_max(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    return 0.5 * (a + b + np.sqrt((a - b) ** 2 + w * w))


def profile_points(E, betas) -> np.ndarray:
    phis = np.linspace(0, 2 * math.pi, N_ANG, endpoint=False)
    rr = betas(phis)
    ok = np.isfinite(rr)
    rr = np.where(ok, rr, 0.0)
    return np.stack([E[0] + rr * np.cos(phis), E[1] + rr * np.sin(phis)], axis=1)


def poly_dist(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Distance from each point to a closed polyline."""
    d = np.full(len(pts), np.inf)
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-12:
            continue
        t = np.clip(((pts - a) @ ab) / L2, 0.0, 1.0)
        proj = a + t[:, None] * ab
        d = np.minimum(d, np.linalg.norm(pts - proj, axis=1))
    return d


def nelder_mead(f, x0, step=(0.15, 0.15, 0.15), iters=250):
    x0 = np.array(x0, dtype=float)
    n = len(x0)
    pts = [x0.copy()]
    for i in range(n):
        p = x0.copy()
        p[i] += step[i]
        pts.append(p)
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        centroid = np.mean(pts[:-1], axis=0)
        xr = centroid + (centroid - pts[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = centroid + 2.0 * (centroid - pts[-1])
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = centroid + 0.5 * (pts[-1] - centroid)
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                for i in range(1, len(pts)):
                    pts[i] = pts[0] + 0.5 * (pts[i] - pts[0])
                    vals[i] = f(pts[i])
        if max(abs(v - vals[0]) for v in vals[1:]) < 1e-7:
            break
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def analyze(path: str, verbose=False) -> dict | None:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    shapes = {s["label"]: np.array(s["points"], dtype=float)
              for s in data.get("shapes", [])}
    need = {"optic", "macular", "posterior_pole_boundary", "central_inner"}
    if not need.issubset(shapes):
        return None

    (cd, Rd), _ = fit_circle_rms(shapes["optic"])
    (cm, Rm), _ = fit_circle_rms(shapes["macular"])
    ell = mvee_two_circles(cd, Rd, cm, Rm)
    E = ell["c"]
    ax_u = (cm - cd) / np.linalg.norm(cm - cd)
    pp = shapes["posterior_pole_boundary"]
    ci = shapes["central_inner"]

    phis = np.linspace(0, 2 * math.pi, N_ANG, endpoint=False)
    r_pp = np.array([ray_polyline_hit(E, p, pp) for p in phis])

    def model_loss(params):
        beta, delta, w_over = params
        if not (1.0 < beta < 4.0 and -2.0 < delta < 2.0 and 0.02 < w_over < 2.5):
            return 1e9
        Bc = cd - delta * Rd * ax_u
        Rb = beta * Rd
        r_bub = np.array([ray_circle_hit(E, p, Bc, Rb) for p in phis])
        # rays that miss a structure don't constrain the profile
        r_ppg = np.where(np.isfinite(r_pp), r_pp, 0.0)
        r_bub = np.where(np.isfinite(r_bub), r_bub, 0.0)
        both = (r_bub > 0) & (r_ppg > 0)
        rr = np.where(both, smooth_max(r_ppg, r_bub, w_over * Rd),
                      np.maximum(r_ppg, r_bub))
        curve = np.stack([E[0] + rr * np.cos(phis), E[1] + rr * np.sin(phis)], axis=1)
        d_theirs = poly_dist(ci, curve).mean()          # their pts -> my curve
        d_mine = poly_dist(curve[::8], ci).mean()       # my pts -> their polyline
        return (d_theirs + d_mine) / 2 / Rd

    best, loss = nelder_mead(model_loss, (2.1, 0.5, 0.8))
    beta, delta, w_over = best

    # quality with RULE-based orange (E-centered, a + 2Rd) instead of hand pp
    r_pp_rule = np.array([ray_circle_hit(E, p, E, ell["a"] + 2.0 * Rd)
                          for p in phis])
    Bc = cd - delta * Rd * ax_u
    r_bub = np.array([ray_circle_hit(E, p, Bc, beta * Rd) for p in phis])
    r_pp_rule = np.where(np.isfinite(r_pp_rule), r_pp_rule, 0.0)
    r_bub = np.where(np.isfinite(r_bub), r_bub, 0.0)
    both = (r_bub > 0) & (r_pp_rule > 0)
    rr = np.where(both, smooth_max(r_pp_rule, r_bub, w_over * Rd),
                  np.maximum(r_pp_rule, r_bub))
    curve = np.stack([E[0] + rr * np.cos(phis), E[1] + rr * np.sin(phis)], axis=1)
    d_rule = (poly_dist(ci, curve).mean() + poly_dist(curve[::8], ci).mean()) / 2 / Rd

    if verbose:
        print(f"{os.path.basename(path)}: beta={beta:.2f} delta={delta:.2f} "
              f"w={w_over:.2f} loss={loss:.3f}Rd loss_rule_orange={d_rule:.3f}Rd "
              f"Rd={Rd:.0f}px")
    return {"file": os.path.basename(path), "beta": beta, "delta": delta,
            "w_over_Rd": w_over, "loss_Rd": loss, "loss_rule_Rd": d_rule,
            "Rd": Rd, "d_over_Rd": float(np.linalg.norm(cm - cd)) / Rd,
            "Rm_over_Rd": Rm / Rd, "ell_a_over_Rd": ell["a"] / Rd,
            "ell_b_over_Rd": ell["b"] / Rd}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(__file__),
                                                  "calibration_data"))
    ap.add_argument("-v", action="store_true")
    args = ap.parse_args()
    rows = []
    for p in sorted(glob.glob(os.path.join(args.dir, "*.json"))):
        try:
            r = analyze(p, verbose=True)
            if r:
                rows.append(r)
        except Exception as e:  # noqa: BLE001
            print(f"{p}: ERROR {e}")
    if not rows:
        return

    def med(key):
        v = sorted(r[key] for r in rows)
        return v[len(v) // 2]

    print("\n=========== AGGREGATE (n=%d) ===========" % len(rows))
    for k in ("beta", "delta", "w_over_Rd", "loss_Rd", "loss_rule_Rd",
              "d_over_Rd", "Rm_over_Rd", "ell_a_over_Rd", "ell_b_over_Rd"):
        vals = [r[k] for r in rows]
        print(f"  {k:16s} median={med(k):.3f}  min={min(vals):.3f} "
              f"max={max(vals):.3f}  std={np.std(vals):.3f}")
    defaults = {k: round(float(med(k)), 3) for k in
                ("beta", "delta", "w_over_Rd")}
    print("\nSuggested DRParams central_inner defaults:", json.dumps(defaults))


if __name__ == "__main__":
    main()
