"""Fit the capsule model of central_inner to hand-drawn polygons.

Model:  central_inner = posterior_pole_disk  ∩  capsule,
where the capsule is the disc->macula axis segment thickened by radius rho
(the papillomacular protection corridor). Boundary = radial profile from the
MVEE-ellipse center E:  r(phi) = min(r_pp(phi), r_capsule(phi)).

Free params (in disc radii, Rd):
  rho   capsule radius
  a_t   capsule temporal end extension beyond macula center
  s     capsule nasal end center shift from disc center (+ toward macula)

Fit per sample with Nelder-Mead against the hand central_inner polygon,
aggregate medians -> defaults for labelmyeye/dr_geometry.DRParams.
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
from calibrate_from_screenshot import mvee_two_circles  # noqa: E402
from calibrate_fit_ci import (nelder_mead, poly_dist, ray_polyline_hit)  # noqa: E402
from calibrate_from_jsons import fit_circle_rms  # noqa: E402

N_ANG = 720


def capsule_profile(E, phis, P0, P1, rho):
    """First-hit distance of rays E+t*d(phi) against a capsule. NaN where miss."""
    u = (P1 - P0)
    L = float(np.linalg.norm(u))
    u = u / max(L, 1e-9)
    n = np.array([-u[1], u[0]])
    d = np.stack([np.cos(phis), np.sin(phis)], axis=1)      # N x 2
    best = np.full(len(phis), np.inf)

    # shaft: two lines P0 + off*n, direction u
    for off in (rho, -rho):
        Q = P0 + off * n
        denom = d[:, 0] * u[1] - d[:, 1] * u[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(np.abs(denom) > 1e-12,
                         ((Q[0] - E[0]) * u[1] - (Q[1] - E[1]) * u[0]) / denom,
                         np.nan)
        hit_x = E[:, None] if False else None  # placeholder, computed below
        ok = np.isfinite(t) & (t > 1e-6)
        hx = E[0] + np.where(ok, t, 0.0) * d[:, 0]
        hy = E[1] + np.where(ok, t, 0.0) * d[:, 1]
        proj = (hx - P0[0]) * u[0] + (hy - P0[1]) * u[1]
        ok &= (proj >= -1e-9) & (proj <= L + 1e-9)
        best = np.where(ok, np.minimum(best, np.where(ok, t, np.inf)), best)

    # caps: circles at P0 and P1
    for C in (P0, P1):
        f = np.array([E[0] - C[0], E[1] - C[1]])
        b = d @ f
        cc = float(f @ f) - rho * rho
        disc = b * b - cc
        with np.errstate(invalid="ignore"):
            t = np.where(disc >= 0, -b + np.sqrt(np.maximum(disc, 0.0)), np.nan)
        ok = np.isfinite(t) & (t > 1e-6)
        best = np.where(ok, np.minimum(best, np.where(ok, t, np.inf)), best)

    return np.where(np.isfinite(best), best, np.nan)


def analyze(path: str) -> dict | None:
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
    ax_u = (cm - cd) / float(np.linalg.norm(cm - cd))
    pp = shapes["posterior_pole_boundary"]
    ci = shapes["central_inner"]

    phis = np.linspace(0, 2 * math.pi, N_ANG, endpoint=False)
    r_pp = np.array([ray_polyline_hit(E, p, pp) for p in phis])
    r_pp = np.where(np.isfinite(r_pp), r_pp, np.nanmax(r_pp[np.isfinite(r_pp)])
                    if np.isfinite(r_pp).any() else 1e9)

    def profile(params):
        rho, a_t, s = params
        P0 = cd + s * Rd * ax_u
        P1 = cm + a_t * Rd * ax_u
        r_cap = capsule_profile(E, phis, P0, P1, rho * Rd)
        rr = np.where(np.isfinite(r_cap), np.fmin(r_pp, r_cap), r_pp)
        return np.stack([E[0] + rr * np.cos(phis),
                         E[1] + rr * np.sin(phis)], axis=1)

    def loss(params):
        rho, a_t, s = params
        if not (1.2 < rho < 4.8 and -0.5 < a_t < 3.5 and -1.5 < s < 2.5):
            return 1e9
        curve = profile(params)
        d_theirs = poly_dist(ci, curve).mean()
        d_mine = poly_dist(curve[::8], ci).mean()
        return (d_theirs + d_mine) / 2 / Rd

    best, loss = nelder_mead(loss, (2.6, 1.0, 0.0),
                             step=(0.2, 0.3, 0.3), iters=300)
    return {"file": os.path.basename(path), "rho": best[0], "a_t": best[1],
            "s": best[2], "loss_Rd": loss, "Rd": Rd,
            "d_over_Rd": float(np.linalg.norm(cm - cd)) / Rd,
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
            r = analyze(p)
        except Exception as e:  # noqa: BLE001
            print(f"{os.path.basename(p)}: ERROR {e}")
            continue
        if r is None:
            continue
        rows.append(r)
        if args.v:
            print(f"{r['file']}: rho={r['rho']:.2f} a_t={r['a_t']:.2f} "
                  f"s={r['s']:.2f} loss={r['loss_Rd']:.3f}Rd")
    if not rows:
        return
    print("\n=========== AGGREGATE (n=%d) ===========" % len(rows))
    for k in ("rho", "a_t", "s", "loss_Rd", "d_over_Rd", "Rm_over_Rd",
              "ell_a_over_Rd", "ell_b_over_Rd"):
        vals = [r[k] for r in rows]
        med = sorted(vals)[len(vals) // 2]
        print(f"  {k:16s} median={med:.3f}  min={min(vals):.3f} "
              f"max={max(vals):.3f}  std={np.std(vals):.3f}")


if __name__ == "__main__":
    main()
