"""Fit the notch model of central_inner to hand-drawn polygons.

Model (radial profile from the pp-circle centre E):

    r(phi) = R_pp * (1 - max(depth_sup * bump(rel - a_sup),
                            depth_inf * bump(rel - a_inf)))

where rel = phi - axis_angle (0 = toward macula), bump = raised cosine of
half-width w, and (depth_sup, depth_inf, a_sup, a_inf, w) are fitted.
This reproduces the hand-drawn kidney: hugging the posterior pole on the
temporal and nasal sides, dipping inward above and below the
papillomacular axis.
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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calibrate_from_jsons import fit_circle_rms  # noqa: E402
from calibrate_fit_ci import nelder_mead, poly_dist  # noqa: E402
from labelmyeye.dr_geometry import (  # noqa: E402
    DRParams, oval_radius, posterior_pole_ellipse)

N_ANG = 720


def notch_profile(phis, axis_ang, pp, depth_sup, depth_inf, a_sup, a_inf, s):
    rel = (phis - axis_ang + math.pi) % (2 * math.pi) - math.pi

    def bump(center):
        x = (rel - center + math.pi) % (2 * math.pi) - math.pi
        return np.exp(-0.5 * (x / s) ** 2)

    r_pp = np.asarray(oval_radius(pp, rel - pp["theta"]))
    depth = np.maximum(depth_sup * bump(a_sup), depth_inf * bump(a_inf))
    return r_pp * (1.0 - depth)


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
    params = DRParams()
    pp = posterior_pole_ellipse(cd, Rd, cm, Rm, params)
    E = pp["c"]
    axv = (np.array(cm, dtype=float) - np.array(cd, dtype=float))
    axis_ang = math.atan2(axv[1], axv[0])
    ci = shapes["central_inner"]
    phis = np.linspace(0, 2 * math.pi, N_ANG, endpoint=False)

    def curve(p):
        depth_sup, depth_inf, a_sup, a_inf, w = p
        rr = notch_profile(phis, axis_ang, pp,
                           depth_sup, depth_inf, a_sup, a_inf, w)
        return np.stack([E[0] + rr * np.cos(phis),
                         E[1] + rr * np.sin(phis)], axis=1)

    def loss(p):
        depth_sup, depth_inf, a_sup, a_inf, w = p
        if not (0.05 < depth_sup < 0.9 and 0.05 < depth_inf < 0.9
                and math.radians(60) < a_sup < math.radians(170)
                and -math.radians(170) < a_inf < -math.radians(60)
                and math.radians(6) < w < math.radians(70)):
            return 1e9
        c = curve(p)
        return (poly_dist(ci, c).mean() + poly_dist(c[::8], ci).mean()) / 2 / Rd

    best, loss_v = nelder_mead(
        loss, (0.30, 0.28, math.radians(130), math.radians(-135),
               math.radians(22)),
        step=(0.06, 0.06, math.radians(12), math.radians(12), math.radians(12)),
        iters=400)
    return {"file": os.path.basename(path),
            "depth_sup": best[0], "depth_inf": best[1],
            "a_sup_deg": math.degrees(best[2]),
            "a_inf_deg": math.degrees(best[3]),
            "w_deg": math.degrees(best[4]),
            "loss_Rd": loss_v}


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
            print(f"{r['file']}: d_sup={r['depth_sup']:.2f} d_inf={r['depth_inf']:.2f}"
                  f" a_sup={r['a_sup_deg']:.0f} a_inf={r['a_inf_deg']:.0f}"
                  f" w={r['w_deg']:.0f} loss={r['loss_Rd']:.3f}Rd")
    if not rows:
        return
    print("\n=========== AGGREGATE (n=%d) ===========" % len(rows))
    for k in ("depth_sup", "depth_inf", "a_sup_deg", "a_inf_deg", "w_deg",
              "loss_Rd"):
        vals = [r[k] for r in rows]
        med = sorted(vals)[len(vals) // 2]
        print(f"  {k:12s} median={med:.2f}  min={min(vals):.2f} "
              f"max={max(vals):.2f}  std={np.std(vals):.2f}")


if __name__ == "__main__":
    main()
