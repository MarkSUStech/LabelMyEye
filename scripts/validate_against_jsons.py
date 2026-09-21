"""Validate derived DR structures against the 14 hand-annotated JSONs.

Fits optic/macular circles from each JSON's hand polygons, builds derived
structures with dr_geometry, and reports symmetric boundary distance (in disc
radii) between derived and hand-drawn posterior_pole_boundary/central_inner.
Renders overlay PNGs for --render samples.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from labelmyeye.dr_geometry import (  # noqa: E402
    DRParams, central_inner_polygon, oval_polygon, posterior_pole_ellipse)
from calibrate_from_jsons import fit_circle_rms  # noqa: E402
from calibrate_fit_ci import poly_dist  # noqa: E402


def pp_polygon(cd, Rd, cm, Rm, params):
    e = posterior_pole_ellipse(cd, Rd, cm, Rm, params)
    return oval_polygon(e, 128)


def boundary_stats(derived, hand):
    d1 = poly_dist(np.asarray(hand, dtype=float),
                   np.asarray(derived, dtype=float)).mean()
    d2 = poly_dist(np.asarray(derived[::2], dtype=float),
                   np.asarray(hand, dtype=float)).mean()
    mx = max(poly_dist(np.asarray(hand, dtype=float),
                       np.asarray(derived, dtype=float)).max(),
             poly_dist(np.asarray(derived[::2], dtype=float),
                       np.asarray(hand, dtype=float)).max())
    return d1, (d1 + d2) / 2, mx


def render(path, out_png, derived):
    data = json.load(open(path, encoding="utf-8"))
    S = {s["label"]: np.array(s["points"], float) for s in data["shapes"]}
    W, H = int(data["imageWidth"]) // 2, int(data["imageHeight"]) // 2
    img = Image.open(next(
        os.path.join(os.path.dirname(path), os.path.splitext(
            os.path.basename(path))[0] + ext)
        for ext in (".png", ".jpg") if os.path.isfile(os.path.join(
            os.path.dirname(path),
            os.path.splitext(os.path.basename(path))[0] + ext)))).convert("RGB")
    img = img.resize((W, H))
    dr = ImageDraw.Draw(img)
    hand_cols = {"posterior_pole_boundary": (255, 165, 0),
                 "central_inner": (190, 60, 255)}
    for lab, col in hand_cols.items():
        if lab in S:
            pts = [tuple(q / 2) for q in np.asarray(S[lab])]
            dr.line(pts + [pts[0]], fill=col, width=3)
    for lab, col in (("posterior_pole_boundary", (255, 255, 255)),
                     ("central_inner", (0, 255, 120))):
        pts = [tuple(q / 2) for q in np.asarray(derived[lab], dtype=float)]
        dr.line(pts + [pts[0]], fill=col, width=2)
    img.save(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(
        os.path.dirname(__file__), "calibration_data"))
    ap.add_argument("--render", nargs="*", default=[])
    args = ap.parse_args()
    params = DRParams().validated()
    rows = []
    for p in sorted(glob.glob(os.path.join(args.dir, "*.json"))):
        data = json.load(open(p, encoding="utf-8"))
        S = {s["label"]: np.array(s["points"], float) for s in data["shapes"]}
        if not {"optic", "macular", "posterior_pole_boundary",
                "central_inner"}.issubset(S):
            continue
        (cd, Rd), _ = fit_circle_rms(S["optic"])
        (cm, Rm), _ = fit_circle_rms(S["macular"])
        pp = pp_polygon(cd, Rd, cm, Rm, params)
        ci = central_inner_polygon(cd, Rd, cm, Rm, params)
        pp_mean, pp_sym, pp_max = boundary_stats(pp, S["posterior_pole_boundary"])
        ci_mean, ci_sym, ci_max = boundary_stats(ci, S["central_inner"])
        rows.append({"file": os.path.basename(p), "pp_sym_Rd": pp_sym / Rd,
                     "ci_sym_Rd": ci_sym / Rd, "pp_max_Rd": pp_max / Rd,
                     "ci_max_Rd": ci_max / Rd, "Rd": Rd})
        print(f"{os.path.basename(p)}:  pp sym={pp_sym/Rd:.2f}Rd max={pp_max/Rd:.2f}"
              f" |  ci sym={ci_sym/Rd:.2f}Rd max={ci_max/Rd:.2f}")
    if rows:
        for k in ("pp_sym_Rd", "ci_sym_Rd"):
            v = sorted(r[k] for r in rows)
            print(f"median {k} = {v[len(v)//2]:.3f}")
    for name in args.render:
        p = os.path.join(args.dir, name + ".json")
        if not os.path.isfile(p):
            continue
        data = json.load(open(p, encoding="utf-8"))
        S = {s["label"]: np.array(s["points"], float) for s in data["shapes"]}
        (cd, Rd), _ = fit_circle_rms(S["optic"])
        (cm, Rm), _ = fit_circle_rms(S["macular"])
        derived = {
            "posterior_pole_boundary": pp_polygon(cd, Rd, cm, Rm, params),
            "central_inner": central_inner_polygon(cd, Rd, cm, Rm, params),
        }
        render(p, os.path.join(args.dir, f"_valid_{name}.png"), derived)
        print("rendered", name)


if __name__ == "__main__":
    main()
