"""Calibrate DR geometry rules from hand-annotated LabelMe JSONs.

Reads scripts/calibration_data/*.json (labels: optic, macular,
posterior_pole_boundary, central_inner — polygons produced by the group),
fits circles, builds the two-circle minimal enclosing ellipse, and measures
which parametric rule reproduces the hand-drawn posterior_pole_boundary and
central_inner. Aggregates over all samples.

Usage: python scripts/calibrate_from_jsons.py [--dir scripts/calibration_data]
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


def circle_from_polygon(pts: np.ndarray):
    (c, r), resid = fit_circle_rms(pts)
    return c, r, resid


def fit_circle_rms(pts: np.ndarray):
    c, r = fit_circle(pts)
    d = np.abs(np.linalg.norm(pts - c, axis=1) - r)
    return (c, r), float(np.sqrt(np.mean(d**2)))


def point_line_dist(pts: np.ndarray, p0: np.ndarray, direction: np.ndarray) -> np.ndarray:
    n = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
    return np.abs((pts - p0) @ n)


def external_tangents(c1, r1, c2, r2):
    """Outer tangent lines of two disjoint circles. Returns list of (p0, dir)."""
    d = np.linalg.norm(c2 - c1)
    u = (c2 - c1) / d
    lines = []
    # angle of external tangent
    alpha = math.asin(max(-1.0, min(1.0, (r1 - r2) / d)))
    for s in (1.0, -1.0):
        theta = alpha + s * math.pi / 2
        n = np.array([math.cos(theta), math.sin(theta)])
        # line: points p with n·p = n·c1 + s*r1 = n·c2 + s*r2  (signed)
        p0 = c1 + s * r1 * n
        lines.append((p0, np.array([-n[1], n[0]])))
    return lines


def analyze_sample(path: str) -> dict | None:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    shapes = {s["label"]: np.array(s["points"], dtype=float)
              for s in data.get("shapes", [])}
    needed = {"optic", "macular", "posterior_pole_boundary", "central_inner"}
    if not needed.issubset(shapes):
        return {"path": os.path.basename(path), "missing": sorted(needed - set(shapes))}

    cd, Rd, res_d = circle_from_polygon(shapes["optic"])
    cm, Rm, res_m = circle_from_polygon(shapes["macular"])
    co, Ro, res_o = circle_from_polygon(shapes["posterior_pole_boundary"])
    ci = shapes["central_inner"]

    axis = cm - cd
    d = float(np.linalg.norm(axis))
    ax_u = axis / d
    ax_ang = math.atan2(ax_u[1], ax_u[0])
    perp = np.array([-ax_u[1], ax_u[0]])

    ell = mvee_two_circles(cd, Rd, cm, Rm)
    # MVEE center offset from midpoint, along axis (in Rd)
    mid = (cd + cm) / 2
    ell_m_off = float((ell["c"] - mid) @ ax_u) / Rd

    k_pp_ell = (Ro - ell["a"]) / Rd
    k_pp_mid = (Ro - (d / 2 + Rm + 2 * Rd)) / Rd
    co_off = float((co - mid) @ ax_u) / Rd

    # ---- central_inner decomposition
    dist_o = np.abs(np.linalg.norm(ci - co, axis=1) - Ro)
    on_arc = dist_o < max(3.0, 0.025 * Ro)
    rest = ci[~on_arc]

    # bubble: points clearly nasal of disc center
    proj_rest = (rest - cd) @ ax_u
    cand = rest[(proj_rest < 0.5 * Rd)]
    if len(cand) >= 4:
        (bc, Rb), res_b = fit_circle_rms(cand)
        # refine over all rest
        dd = np.abs(np.linalg.norm(rest - bc, axis=1) - Rb)
        inl = dd < max(3.0, 0.05 * Rd)
        if inl.sum() >= 4:
            (bc, Rb), res_b = fit_circle_rms(rest[inl])
        else:
            inl = np.zeros(len(rest), dtype=bool)
    else:
        bc, Rb, res_b, inl = None, None, None, np.zeros(len(rest), dtype=bool)

    funnel = rest[~inl] if bc is not None else rest

    # arc angular span (relative to temporal direction = 0)
    arc_pts = ci[on_arc]
    if len(arc_pts) >= 2:
        ang = (np.degrees(np.arctan2(*(arc_pts - co).T[::-1])) - math.degrees(ax_ang)
               + 180) % 360 - 180
        ang = np.sort(ang)
        gaps = np.diff(ang)
        gi = int(np.argmax(gaps)) if len(gaps) else -1
        if gi >= 0 and gaps[gi] > 90:  # real opening on the nasal side
            lo_a, hi_a = ang[gi + 1], ang[gi]
        else:
            lo_a, hi_a = float(ang[0]), float(ang[-1])
        arc_center = (lo_a + hi_a) / 2 if hi_a > lo_a else (lo_a + hi_a + 360) / 2
    else:
        lo_a = hi_a = arc_center = None

    # bubble center offset
    if bc is not None:
        off_along = float((bc - cd) @ ax_u) / Rd  # + toward macula
        off_perp = float((bc - cd) @ perp) / Rd
    else:
        off_along = off_perp = None

    # funnel diagnostics
    if len(funnel):
        ell_d = point_dist_to_ellipse_local(ell, funnel) / Rd
        f_ratio = np.linalg.norm(funnel - co, axis=1) / Ro
        tan_d = np.full(len(funnel), np.nan)
        if bc is not None:
            tan_d = np.min(np.stack(
                [point_line_dist(funnel, p0, dr) for p0, dr in
                 external_tangents(bc, Rb, co, Ro)]), axis=0)
    else:
        ell_d = np.array([np.nan])
        f_ratio = np.array([np.nan])
        tan_d = np.array([np.nan])

    return {
        "file": os.path.basename(path),
        "resid_px": {"optic": res_d, "macular": res_m, "pp": res_o},
        "d_over_Rd": d / Rd, "Rm_over_Rd": Rm / Rd,
        "ell_a_over_Rd": ell["a"] / Rd, "ell_b_over_Rd": ell["b"] / Rd,
        "ell_m_off_Rd": ell_m_off,
        "Ro_over_Rd": Ro / Rd,
        "k_pp_from_ell": k_pp_ell, "k_pp_from_mid": k_pp_mid,
        "co_off_from_mid_Rd": co_off,
        "ci_n": len(ci), "arc_n": int(on_arc.sum()),
        "arc_lo_deg": lo_a, "arc_hi_deg": hi_a, "arc_center_deg": arc_center,
        "Rb_over_Rd": (Rb / Rd if bc is not None else None),
        "bub_off_along_Rd": off_along, "bub_off_perp_Rd": off_perp,
        "bub_resid": res_b,
        "fun_n": int(len(funnel)),
        "fun_ell_med": float(np.nanmedian(ell_d)),
        "fun_ell_iqr": [float(np.nanpercentile(ell_d, 25)),
                        float(np.nanpercentile(ell_d, 75))],
        "fun_f_med": float(np.nanmedian(f_ratio)),
        "fun_f_iqr": [float(np.nanpercentile(f_ratio, 25)),
                      float(np.nanpercentile(f_ratio, 75))],
        "fun_tan_med_px": float(np.nanmedian(tan_d)),
        "fun_tan_p90_px": float(np.nanpercentile(tan_d, 90)),
    }


def point_dist_to_ellipse_local(ell: dict, pts: np.ndarray) -> np.ndarray:
    th = -ell["theta"]
    ct, st = math.cos(th), math.sin(th)
    dd = pts - ell["c"]
    xr = dd[:, 0] * ct - dd[:, 1] * st
    yr = dd[:, 0] * st + dd[:, 1] * ct
    rho = np.sqrt((xr / ell["a"]) ** 2 + (yr / ell["b"]) ** 2)
    grad = np.sqrt((xr / ell["a"] ** 2) ** 2 + (yr / ell["b"] ** 2) ** 2)
    return (rho - 1.0) / np.maximum(grad, 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(__file__),
                                                  "calibration_data"))
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.dir, "*.json")))
    rows = []
    for p in files:
        try:
            r = analyze_sample(p)
        except Exception as e:  # noqa: BLE001
            r = {"file": os.path.basename(p), "error": str(e)}
        rows.append(r)
        print(json.dumps(r, default=float, ensure_ascii=False))

    ok = [r for r in rows if "d_over_Rd" in r]
    if not ok:
        print("no valid samples")
        return

    def med(key):
        vals = [r[key] for r in ok if r.get(key) is not None
                and not (isinstance(r[key], float) and math.isnan(r[key]))]
        return (float(np.median(vals)), len(vals)) if vals else (None, 0)

    print("\n================ AGGREGATE ================")
    for k in ("d_over_Rd", "Rm_over_Rd", "ell_a_over_Rd", "ell_b_over_Rd",
              "ell_m_off_Rd", "Ro_over_Rd", "k_pp_from_ell", "k_pp_from_mid",
              "co_off_from_mid_Rd", "Rb_over_Rd", "bub_off_along_Rd",
              "bub_off_perp_Rd", "arc_center_deg", "fun_ell_med", "fun_f_med",
              "fun_tan_med_px"):
        m, n = med(k)
        print(f"  median {k:22s} = {m if m is None else round(m, 3)}  (n={n})")
    vals = [r["k_pp_from_ell"] for r in ok if r.get("k_pp_from_ell") is not None]
    if vals:
        print(f"  k_pp_from_ell spread: min={min(vals):.2f} max={max(vals):.2f} "
              f"std={np.std(vals):.2f}")
    vals = [r["Rb_over_Rd"] for r in ok if r.get("Rb_over_Rd") is not None]
    if vals:
        print(f"  Rb_over_Rd spread: min={min(vals):.2f} max={max(vals):.2f} "
              f"std={np.std(vals):.2f}")


if __name__ == "__main__":
    main()
