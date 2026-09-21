"""Unit tests for labelmyeye/dr_geometry.py (run: python scripts/test_dr_geometry.py)."""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from labelmyeye.dr_geometry import (  # noqa: E402
    DRParams, central_inner_polygon, circle_polygon, derive_structures,
    ellipse_polygon, elliptic_radius, posterior_pole_ellipse,
    tangent_ellipse_two_circles, resample_polygon, sutherland_hodgman,
    _signed_area,
)


def test_circle_polygon():
    pts = circle_polygon((10, 20), 50, 64)
    assert len(pts) == 64
    for x, y in pts:
        assert abs(math.hypot(x - 10, y - 20) - 50) < 1e-9
    print("ok circle_polygon")


def test_ellipse_polygon():
    pts = ellipse_polygon((0, 0), 100, 40, 0.3, 64)
    ct, st = math.cos(0.3), math.sin(0.3)
    for x, y in pts:
        xr = x * ct + y * st
        yr = -x * st + y * ct
        v = (xr / 100) ** 2 + (yr / 40) ** 2
        assert abs(v - 1) < 1e-6
    print("ok ellipse_polygon")


def test_tangent_ellipse_symmetric():
    # two equal disks r=1 at ±2: tips touch the outer poles, a == 3
    ell = tangent_ellipse_two_circles((0, 0), 1.0, (4, 0), 1.0)
    assert abs(ell["c"][0] - 2.0) < 1e-6, ell
    assert abs(ell["c"][1]) < 1e-6
    assert abs(ell["a"] - 3.0) < 1e-6, ell
    # tips lie exactly on the circles: P_left=(-1,0), P_right=(5,0)
    assert abs(math.hypot(ell["c"][0] - 5.0, ell["c"][1] - 0.0) - 3.0) < 1e-6
    assert abs(math.hypot(ell["c"][0] + 1.0, ell["c"][1] - 0.0) - 3.0) < 1e-6
    # both disks inside
    for t in (0.0, 4.0):
        for i in range(32):
            th = 2 * math.pi * i / 32
            x, y = t + math.cos(th), math.sin(th)
            xr = x - ell["c"][0]
            yr = y
            assert (xr / ell["a"]) ** 2 + (yr / ell["b"]) ** 2 <= 1 + 1e-4
    # minimality of b: shrinking it slightly must poke out
    xr = 2.0 - ell["c"][0]
    b = ell["b"]

    def g_at(bv):
        den = 1.0 * (1.0 / 9.0 - 1.0 / (bv * bv))
        cs = [1.0, -1.0]
        if abs(den) > 1e-12:
            cs.append(-(2.0 / 9.0) / den)
        return max(((2.0 + c) ** 2) / 9.0 + (1.0 - c * c) / (bv * bv)
                   for c in cs)

    assert g_at(b) <= 1 + 1e-6
    assert g_at(b * 0.98) > 1.0
    print("ok tangent ellipse symmetric", ell["a"], ell["b"])


def test_tangent_ellipse_random_containment():
    rng = np.random.default_rng(7)
    for _ in range(20):
        c1 = rng.uniform(-50, 50, 2)
        c2 = c1 + rng.uniform(60, 300, 2)
        r1, r2 = rng.uniform(20, 120, 2)
        ell = tangent_ellipse_two_circles(c1, r1, c2, r2)
        ct, st = math.cos(ell["theta"]), math.sin(ell["theta"])
        for c, r in ((c1, r1), (c2, r2)):
            for i in range(24):
                th = 2 * math.pi * i / 24
                x, y = c[0] + r * math.cos(th), c[1] + r * math.sin(th)
                dx, dy = x - ell["c"][0], y - ell["c"][1]
                xr = dx * ct + dy * st
                yr = -dx * st + dy * ct
                assert (xr / ell["a"]) ** 2 + (yr / ell["b"]) ** 2 <= 1 + 1e-3, \
                    "disk pokes out of tangent ellipse"
    print("ok tangent ellipse random containment")


def test_sh_square_square():
    a = [(0, 0), (10, 0), (10, 10), (0, 10)]
    b = [(5, 5), (15, 5), (15, 15), (5, 15)]
    inter = sutherland_hodgman(a, b)
    area = _signed_area(inter)
    assert abs(area - 25.0) < 1e-9, area
    # disjoint -> empty
    c = [(100, 100), (110, 100), (110, 110), (100, 110)]
    assert sutherland_hodgman(a, c) == []
    print("ok sutherland_hodgman")


def test_resample():
    pts = circle_polygon((0, 0), 10, 64)
    out = resample_polygon(pts, 32)
    assert len(out) == 32
    area0 = abs(_signed_area(pts))
    area1 = abs(_signed_area(out))
    assert abs(area0 - area1) / area0 < 0.01
    print("ok resample_polygon")


def test_central_inner_sanity():
    disc, Rd = (300, 500), 80.0
    mac = (720, 580)
    p = DRParams()
    ci = central_inner_polygon(disc, Rd, mac, 0.85 * Rd, p)
    assert len(ci) == p.n_ci
    ell = tangent_ellipse_two_circles(disc, Rd, mac, 0.85 * Rd)
    pp = posterior_pole_ellipse(disc, Rd, mac, 0.85 * Rd, p)
    a_pp, b_pp = pp["a_t"], pp["b"]
    # edge margins: temporal 2PD (=4 radii), nasal exactly 1PD (=2 radii),
    # vertical 2PD — measured from the green ellipse edge
    assert abs(pp["a_t"] - ell["a"] - 4.0 * Rd) < 1e-6, (pp, ell)
    assert abs(pp["a_n"] - ell["a"] - 2.0 * Rd) < 1e-6, (pp, ell)
    assert abs(pp["b"] - ell["b"] - 4.0 * Rd) < 1e-6, (pp, ell)

    axis_ang = math.atan2(mac[1] - disc[1], mac[0] - disc[0])

    def radial_at(angle_deg):
        th = axis_ang + math.radians(angle_deg)
        best = 0.0
        x, y = ell["c"]
        for px, py in ci:
            a = math.atan2(py - y, px - x) - th
            if abs(math.atan2(math.sin(a), math.cos(a))) < 0.04:
                best = max(best, math.hypot(px - x, py - y))
        return best

    # inside the pp oval (check against its radial distance per vertex)
    for x, y in ci:
        dx, dy = x - pp["c"][0], y - pp["c"][1]
        ct, st = math.cos(pp["theta"]), math.sin(pp["theta"])
        xr = dx * ct + dy * st
        yr = -dx * st + dy * ct
        r_pp = elliptic_radius(a_pp, b_pp, math.atan2(yr, xr))
        assert math.hypot(xr, yr) <= r_pp + 1e-6

    # temporal direction hugs the pp boundary; notch centre dips inward
    r_temporal = radial_at(0)
    r_pp_temporal = elliptic_radius(a_pp, b_pp, -pp["theta"] - axis_ang)
    assert r_temporal > 0.92 * r_pp_temporal, (r_temporal, r_pp_temporal)
    r_notch = radial_at(p.ci_angle_sup)
    r_pp_notch = elliptic_radius(a_pp, b_pp,
                                 math.radians(p.ci_angle_sup) + axis_ang
                                 - pp["theta"])
    assert r_notch < 0.85 * r_pp_notch, (r_notch, r_pp_notch)

    # disc centre inside the region (point-in-polygon by ray casting)
    def inside(pt, poly):
        x, y = pt
        n = len(poly)
        c = False
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if (y1 > y) != (y2 > y):
                xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if xin > x:
                    c = not c
        return c

    assert inside(disc, ci), "optic disc must be inside central_inner"
    assert inside(mac, ci), "macula must be inside central_inner"
    print("ok central_inner sanity")


def test_derive_structures():
    out = derive_structures((300, 500), 80.0, (720, 580), 68.0, DRParams())
    assert set(out) >= {"macular_area", "posterior_pole_boundary",
                        "central_inner", "ellipse"}
    # dr_geometry level: dense polygon samplings
    assert len(out["macular_area"]) == 64
    assert len(out["posterior_pole_boundary"]) == 64
    assert len(out["central_inner"]) >= 24
    # left-eye-like direction (macula flipped) also works
    out2 = derive_structures((900, 400), 70.0, (480, 470), 60.0)
    assert len(out2["central_inner"]) >= 24
    print("ok derive_structures")


if __name__ == "__main__":
    test_circle_polygon()
    test_ellipse_polygon()
    test_tangent_ellipse_symmetric()
    test_tangent_ellipse_random_containment()
    test_sh_square_square()
    test_resample()
    test_central_inner_sanity()
    test_derive_structures()
    print("ALL GEOMETRY TESTS PASSED")
