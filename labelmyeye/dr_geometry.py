"""DR (diabetic retinopathy) derived-structure geometry.

Given the optic-disc circle and macula circle (in image pixel coordinates),
derives the annotation structures used for photocoagulation planning:

  macular_area            minimal-area ellipse enclosing both circles
                          (tangent to both; major axis along the
                          disc->macula line) — the temporal vascular-arcade
                          region / "macular area".
  posterior_pole_boundary ellipse with the same centre as macular_area,
                          both semi-axes extended by k_pp * Rd (a 2PD margin
                          all around)
  central_inner           photocoagulation-sensitive zone ("kidney"): the
                          posterior-pole disk with two smooth notches cut in
                          above and below the disc->macula axis (the laser-
                          accessible zones flanking the papillomacular
                          bundle). Hugs the posterior pole temporally and
                          nasally, wraps both the disc and the macula.

All outputs are dense closed polygons (list of (x, y) tuples in image
pixels), compatible with the app's polygon-only Shape model. Geometry is
pure numpy; no third-party deps beyond numpy.

Coordinate convention: x right, y down (image pixel coordinates). "Temporal"
direction = from disc centre toward macula centre, regardless of eye side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------- params


@dataclass
class DRParams:
    """Coefficients in PD units. 1 PD = disc DIAMETER = 2 x disc radius."""

    k_pp_temporal: float = 2.0   # pp margin beyond temporal ellipse edge (PD)
    k_pp_nasal: float = 1.0      # pp margin beyond nasal ellipse edge (PD)
    k_pp_vertical: float = 2.0   # pp margin beyond top/bottom edges (PD)
    k_mac: float = 0.85          # expected macula-circle radius = k_mac * Rd
    ci_depth_sup: float = 0.50   # superior notch depth (fraction of local R)
    ci_depth_inf: float = 0.46   # inferior notch depth (fraction of local R)
    ci_width: float = 40.0       # notch Gaussian sigma in degrees (smoothness)
    ci_angle_sup: float = 123.0  # superior notch centre (deg from axis, +)
    ci_angle_inf: float = -128.0  # inferior notch centre (deg from axis)
    ci_bubble_pd: float = 0.0    # optional disc-wrap bubble (off by default)
    n_circle: int = 64           # polygon points per circle / ellipse
    n_ci: int = 96               # polygon points for central_inner

    def validated(self) -> "DRParams":
        return DRParams(
            k_pp_temporal=min(max(float(self.k_pp_temporal), 0.5), 6.0),
            k_pp_nasal=min(max(float(self.k_pp_nasal), 0.5), 6.0),
            k_pp_vertical=min(max(float(self.k_pp_vertical), 0.5), 6.0),
            k_mac=min(max(float(self.k_mac), 0.4), 2.0),
            ci_depth_sup=min(max(float(self.ci_depth_sup), 0.0), 0.9),
            ci_depth_inf=min(max(float(self.ci_depth_inf), 0.0), 0.9),
            ci_width=min(max(float(self.ci_width), 8.0), 90.0),
            ci_angle_sup=float(self.ci_angle_sup),
            ci_angle_inf=float(self.ci_angle_inf),
            ci_bubble_pd=min(max(float(self.ci_bubble_pd), 0.0), 3.0),
            n_circle=int(min(max(self.n_circle, 16), 256)),
            n_ci=int(min(max(self.n_ci, 24), 512)),
        )


# ------------------------------------------------------------ basic polygons


def circle_polygon(center, radius: float, n: int = 64):
    cx, cy = float(center[0]), float(center[1])
    return [(float(cx + radius * math.cos(2 * math.pi * i / n)),
             float(cy + radius * math.sin(2 * math.pi * i / n)))
            for i in range(n)]


def ellipse_polygon(center, a: float, b: float, theta: float, n: int = 64):
    """CCW-in-image-space sampling of ellipse (semi-axes a along theta)."""
    cx, cy = float(center[0]), float(center[1])
    ct, st = math.cos(theta), math.sin(theta)
    pts = []
    for i in range(n):
        t = 2.0 * math.pi * i / n
        x, y = a * math.cos(t), b * math.sin(t)
        pts.append((float(cx + x * ct - y * st), float(cy + x * st + y * ct)))
    return pts


def _signed_area(pts) -> float:
    pts = np.asarray(pts, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _ensure_ccw(pts):
    return list(pts) if _signed_area(pts) > 0 else list(reversed(list(pts)))


def resample_polygon(pts, n: int):
    """Resample a closed polygon to n points, evenly by arc length."""
    arr = np.asarray(pts, dtype=float)
    if len(arr) < 3:
        return list(map(tuple, arr))
    closed = np.vstack([arr, arr[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 0:
        return list(map(tuple, arr[:n]))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, total, n, endpoint=False)
    out = []
    j = 0
    for t in targets:
        while j < len(seg) - 1 and cum[j + 1] < t:
            j += 1
        seglen = seg[j] if seg[j] > 0 else 1e-9
        f = (t - cum[j]) / seglen
        p = closed[j] * (1 - f) + closed[j + 1] * f
        out.append((float(p[0]), float(p[1])))
    return out


# ------------------------------------------------------- convex intersection


def sutherland_hodgman(subject, clip):
    """Intersect two convex polygons (both CCW). Returns list of points."""
    subject = _ensure_ccw(subject)
    clip = _ensure_ccw(clip)
    output = list(subject)
    m = len(clip)
    for i in range(m):
        if not output:
            return []
        a = clip[i]
        b = clip[(i + 1) % m]
        ex, ey = b[0] - a[0], b[1] - a[1]
        input_list = output
        output = []
        n = len(input_list)
        for j in range(n):
            cur = input_list[j]
            prv = input_list[j - 1]
            cur_side = ex * (cur[1] - a[1]) - ey * (cur[0] - a[0])
            prv_side = ex * (prv[1] - a[1]) - ey * (prv[0] - a[0])
            cur_in = cur_side >= 0   # inside = left of a->b (CCW clip)
            prv_in = prv_side >= 0
            if cur_in:
                if not prv_in:
                    output.append(_intersect_point(prv, cur, a, ex, ey))
                output.append(cur)
            elif prv_in:
                output.append(_intersect_point(prv, cur, a, ex, ey))
    return output


def _intersect_point(p1, p2, a, ex, ey):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    denom = ex * dy - ey * dx          # cross(e, d)
    if abs(denom) < 1e-12:
        return (float(p1[0]), float(p1[1]))
    t = (ex * (a[1] - p1[1]) - ey * (a[0] - p1[0])) / denom  # cross(e, a-p1)/det
    return (float(p1[0] + t * dx), float(p1[1] + t * dy))


def _capsule_polygon(p0, p1, radius: float, per_cap: int = 32):
    """Convex polygon approximating the capsule (segment p0-p1 thickened)."""
    ax, ay = float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1])
    L = math.hypot(ax, ay)
    if L < 1e-9:
        return circle_polygon(p0, radius, 2 * per_cap)
    ux, uy = ax / L, ay / L
    nx, ny = -uy, ux
    pts = []
    # cap at p1 (semicircle from +n to -n through u)
    for i in range(per_cap + 1):
        t = -math.pi / 2.0 + math.pi * i / per_cap
        c, s = math.cos(t), math.sin(t)
        pts.append((p1[0] + radius * (c * ux + s * nx),
                    p1[1] + radius * (c * uy + s * ny)))
    # cap at p0 (semicircle from -n to +n through -u)
    for i in range(per_cap + 1):
        t = -math.pi / 2.0 + math.pi * i / per_cap
        c, s = math.cos(t), math.sin(t)
        pts.append((p0[0] + radius * (-c * ux - s * nx),
                    p0[1] + radius * (-c * uy - s * ny)))
    return pts


# --------------------------------------------------------------- structures


def tangent_ellipse_two_circles(c1, r1: float, c2, r2: float) -> dict:
    """Ellipse tangent to both circles, enclosing both.

    The major-axis tips touch circle 1 at its far pole (away from circle 2)
    and circle 2 at its far pole; the semi-minor axis is the shortest that
    still keeps both circles inside (tangent to the binding circle at one
    point on each side).  This is the classic "vascular-arcade" ellipse:
    it visibly wraps — and touches — both circles.

    Returns {'c': (x, y), 'a': semi-major, 'b': semi-minor, 'theta': angle}.
    """
    c1 = np.asarray(c1, dtype=float)
    c2 = np.asarray(c2, dtype=float)
    u = c2 - c1
    d = float(np.linalg.norm(u))
    if d < 1e-9:
        r = max(float(r1), float(r2))
        return {"c": (float(c1[0]), float(c1[1])), "a": r, "b": r, "theta": 0.0}
    u = u / d
    p_left = c1 - r1 * u          # far pole of circle 1
    p_right = c2 + r2 * u         # far pole of circle 2
    center = (p_left + p_right) / 2.0
    a = float(np.linalg.norm(p_right - p_left)) / 2.0
    t1 = float((c1 - center) @ u)
    t2 = float((c2 - center) @ u)

    def overflows(b: float) -> bool:
        for t, r in ((t1, float(r1)), (t2, float(r2))):
            den = r * (1.0 / (a * a) - 1.0 / (b * b))
            cs = [1.0, -1.0]
            if abs(den) > 1e-12:
                cstar = -(t / (a * a)) / den
                if -1.0 <= cstar <= 1.0:
                    cs.append(cstar)
            g = max(((t + r * c) ** 2) / (a * a)
                    + (r * r * (1.0 - c * c)) / (b * b) for c in cs)
            if g > 1.0 + 1e-12:
                return True
        return False

    lo, hi = 1e-3, a
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if overflows(mid):
            lo = mid
        else:
            hi = mid
    b = hi
    return {"c": (float(center[0]), float(center[1])), "a": a, "b": b,
            "theta": math.atan2(float(u[1]), float(u[0]))}


def macular_area_polygon(disc_c, disc_r, mac_c, mac_r=None,
                         params: DRParams | None = None):
    """Green arcade ellipse: tangent to both circles, enclosing both."""
    p = (params or DRParams()).validated()
    if mac_r is None:
        mac_r = p.k_mac * disc_r
    ell = tangent_ellipse_two_circles(disc_c, disc_r, mac_c, mac_r)
    return ellipse_polygon(ell["c"], ell["a"], ell["b"], ell["theta"],
                           p.n_circle), ell


# historical alias (minimal enclosing ellipse used before the tangent rule)
mvee_two_circles = tangent_ellipse_two_circles


def elliptic_radius(a: float, b: float, delta: float) -> float:
    """Radius of an ellipse (semi-axes a,b) at angle delta from its major axis."""
    ca, sa = math.cos(delta), math.sin(delta)
    denom = math.sqrt((b * ca) ** 2 + (a * sa) ** 2)
    return (a * b) / max(denom, 1e-9)


def posterior_pole_ellipse(disc_c, disc_r, mac_c, mac_r=None,
                           params: DRParams | None = None) -> dict:
    """Orange posterior-pole boundary — measured from the green ellipse EDGE.

    The green (arcade) ellipse's outline is pushed outward:
      - 2 PD (disc DIAMETERS) beyond the temporal edge,
      - 1 PD beyond the nasal edge,
      - 2 PD beyond the superior/inferior edges.
    1 PD = 2 * disc radius. Returns an asymmetric oval:
      {'c': centre, 'a_t': temporal semi-extent, 'a_n': nasal semi-extent,
       'b': vertical semi-extent, 'theta': axis angle}.
    """
    p = (params or DRParams()).validated()
    if mac_r is None:
        mac_r = p.k_mac * disc_r
    ell = tangent_ellipse_two_circles(disc_c, disc_r, mac_c, mac_r)
    pd = 2.0 * disc_r                       # one disc diameter
    return {"c": ell["c"],
            "a_t": ell["a"] + p.k_pp_temporal * pd,
            "a_n": ell["a"] + p.k_pp_nasal * pd,
            "b": ell["b"] + p.k_pp_vertical * pd,
            "theta": ell["theta"]}


def nasal_depth_floor(params: DRParams) -> float:
    """Total notch depth the Gaussian tails produce at the nasal pole (180°
    from the axis). central_inner subtracts this floor so the nasal margin
    is measured to the drawn outline instead of being eroded by the tails."""
    p = (params or DRParams()).validated()
    s = math.radians(p.ci_width)
    g_sup = math.exp(-0.5 * (math.radians(180.0 - p.ci_angle_sup) / s) ** 2)
    g_inf = math.exp(-0.5 * (math.radians(180.0 + p.ci_angle_inf) / s) ** 2)
    return max(p.ci_depth_sup * g_sup, p.ci_depth_inf * g_inf)


def oval_radius(pp: dict, delta):
    """Radial distance of the asymmetric posterior-pole oval at ellipse-frame
    angle delta (scalar or numpy array) from its major axis."""
    if isinstance(delta, np.ndarray):
        cosd, sind = np.cos(delta), np.sin(delta)
        a = np.where(cosd >= 0, pp["a_t"], pp["a_n"])
        b = pp["b"]
        return a * b / np.maximum(np.sqrt((b * cosd) ** 2 + (a * sind) ** 2),
                                  1e-9)
    cosd, sind = math.cos(delta), math.sin(delta)
    a = pp["a_t"] if cosd >= 0 else pp["a_n"]
    b = pp["b"]
    return (a * b) / max(math.hypot(b * cosd, a * sind), 1e-9)


def _ray_circle_distance(origin: np.ndarray, center: np.ndarray, radius: float,
                         phis: np.ndarray) -> np.ndarray:
    """Distance from origin along each ray direction phis to its exit point
    of the circle (center, radius); NaN where the ray misses."""
    d = np.stack([np.cos(phis), np.sin(phis)], axis=1)
    f = origin - center
    b = d @ f
    cc = float(f @ f) - radius * radius
    disc = b * b - cc
    t = np.where(disc >= 0, -b + np.sqrt(np.maximum(disc, 0.0)), np.nan)
    return np.where(t > 1e-6, t, np.nan)


def central_inner_polygon(disc_c, disc_r, mac_c, mac_r=None,
                          params: DRParams | None = None):
    """Purple photocoagulation-sensitive zone ("kidney").

    Posterior-pole ellipse with two smooth notches cut into the sectors above
    and below the disc->macula axis (the laser-accessible zones outside the
    papillomacular bundle / vascular arcades). Profile from the pp centre:

        r(phi) = R_pp(phi) * (1 - max(d_sup*bump(rel-a_sup),
                                     d_inf*bump(rel-a_inf)))

    where R_pp(phi) is the posterior-pole ellipse's radial distance and the
    bumps are raised cosines of half-width ci_width. Hugs the posterior pole
    on the temporal and nasal sides, dips inward in the two notch sectors,
    and always encloses disc and macula.
    """
    p = (params or DRParams()).validated()
    if mac_r is None:
        mac_r = p.k_mac * disc_r
    pp = posterior_pole_ellipse(disc_c, disc_r, mac_c, mac_r, p)
    center = pp["c"]
    ax = (float(mac_c[0]) - float(disc_c[0]), float(mac_c[1]) - float(disc_c[1]))
    axis_ang = math.atan2(ax[1], ax[0])
    a_sup = math.radians(p.ci_angle_sup)
    a_inf = math.radians(p.ci_angle_inf)

    def bump(rel: np.ndarray, a: float) -> np.ndarray:
        # Gaussian bump: infinitely smooth transition at the shoulders;
        # ci_width is the Gaussian sigma in degrees (transition smoothness)
        x = (rel - a + math.pi) % (2.0 * math.pi) - math.pi
        s = math.radians(p.ci_width)
        return np.exp(-0.5 * (x / s) ** 2)

    phis = np.linspace(0.0, 2.0 * math.pi, p.n_ci, endpoint=False)
    rel = (phis - axis_ang + math.pi) % (2.0 * math.pi) - math.pi
    # ellipse-frame angle: the pp oval's major axis lies along the
    # disc->macula axis (theta == axis_ang), so its frame offset cancels out
    delta = rel + (axis_ang - pp["theta"])
    r_pp = np.asarray(oval_radius(pp, delta))
    depth = np.maximum(p.ci_depth_sup * bump(rel, a_sup),
                       p.ci_depth_inf * bump(rel, a_inf))
    # the Gaussian tails bleed into the nasal sector; subtract a floor that is
    # weighted to peak at the nasal pole only, so the nasal 1PD margin stays
    # clean while the two notch depths keep their calibrated values
    floor_val = nasal_depth_floor(p)
    if floor_val > 0.0:
        d_nasal = np.abs((rel - math.pi + math.pi) % (2.0 * math.pi) - math.pi)
        s_n = math.radians(30.0)
        w_nasal = np.exp(-0.5 * (d_nasal / s_n) ** 2)
        depth = np.maximum(depth - floor_val * w_nasal, 0.0)
    rr = r_pp * (1.0 - depth)

    # disc-wrap bubble: the outline never comes closer to the disc than
    # ci_bubble_pd beyond the disc edge — guarantees a rounded nasal cap
    # that fully encloses the disc regardless of notch depth
    if p.ci_bubble_pd > 0.0:
        r_bub = _ray_circle_distance(np.asarray(center, dtype=float),
                                     np.asarray(disc_c, dtype=float),
                                     disc_r * (1.0 + 2.0 * p.ci_bubble_pd),
                                     phis)
        rr = np.maximum(rr, np.where(np.isfinite(r_bub), r_bub, 0.0))

    return [(float(center[0] + r * math.cos(t)),
             float(center[1] + r * math.sin(t)))
            for t, r in zip(phis, rr)]


def oval_polygon(pp: dict, n: int = 96):
    """Sample the asymmetric posterior-pole oval as a closed polygon."""
    cx, cy = pp["c"]
    phis = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    rr = np.asarray(oval_radius(pp, phis - pp["theta"]))
    return [(float(cx + r * math.cos(t)), float(cy + r * math.sin(t)))
            for t, r in zip(phis, rr)]


def derive_structures(disc_c, disc_r, mac_c, mac_r=None,
                      params: DRParams | None = None) -> dict:
    """All derived polygons for one eye.

    Returns {'macular_area': [...], 'posterior_pole_boundary': [...],
             'central_inner': [...], 'ellipse': {...},
             'posterior_pole': {...oval params...}}.
    """
    p = (params or DRParams()).validated()
    ma, ell = macular_area_polygon(disc_c, disc_r, mac_c, mac_r, p)
    pp = posterior_pole_ellipse(disc_c, disc_r, mac_c, mac_r, p)
    pp_poly = oval_polygon(pp, p.n_circle)
    ci = central_inner_polygon(disc_c, disc_r, mac_c, mac_r, p)
    return {
        "macular_area": ma,
        "posterior_pole_boundary": pp_poly,
        "central_inner": ci,
        "ellipse": ell,
        "posterior_pole": pp,
    }
