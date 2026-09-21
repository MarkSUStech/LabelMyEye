"""DR auto-annotation orchestration: click -> SAM mask -> circle -> structures.

Keeps every model/geometry detail out of the UI layer:
  - model discovery (assets/models/mobile_sam_{encoder,decoder}.onnx)
  - click -> segmentation -> circle fit -> Shape
  - derived structures (macular_area / posterior_pole_boundary /
    central_inner) from the two circles, grouped via group_id
  - re-derivation from hand-edited optic/macular polygons

All Shapes are ordinary polygons: fully editable by the clinician.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

from labelmyeye.dr_geometry import DRParams, derive_structures, oval_polygon
from labelmyeye.shape import Shape

OPTIC = "optic"
MACULAR = "macular"
MACULAR_AREA = "macular_area"
POSTERIOR_POLE = "posterior_pole_boundary"
CENTRAL_INNER = "central_inner"

DR_LABELS = (OPTIC, MACULAR, MACULAR_AREA, POSTERIOR_POLE, CENTRAL_INNER)

_AUTO_DESC = "auto"
_AUTO_LOWCONF_DESC = "auto(low-conf)"


# ---------------------------------------------------------------- model path


def _base_dir() -> str:
    if getattr(sys, "frozen", False):          # PyInstaller
        return str(sys._MEIPASS)               # noqa: SLF001
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def model_paths() -> tuple[str, str] | None:
    base = os.path.join(_base_dir(), "assets", "models")
    enc = os.path.join(base, "mobile_sam_encoder.onnx")
    dec = os.path.join(base, "mobile_sam_decoder.onnx")
    if os.path.isfile(enc) and os.path.isfile(dec):
        return enc, dec
    return None


def get_segmenter():
    paths = model_paths()
    if paths is None:
        raise RuntimeError(
            "未找到分割模型（assets/models/mobile_sam_encoder/decoder.onnx）。\n"
            "运行 scripts/prepare_sam_onnx.py 生成模型文件。")
    from labelmyeye.sam_seg import SamSegmenter
    return SamSegmenter(*paths)


# ------------------------------------------------------------ circle helpers


def circle_from_polygon(points) -> tuple[float, float, float] | None:
    """Circle from a native circle shape, ellipse shape or dense polygon.

    Returns (cx, cy, r) with r = semi-major for ellipses.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) == 2:
        # native circle: [centre, edge point]
        cx, cy = float(pts[0][0]), float(pts[0][1])
        r = math.hypot(pts[1][0] - cx, pts[1][1] - cy)
        return (cx, cy, r) if r > 0 else None
    if len(pts) == 3:
        # native ellipse: use the centre and the semi-major axis
        cx, cy = float(pts[0][0]), float(pts[0][1])
        a = math.hypot(pts[1][0] - cx, pts[1][1] - cy)
        return (cx, cy, a) if a > 0 else None
    if len(pts) < 5:
        return None
    x, y = pts[:, 0], pts[:, 1]
    A = np.stack([2 * x, 2 * y, np.ones_like(x)], axis=1)
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = float(sol[0]), float(sol[1]), float(sol[2])
    r2 = c + cx * cx + cy * cy
    if r2 <= 0:
        return None
    return cx, cy, math.sqrt(r2)


def find_circles(shapes, prefer_native: bool = True) -> dict:
    """Map of label -> (cx, cy, r) for optic & macular shapes (native
    circle/ellipse or dense polygon)."""
    out = {}
    for s in shapes:
        if s.label not in (OPTIC, MACULAR):
            continue
        if s.shape_type in ("circle", "ellipse"):
            fit = circle_from_polygon(s.points)
            if fit is not None:
                out[s.label] = fit
        elif s.shape_type == "polygon" and s.label not in out:
            fit = circle_from_polygon(s.points)
            if fit is not None:
                out[s.label] = fit
    return out


def next_group_id(shapes) -> int:
    gids = [s.group_id for s in shapes if s.group_id is not None]
    return (max(gids) + 1) if gids else 1


# ------------------------------------------------------------- shape builders


def circle_shape(label: str, cx: float, cy: float, r: float, layer: str,
                 group_id: int | None, description: str) -> Shape:
    """Native circle: points = [centre, edge point] (LabelMe convention)."""
    pts = [(float(cx), float(cy)), (float(cx + r), float(cy))]
    return Shape(label=label, points=pts, shape_type="circle",
                 group_id=group_id, description=description, layer=layer)


def ellipse_shape(label: str, center, a: float, b: float, theta: float,
                  layer: str, group_id: int | None,
                  description: str) -> Shape:
    """Native ellipse: points = [centre, major end, minor end]."""
    cx, cy = float(center[0]), float(center[1])
    pts = [
        (cx, cy),
        (cx + a * math.cos(theta), cy + a * math.sin(theta)),
        (cx - b * math.sin(theta), cy + b * math.cos(theta)),
    ]
    return Shape(label=label, points=pts, shape_type="ellipse",
                 group_id=group_id, description=description, layer=layer)


def derived_shapes(disc: tuple[float, float, float],
                   mac: tuple[float, float, float], layer: str,
                   group_id: int | None, params: DRParams | None = None,
                   low_conf: bool = False) -> list[Shape]:
    """The three derived structures for one eye, as Shapes."""
    structs = derive_structures((disc[0], disc[1]), disc[2],
                                (mac[0], mac[1]), mac[2], params)
    desc = _AUTO_LOWCONF_DESC if low_conf else _AUTO_DESC
    ell = structs["ellipse"]
    pp = structs["posterior_pole"]
    ma = ellipse_shape(MACULAR_AREA, ell["c"], ell["a"], ell["b"],
                       ell["theta"], layer, group_id, desc)
    if abs(pp["a_t"] - pp["a_n"]) < 1e-6:
        # symmetric margins -> true ellipse, native editable shape
        po = ellipse_shape(POSTERIOR_POLE, pp["c"], pp["a_t"], pp["b"],
                           pp["theta"], layer, group_id, desc)
    else:
        # asymmetric oval (temporal != nasal margins) -> smooth polygon
        po = Shape(label=POSTERIOR_POLE, points=oval_polygon(pp, 96),
                   shape_type="polygon", group_id=group_id,
                   description=desc, layer=layer)
    ci_pts = [(float(x), float(y)) for x, y in structs["central_inner"]]
    ci = Shape(label=CENTRAL_INNER, points=ci_pts, shape_type="polygon",
               group_id=group_id, description=desc, layer=layer)
    return [ma, po, ci]


def full_annotation_shapes(disc, mac, layer: str, group_id: int | None = None,
                           params: DRParams | None = None,
                           low_conf: bool = False) -> list[Shape]:
    """optic + macular + the three derived structures, sharing one group_id.

    disc/mac are (cx, cy, r) tuples in image pixel coordinates.
    """
    desc = _AUTO_LOWCONF_DESC if low_conf else _AUTO_DESC
    shapes = [
        circle_shape(OPTIC, *disc, layer, group_id, desc),
        circle_shape(MACULAR, *mac, layer, group_id, desc),
    ]
    shapes += derived_shapes(disc, mac, layer, group_id, params, low_conf)
    return shapes
