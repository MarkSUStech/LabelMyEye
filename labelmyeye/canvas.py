"""Annotation canvas with view zoom and before/after registration transform."""

from __future__ import annotations

import colorsys
import math
import zlib
from typing import List, Optional, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
    QTransform,
    QWheelEvent,
)
from PySide6.QtWidgets import QMenu, QWidget

from labelmyeye.shape import Point, Shape

CREATE = "create"
EDIT = "edit"
MOVE_BEFORE = "move_before"  # pan/scale the before overlay for registration
AI_OPTIC = "ai_optic"        # click -> segment optic disc
AI_MACULAR = "ai_macular"    # click -> segment macula

CIRCLE = "circle"    # LabelMe convention: points = [centre, edge point]
ELLIPSE = "ellipse"  # points = [centre, major-end, minor-end]

# dense polygons (>= this many vertices) deform smoothly when a vertex is
# dragged — neighbouring points follow with a Gaussian falloff
SMOOTH_DEFORM_MIN_POINTS = 24
SMOOTH_DEFORM_REACH = 5  # influence = perimeter points / REACH, each side

# stable colors for the DR structures (matches the annotation convention)
DR_PALETTE = {
    "optic": (237, 96, 97),                 # red
    "macular": (68, 130, 217),              # blue
    "macular_area": (60, 167, 103),         # green
    "posterior_pole_boundary": (242, 153, 74),  # orange
    "central_inner": (155, 81, 224),        # purple
}


def label_color(name: str, alpha: int = 180) -> QColor:
    if name in DR_PALETTE:
        r, g, b = DR_PALETTE[name]
        return QColor(r, g, b, alpha)
    # stable across runs (str hash is salted per process)
    h = zlib.crc32(name.encode("utf-8")) % 360
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, 0.75, 0.95)
    return QColor(int(r * 255), int(g * 255), int(b * 255), alpha)


def ellipse_params(pts):
    """Ellipse control points [(cx,cy),(major end),(minor end)] ->
    (cx, cy, semi-major a, semi-minor b, rotation theta in radians)."""
    (cx, cy), (x1, y1), (x2, y2) = pts[0], pts[1], pts[2]
    dx, dy = x1 - cx, y1 - cy
    a = math.hypot(dx, dy)
    theta = math.atan2(dy, dx)
    nx, ny = -math.sin(theta), math.cos(theta)
    b = abs((x2 - cx) * nx + (y2 - cy) * ny)
    return cx, cy, a, max(b, 1e-6), theta


class Canvas(QWidget):
    shapes_changed = Signal()
    status_message = Signal(str)
    mode_changed = Signal(str)
    request_label = Signal()
    before_transform_changed = Signal()
    ai_click = Signal(float, float, str)  # active-layer coords + "optic"/"macular"
    ai_box = Signal(float, float, float, float, str)  # x0,y0,x1,y1 + label

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(400, 300)

        # images
        self.after_pixmap: Optional[QPixmap] = None
        self.before_pixmap: Optional[QPixmap] = None
        self.after_size = (0, 0)  # w, h in image pixels
        self.before_size = (0, 0)

        # shapes
        self.shapes: List[Shape] = []
        self.current: Optional[Shape] = None  # in-progress polygon
        self.selected: Optional[Shape] = None
        self.selected_vertex: Optional[int] = None
        self.h_vertex: Optional[Tuple[Shape, int]] = None  # hover vertex
        self.h_edge: Optional[Tuple[Shape, int]] = None  # hover edge (shape, start index)
        self._selected_vertex_sticky: Optional[int] = None  # keeps vertex selected after release

        # view transform (screen <-> after image coords)
        self.scale = 1.0
        self.offset = QPointF(0, 0)  # screen offset of image origin

        # before -> after registration transform
        self.before_scale = 1.0
        self.before_tx = 0.0
        self.before_ty = 0.0
        self.before_rotation = 0.0  # degrees
        self.before_opacity = 0.45

        self.mode = CREATE
        self.active_layer = "after"  # "after" | "before" | "single"
        self.dual_mode = False
        self.show_before = True
        self.show_after = True
        self.show_before_labels = True
        # AI box-prompt drag state
        self._ai_anchor = None
        self._ai_rubber = None
        self.show_after_labels = True

        self._panning = False
        self._moving_before = False
        self._moving_shape = False
        self._moving_vertex = False
        self._last_pos = QPointF()
        self._press_pos = QPointF()
        self._shape_start_points: List[Point] = []
        self._did_drag = False

        self.crosshair = True
        self.line_width = 2.0

        # callbacks for undo snapshots (set by MainWindow)
        self.on_before_edit = None  # callable()
        self.on_edit_label = None  # callable(Shape)

    # ---------- image helpers ----------
    @staticmethod
    def pil_to_pixmap(pil_image) -> QPixmap:
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        w, h = pil_image.size
        data = pil_image.tobytes("raw", "RGB")
        qimg = QImage(data, w, h, w * 3, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())

    def set_after_image(self, pil_image) -> None:
        self.after_pixmap = self.pil_to_pixmap(pil_image)
        self.after_size = (pil_image.width, pil_image.height)
        self.fit_view()
        self.update()

    def set_before_image(self, pil_image) -> None:
        self.before_pixmap = self.pil_to_pixmap(pil_image)
        self.before_size = (pil_image.width, pil_image.height)
        # center before on after if sizes differ
        aw, ah = self.after_size
        bw, bh = self.before_size
        if aw and ah and bw and bh:
            self.before_scale = min(aw / bw, ah / bh)
            self.before_tx = (aw - bw * self.before_scale) / 2
            self.before_ty = (ah - bh * self.before_scale) / 2
            self.before_rotation = 0.0
        self.dual_mode = True
        self.update()

    def clear_images(self) -> None:
        self.after_pixmap = None
        self.before_pixmap = None
        self.after_size = (0, 0)
        self.before_size = (0, 0)
        self.shapes = []
        self.current = None
        self.selected = None
        self.dual_mode = False
        self.update()

    def fit_view(self) -> None:
        if not self.after_size[0]:
            return
        aw, ah = self.after_size
        margin = 20
        sx = (self.width() - margin * 2) / aw
        sy = (self.height() - margin * 2) / ah
        self.scale = max(0.02, min(sx, sy))
        self.offset = QPointF(
            (self.width() - aw * self.scale) / 2,
            (self.height() - ah * self.scale) / 2,
        )
        self.update()

    # ---------- coordinate transforms ----------
    def after_to_screen(self, x: float, y: float) -> QPointF:
        return QPointF(x * self.scale + self.offset.x(), y * self.scale + self.offset.y())

    def screen_to_after(self, p: QPointF) -> Point:
        return (
            (p.x() - self.offset.x()) / self.scale,
            (p.y() - self.offset.y()) / self.scale,
        )

    def before_world_transform(self) -> QTransform:
        """Before-image → after-image. Must match paintEvent exactly."""
        bw, bh = self.before_size
        cx, cy = bw / 2.0, bh / 2.0
        t = QTransform()
        t.translate(self.before_tx, self.before_ty)
        t.scale(self.before_scale, self.before_scale)
        t.translate(cx, cy)
        t.rotate(self.before_rotation)  # Qt: degrees, Y-down (clockwise positive)
        t.translate(-cx, -cy)
        return t

    def before_to_after(self, x: float, y: float) -> Point:
        p = self.before_world_transform().map(QPointF(x, y))
        return (p.x(), p.y())

    def after_to_before(self, x: float, y: float) -> Point:
        inv, ok = self.before_world_transform().inverted()
        if not ok:
            return (x, y)
        p = inv.map(QPointF(x, y))
        return (p.x(), p.y())

    def rotate_before(self, delta_deg: float, pivot_after: Optional[Point] = None) -> None:
        """Rotate before image; keep pivot_after fixed in after-space."""
        if abs(delta_deg) < 1e-9:
            return
        if pivot_after is None:
            aw, ah = self.after_size
            pivot_after = (aw / 2.0, ah / 2.0)
        # before-space point currently under the pivot
        bx, by = self.after_to_before(*pivot_after)
        self.before_rotation = (self.before_rotation + delta_deg) % 360.0
        if self.before_rotation > 180.0:
            self.before_rotation -= 360.0
        # where that before point landed after rotation (old translation)
        nx, ny = self.before_to_after(bx, by)
        self.before_tx += pivot_after[0] - nx
        self.before_ty += pivot_after[1] - ny
        self.before_transform_changed.emit()
        self.status_message.emit(f"Before rotation: {self.before_rotation:.1f}°")
        self.update()

    def scale_before(self, factor: float, pivot_after: Optional[Point] = None) -> None:
        """Scale before image; keep pivot_after fixed in after-space."""
        if factor <= 0 or abs(factor - 1.0) < 1e-9:
            return
        if pivot_after is None:
            aw, ah = self.after_size
            pivot_after = (aw / 2.0, ah / 2.0)
        bx, by = self.after_to_before(*pivot_after)
        new_s = max(0.05, min(20.0, self.before_scale * factor))
        if abs(new_s - self.before_scale) < 1e-12:
            return
        self.before_scale = new_s
        nx, ny = self.before_to_after(bx, by)
        self.before_tx += pivot_after[0] - nx
        self.before_ty += pivot_after[1] - ny
        self.before_transform_changed.emit()
        self.status_message.emit(f"Before scale: {self.before_scale:.3f}")
        self.update()

    def set_before_rotation(self, degrees: float, pivot_after: Optional[Point] = None) -> None:
        self.rotate_before(degrees - self.before_rotation, pivot_after)

    def shape_to_after_points(self, shape: Shape) -> List[Point]:
        if shape.layer == "before":
            return [self.before_to_after(x, y) for x, y in shape.points]
        return list(shape.points)

    def screen_to_layer(self, p: QPointF, layer: str) -> Point:
        ax, ay = self.screen_to_after(p)
        if layer == "before":
            return self.after_to_before(ax, ay)
        return (ax, ay)

    def get_before_transform(self) -> dict:
        return {
            "scale": self.before_scale,
            "tx": self.before_tx,
            "ty": self.before_ty,
            "rotation": self.before_rotation,
            "opacity": self.before_opacity,
        }

    def set_before_transform(self, t: dict) -> None:
        self.before_scale = float(t.get("scale", 1.0))
        self.before_tx = float(t.get("tx", 0.0))
        self.before_ty = float(t.get("ty", 0.0))
        self.before_rotation = float(t.get("rotation", 0.0))
        self.before_opacity = float(t.get("opacity", 0.45))
        self.before_transform_changed.emit()
        self.update()

    # ---------- shapes API ----------
    def set_shapes(self, shapes: List[Shape]) -> None:
        self.shapes = shapes
        self.current = None
        self.selected = None
        self.shapes_changed.emit()
        self.update()

    def snapshot(self) -> dict:
        return {
            "shapes": [s.copy() for s in self.shapes],
            "before_transform": self.get_before_transform(),
        }

    def restore(self, state: dict) -> None:
        self.shapes = [s.copy() for s in state.get("shapes", [])]
        if "before_transform" in state:
            self.set_before_transform(state["before_transform"])
        self.current = None
        self.selected = None
        self.shapes_changed.emit()
        self.update()

    def shape_to_before_points(self, shape: Shape) -> List[Point]:
        if shape.layer == "before":
            return list(shape.points)
        # after / single → before via inverse registration transform
        return [self.after_to_before(x, y) for x, y in shape.points]

    def transformed_shapes_for_export(self) -> List[Shape]:
        """All shapes in before-image coordinates (dual-mode export target)."""
        out: List[Shape] = []
        for s in self.shapes:
            ns = s.copy()
            if self.dual_mode:
                ns.points = self.shape_to_before_points(s)
                ns.layer = "before"
            else:
                ns.points = list(s.points)
            out.append(ns)
        return out

    # ---------- mode ----------
    def set_mode(self, mode: str) -> None:
        self.mode = mode
        if mode != CREATE:
            self.current = None
        self.h_vertex = None
        self.h_edge = None
        self.mode_changed.emit(mode)
        self.update()

    def set_active_layer(self, layer: str) -> None:
        self.active_layer = layer
        self.status_message.emit(f"Active layer: {layer}")
        self.update()

    # ---------- painting ----------
    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(40, 42, 48))

        if self.after_pixmap and self.show_after:
            p.setOpacity(1.0)
            target = self._after_target_rect()
            p.drawPixmap(target, self.after_pixmap)

        if self.before_pixmap and self.dual_mode and self.show_before:
            p.save()
            p.setOpacity(self.before_opacity)
            # view transform, then the same before→after matrix used by labels
            p.translate(self.offset)
            p.scale(self.scale, self.scale)
            p.setTransform(self.before_world_transform(), combine=True)
            p.drawPixmap(0, 0, self.before_pixmap)
            p.restore()

        # draw shapes
        for shape in self.shapes:
            if shape.layer == "before" and not self.show_before_labels:
                continue
            if shape.layer in ("after", "single") and not self.show_after_labels:
                continue
            self._draw_shape(p, shape)

        if self.current and self.current.points:
            self._draw_shape(p, self.current, draft=True)

        # crosshair
        if self.crosshair and self.underMouse():
            pos = self.mapFromGlobal(self.cursor().pos())
            # use last mouse if available
            if hasattr(self, "_mouse_pos"):
                pos = self._mouse_pos
            pen = QPen(QColor(255, 255, 255, 80), 1, Qt.DashLine)
            p.setPen(pen)
            p.drawLine(0, int(pos.y()), self.width(), int(pos.y()))
            p.drawLine(int(pos.x()), 0, int(pos.x()), self.height())

        # AI rubber-band box prompt
        if self._ai_rubber is not None:
            (ax0, ay0), (ax1, ay1) = self._ai_rubber
            s0 = self.after_to_screen(ax0, ay0)
            s1 = self.after_to_screen(ax1, ay1)
            color = QColor(*DR_PALETTE.get(
                "optic" if self.mode == AI_OPTIC else "macular", (255, 255, 255)))
            pen = QPen(color, 2, Qt.DashLine)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(s0, s1).normalized())

        # registration hint
        if self.mode == MOVE_BEFORE and self.dual_mode:
            p.setPen(QColor(255, 200, 80))
            p.drawText(
                12,
                22,
                "Registration: drag move · Ctrl+Wheel scale · Ctrl+Shift+Wheel rotate · [ ] fine rotate",
            )

    def _after_target_rect(self):
        from PySide6.QtCore import QRect

        aw, ah = self.after_size
        return QRect(
            int(self.offset.x()),
            int(self.offset.y()),
            int(aw * self.scale),
            int(ah * self.scale),
        )

    def _draw_shape(self, p: QPainter, shape: Shape, draft: bool = False) -> None:
        pts = self.shape_to_after_points(shape)
        if not pts:
            return
        screen = [self.after_to_screen(x, y) for x, y in pts]
        color = label_color(shape.label or "?")
        if shape.selected:
            color = QColor(255, 220, 60, 220)
        if shape.layer == "before":
            # dashed outline for before-layer labels
            pen = QPen(color, self.line_width + (1 if shape.selected else 0), Qt.DashLine)
        else:
            pen = QPen(color, self.line_width + (1 if shape.selected else 0), Qt.SolidLine)
        if draft:
            pen.setStyle(Qt.DotLine)
        p.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(40 if not shape.selected else 70)
        p.setBrush(fill)

        if shape.shape_type == "circle" and len(screen) >= 2:
            c = screen[0]
            r = math.hypot(screen[1].x() - c.x(), screen[1].y() - c.y())
            p.drawEllipse(c, r, r)
        elif shape.shape_type == "ellipse" and len(screen) >= 3:
            cx, cy, a, b, theta = ellipse_params(
                [(q.x(), q.y()) for q in screen])
            p.save()
            p.translate(cx, cy)
            p.rotate(math.degrees(theta))
            p.drawEllipse(QPointF(0, 0), a, b)
            p.restore()
        elif len(screen) >= 3 and not draft:
            poly = QPolygonF(screen)
            p.drawPolygon(poly)
        elif len(screen) >= 2:
            for i in range(len(screen) - 1):
                p.drawLine(screen[i], screen[i + 1])
            if draft and len(screen) >= 1 and hasattr(self, "_mouse_pos"):
                p.drawLine(screen[-1], self._mouse_pos)

        # vertices
        for i, sp in enumerate(screen):
            is_h = self.h_vertex and self.h_vertex[0] is shape and self.h_vertex[1] == i
            is_sel = (
                shape.selected
                and self._selected_vertex_sticky is not None
                and self._selected_vertex_sticky == i
            )
            r = 6 if (is_h or is_sel) else 3.5
            p.setBrush(QColor(255, 220, 60) if is_sel else QColor(255, 255, 255))
            p.setPen(QPen(color, 1))
            p.drawEllipse(sp, r, r)

        # hover edge highlight + insert hint
        if (
            self.mode == EDIT
            and self.h_edge
            and self.h_edge[0] is shape
            and len(screen) >= 2
        ):
            i = self.h_edge[1]
            a = screen[i]
            b = screen[(i + 1) % len(screen)]
            p.setPen(QPen(QColor(80, 220, 255), 3))
            p.drawLine(a, b)
            mid = QPointF((a.x() + b.x()) / 2, (a.y() + b.y()) / 2)
            p.setBrush(QColor(80, 220, 255))
            p.setPen(QPen(QColor(255, 255, 255), 1))
            p.drawEllipse(mid, 5, 5)

        # label text at first point
        if shape.label and screen:
            p.setPen(QColor(255, 255, 255))
            tag = shape.label
            if shape.layer == "before":
                tag = f"[B] {tag}"
            p.drawText(screen[0] + QPointF(6, -6), tag)

    # ---------- hit testing ----------
    def _vertex_thresh(self) -> float:
        return 8.0 / self.scale

    def find_vertex(self, after_pt: Point) -> Optional[Tuple[Shape, int]]:
        thresh = self._vertex_thresh()
        best = None
        best_d = thresh * thresh
        for shape in reversed(self.shapes):
            pts = self.shape_to_after_points(shape)
            for i, (x, y) in enumerate(pts):
                d = (x - after_pt[0]) ** 2 + (y - after_pt[1]) ** 2
                if d <= best_d:
                    best_d = d
                    best = (shape, i)
        return best

    def find_edge(self, after_pt: Point) -> Optional[Tuple[Shape, int, Point]]:
        """Return (shape, edge_start_index, closest_point_on_edge) in after coords."""
        thresh = self._vertex_thresh()
        best = None
        best_d = thresh
        for shape in reversed(self.shapes):
            if shape.shape_type in (CIRCLE, ELLIPSE):
                continue  # no edge insertion on native circles / ellipses
            pts = self.shape_to_after_points(shape)
            n = len(pts)
            if n < 2:
                continue
            for i in range(n):
                a = pts[i]
                b = pts[(i + 1) % n]
                cx, cy, dist = self._point_to_segment(after_pt, a, b)
                if dist < best_d:
                    best_d = dist
                    best = (shape, i, (cx, cy))
        return best

    @staticmethod
    def _point_to_segment(p: Point, a: Point, b: Point) -> Tuple[float, float, float]:
        ax, ay = a
        bx, by = b
        px, py = p
        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        ab2 = abx * abx + aby * aby
        if ab2 < 1e-12:
            return ax, ay, math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
        cx, cy = ax + t * abx, ay + t * aby
        return cx, cy, math.hypot(px - cx, py - cy)

    def find_shape(self, after_pt: Point) -> Optional[Shape]:
        # point-in-shape, topmost first
        from PySide6.QtGui import QPainterPath

        for shape in reversed(self.shapes):
            pts = self.shape_to_after_points(shape)
            if shape.shape_type == "circle" and len(pts) >= 2:
                c = pts[0]
                r = math.hypot(pts[1][0] - c[0], pts[1][1] - c[1])
                if (after_pt[0] - c[0]) ** 2 + (after_pt[1] - c[1]) ** 2 <= r * r:
                    return shape
                continue
            if shape.shape_type == "ellipse" and len(pts) >= 3:
                cx, cy, a, b, theta = ellipse_params(pts)
                dx, dy = after_pt[0] - cx, after_pt[1] - cy
                ct, st = math.cos(theta), math.sin(theta)
                xr = dx * ct + dy * st
                yr = -dx * st + dy * ct
                if a > 0 and (xr / a) ** 2 + (yr / b) ** 2 <= 1.0:
                    return shape
                continue
            if len(pts) < 3:
                continue
            path = QPainterPath()
            path.moveTo(pts[0][0], pts[0][1])
            for x, y in pts[1:]:
                path.lineTo(x, y)
            path.closeSubpath()
            if path.contains(QPointF(after_pt[0], after_pt[1])):
                return shape
        return None

    def select_shape(self, shape: Optional[Shape], vertex: Optional[int] = None) -> None:
        for s in self.shapes:
            s.selected = False
        self.selected = shape
        self._selected_vertex_sticky = vertex if shape is not None else None
        if shape:
            shape.selected = True
        self.shapes_changed.emit()
        self.update()

    def insert_vertex_on_edge(self, shape: Shape, edge_index: int, after_pt: Point) -> None:
        if self.on_before_edit:
            self.on_before_edit()
        layer_pt = (
            self.after_to_before(*after_pt)
            if shape.layer == "before"
            else after_pt
        )
        shape.points.insert(edge_index + 1, layer_pt)
        self.select_shape(shape, edge_index + 1)
        self.shapes_changed.emit()
        self.update()

    def delete_selected_vertex(self) -> bool:
        """Delete sticky/hovered vertex; returns True if handled."""
        shape = self.selected
        idx = self._selected_vertex_sticky
        if shape is None and self.h_vertex:
            shape, idx = self.h_vertex
        if shape is None or idx is None:
            return False
        if shape.shape_type in (CIRCLE, ELLIPSE):
            self.status_message.emit(
                "圆/椭圆没有可删顶点：拖中心移动、拖边缘点改大小")
            return True
        if len(shape.points) <= 3:
            self.status_message.emit("Polygon needs at least 3 vertices")
            return True
        if self.on_before_edit:
            self.on_before_edit()
        del shape.points[idx]
        self._selected_vertex_sticky = min(idx, len(shape.points) - 1)
        self.select_shape(shape, self._selected_vertex_sticky)
        self.shapes_changed.emit()
        self.update()
        return True

    def edit_selected_label(self) -> None:
        if self.selected is None:
            return
        if self.on_edit_label:
            self.on_edit_label(self.selected)

    # ---------- mouse / wheel ----------
    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        mods = event.modifiers()
        pos = event.position()
        ax, ay = self.screen_to_after(pos)

        # Registration transforms (Before overlay)
        # Avoid Alt+Wheel — on Windows Alt activates the menu bar.
        want_reg = self.dual_mode and (
            self.mode == MOVE_BEFORE or bool(mods & Qt.ControlModifier)
        )
        if want_reg:
            rotate = bool(mods & Qt.ShiftModifier)  # Ctrl+Shift+Wheel or Shift+Wheel in Register
            if rotate:
                self.rotate_before(2.0 if delta > 0 else -2.0, (ax, ay))
            else:
                self.scale_before(1.08 if delta > 0 else 1 / 1.08, (ax, ay))
            event.accept()
            return

        # view zoom toward cursor
        factor = 1.15 if delta > 0 else 1 / 1.15
        old = self.scale
        self.scale = max(0.02, min(40.0, self.scale * factor))
        # keep point under cursor fixed
        ax = (pos.x() - self.offset.x()) / old
        ay = (pos.y() - self.offset.y()) / old
        self.offset = QPointF(pos.x() - ax * self.scale, pos.y() - ay * self.scale)
        self.update()
        event.accept()

    def mousePressEvent(self, event) -> None:
        self.setFocus()
        pos = event.position()
        self._press_pos = pos
        self._last_pos = pos
        self._mouse_pos = pos

        if event.button() == Qt.MiddleButton or (
            event.button() == Qt.LeftButton and event.modifiers() & Qt.ShiftModifier
        ):
            self._panning = True
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() == Qt.RightButton:
            if self.current and len(self.current.points) >= 3:
                self._finish_polygon()
            elif self.current:
                self.current = None
                self.update()
            return

        if event.button() != Qt.LeftButton:
            return

        if self.mode == MOVE_BEFORE and self.dual_mode:
            if self.on_before_edit:
                self.on_before_edit()
            self._moving_before = True
            self.setCursor(Qt.SizeAllCursor)
            return

        if self.mode in (AI_OPTIC, AI_MACULAR):
            # AI always works on the after image in after-image coordinates;
            # press-drag defines a box prompt for the segmenter
            ax, ay = self.screen_to_after(pos)
            self._ai_anchor = (ax, ay)
            self._ai_rubber = None
            label = "optic" if self.mode == AI_OPTIC else "macular"
            self.status_message.emit(
                f"AI 模式：按住并拖出一个框包住{label}，松开后自动分割")
            event.accept()
            return

        after_pt = self.screen_to_after(pos)

        if self.mode == CREATE:
            # close if near first point
            if self.current and len(self.current.points) >= 3:
                first_after = self.shape_to_after_points(self.current)[0]
                if self._dist(after_pt, first_after) < self._vertex_thresh():
                    self._finish_polygon()
                    return
            layer = self.active_layer
            layer_pt = self.screen_to_layer(pos, layer)
            if self.current is None:
                if self.on_before_edit:
                    self.on_before_edit()
                self.current = Shape(label="", points=[layer_pt], layer=layer)
            else:
                self.current.points.append(layer_pt)
            self.update()
            return

        # EDIT mode
        self._did_drag = False
        hit = self.find_vertex(after_pt)
        if hit:
            if self.on_before_edit:
                self.on_before_edit()
            self.select_shape(hit[0], hit[1])
            self.selected_vertex = hit[1]
            self._moving_vertex = True
            if (hit[0].shape_type == "polygon"
                    and len(hit[0].points) >= SMOOTH_DEFORM_MIN_POINTS):
                self.status_message.emit(
                    "平滑变形：拖动这一点，周边的点会平滑跟随")
            return

        edge = self.find_edge(after_pt)
        if edge:
            shape, ei, closest = edge
            self.insert_vertex_on_edge(shape, ei, closest)
            self.selected_vertex = ei + 1
            self._moving_vertex = True
            return

        shape = self.find_shape(after_pt)
        if shape:
            if self.on_before_edit:
                self.on_before_edit()
            self.select_shape(shape, None)
            self._moving_shape = True
            self._shape_start_points = list(shape.points)
            return

        self.select_shape(None)

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        self._mouse_pos = pos
        after_pt = self.screen_to_after(pos)

        if self._panning:
            delta = pos - self._last_pos
            self.offset += delta
            self._last_pos = pos
            self.update()
            return

        if self._moving_before:
            delta = pos - self._last_pos
            self.before_tx += delta.x() / self.scale
            self.before_ty += delta.y() / self.scale
            self._last_pos = pos
            self.update()
            return

        if self._ai_anchor is not None:
            cur = self.screen_to_after(pos)
            self._ai_rubber = (self._ai_anchor, cur)
            self.update()
            return

        if self._moving_vertex and self.selected is not None and self.selected_vertex is not None:
            self._did_drag = True
            layer_pt = self.screen_to_layer(pos, self.selected.layer)
            pts = self.selected.points
            st = self.selected.shape_type
            idx = self.selected_vertex
            if idx == 0 and st in (CIRCLE, ELLIPSE) and len(pts) >= 2:
                # centre handle: translate the whole shape
                dx = layer_pt[0] - pts[0][0]
                dy = layer_pt[1] - pts[0][1]
                pts[0] = layer_pt
                for i in range(1, len(pts)):
                    pts[i] = (pts[i][0] + dx, pts[i][1] + dy)
            elif st == "ellipse" and len(pts) >= 3 and idx in (1, 2):
                pts[idx] = layer_pt
                cx, cy = pts[0]
                if idx == 1:      # major-end handle sets a and rotation
                    a = math.hypot(pts[1][0] - cx, pts[1][1] - cy)
                    theta = math.atan2(pts[1][1] - cy, pts[1][0] - cx)
                else:             # minor-end handle sets b only
                    _, _, a, b, theta = ellipse_params(pts)
                nx, ny = -math.sin(theta), math.cos(theta)
                b = abs((pts[2][0] - cx) * nx + (pts[2][1] - cy) * ny)
                self.selected.points = [
                    (cx, cy),
                    (cx + a * math.cos(theta), cy + a * math.sin(theta)),
                    (cx - b * math.sin(theta), cy + b * math.cos(theta)),
                ]
            elif st == "polygon" and len(pts) >= SMOOTH_DEFORM_MIN_POINTS:
                # soft deformation: the dragged vertex leads, neighbours in a
                # Gaussian window along the outline follow smoothly
                idx = self.selected_vertex
                n = len(pts)
                old = pts[idx]
                dx, dy = layer_pt[0] - old[0], layer_pt[1] - old[1]
                reach = max(8, n // SMOOTH_DEFORM_REACH)
                sigma = reach / 2.5
                for k in range(n):
                    d = min(abs(k - idx), n - abs(k - idx))
                    if d > reach:
                        continue
                    wgt = 1.0 if d == 0 else math.exp(-0.5 * (d / sigma) ** 2)
                    if wgt < 1e-3:
                        continue
                    pts[k] = (pts[k][0] + dx * wgt, pts[k][1] + dy * wgt)
            else:
                pts[idx] = layer_pt
            self._selected_vertex_sticky = self.selected_vertex
            self._last_pos = pos
            self.update()
            return

        if self._moving_shape and self.selected is not None:
            self._did_drag = True
            a0 = self.screen_to_after(self._last_pos)
            a1 = self.screen_to_after(pos)
            if self.selected.layer == "before":
                b0 = self.after_to_before(*a0)
                b1 = self.after_to_before(*a1)
                dx, dy = b1[0] - b0[0], b1[1] - b0[1]
            else:
                dx, dy = a1[0] - a0[0], a1[1] - a0[1]
            self.selected.points = [(x + dx, y + dy) for x, y in self.selected.points]
            self._last_pos = pos
            self.update()
            return

        # hover
        self.h_vertex = self.find_vertex(after_pt) if self.mode == EDIT else None
        self.h_edge = None
        if self.mode == EDIT and self.h_vertex is None:
            edge = self.find_edge(after_pt)
            if edge:
                self.h_edge = (edge[0], edge[1])
        if self.h_vertex:
            self.setCursor(Qt.PointingHandCursor)
        elif self.h_edge:
            self.setCursor(Qt.CrossCursor)
        elif self.mode == CREATE:
            self.setCursor(Qt.CrossCursor)
        elif self.mode == MOVE_BEFORE:
            self.setCursor(Qt.SizeAllCursor)
        elif self.mode == EDIT and self.find_shape(after_pt):
            self.setCursor(Qt.SizeAllCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

        if self.current or self.mode == EDIT or self.crosshair:
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self._panning:
            self._panning = False
            self.unsetCursor()
        if self._moving_before:
            self._moving_before = False
            self.unsetCursor()
            self.before_transform_changed.emit()
            self.shapes_changed.emit()
        if self._ai_anchor is not None:
            cur = self.screen_to_after(event.position())
            (ax, ay) = self._ai_anchor
            self._ai_anchor = None
            rubber, self._ai_rubber = self._ai_rubber, None
            label = "optic" if self.mode == AI_OPTIC else "macular"
            if rubber is not None and math.hypot(cur[0]-ax, cur[1]-ay) > 4:
                x0, x1 = sorted((ax, cur[0]))
                y0, y1 = sorted((ay, cur[1]))
                self.ai_box.emit(x0, y0, x1, y1, label)
            else:
                self.status_message.emit(
                    "AI 模式：请按住并拖出一个框（从视盘/黄斑一侧拖到另一侧）")
            self.update()
            return
        if self._moving_vertex or self._moving_shape:
            if self._moving_vertex and self.selected_vertex is not None:
                self._selected_vertex_sticky = self.selected_vertex
            self._moving_vertex = False
            self._moving_shape = False
            self.selected_vertex = None
            self.shapes_changed.emit()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        if self.current and len(self.current.points) >= 3:
            self._finish_polygon()
            return
        if self.mode == EDIT:
            after_pt = self.screen_to_after(event.position())
            shape = self.find_shape(after_pt) or (
                self.h_vertex[0] if self.h_vertex else None
            )
            if shape:
                self.select_shape(shape)
                self.edit_selected_label()

    def contextMenuEvent(self, event) -> None:
        if self.mode != EDIT or self.current:
            return
        self._show_edit_menu(event.globalPos(), after_pt=self.screen_to_after(event.pos()))

    def _show_edit_menu(self, global_pos, after_pt: Point) -> None:
        hit = self.find_vertex(after_pt)
        edge = None if hit else self.find_edge(after_pt)
        shape = None
        vertex_idx = None
        if hit:
            shape, vertex_idx = hit
            self.select_shape(shape, vertex_idx)
        elif edge:
            shape = edge[0]
            self.select_shape(shape)
        else:
            shape = self.find_shape(after_pt)
            if shape:
                self.select_shape(shape)

        menu = QMenu(self)
        act_edit = QAction("Edit Label…", self)
        act_edit.setEnabled(self.selected is not None)
        act_edit.triggered.connect(self.edit_selected_label)
        menu.addAction(act_edit)

        act_del_v = QAction("Delete Vertex", self)
        act_del_v.setEnabled(
            self.selected is not None
            and self._selected_vertex_sticky is not None
            and len(self.selected.points) > 3
        )
        act_del_v.triggered.connect(self.delete_selected_vertex)
        menu.addAction(act_del_v)

        if edge and hit is None:
            act_ins = QAction("Insert Vertex Here", self)

            def _ins(e=edge):
                self.insert_vertex_on_edge(e[0], e[1], e[2])

            act_ins.triggered.connect(_ins)
            menu.addAction(act_ins)

        menu.addSeparator()
        act_del = QAction("Delete Polygon", self)
        act_del.setEnabled(self.selected is not None)
        act_del.triggered.connect(self.delete_selected)
        menu.addAction(act_del)
        menu.exec(global_pos)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        mods = event.modifiers()
        if key == Qt.Key_Escape:
            self.current = None
            self.select_shape(None)
            self.update()
        elif key in (Qt.Key_Return, Qt.Key_Enter):
            if self.current and len(self.current.points) >= 3:
                self._finish_polygon()
        elif key == Qt.Key_E and mods == Qt.NoModifier:
            self.set_mode(EDIT)
            self.status_message.emit("Mode: Edit")
        elif key == Qt.Key_N and mods == Qt.NoModifier:
            self.set_mode(CREATE)
            self.status_message.emit("Mode: Create")
        elif key == Qt.Key_BracketLeft and self.dual_mode:
            pivot = self.screen_to_after(getattr(self, "_mouse_pos", QPointF(self.width() / 2, self.height() / 2)))
            self.rotate_before(-1.0 if not (mods & Qt.ShiftModifier) else -5.0, pivot)
        elif key == Qt.Key_BracketRight and self.dual_mode:
            pivot = self.screen_to_after(getattr(self, "_mouse_pos", QPointF(self.width() / 2, self.height() / 2)))
            self.rotate_before(1.0 if not (mods & Qt.ShiftModifier) else 5.0, pivot)
        elif key == Qt.Key_F:
            self.fit_view()
        else:
            super().keyPressEvent(event)

    def _finish_polygon(self) -> None:
        if not self.current or len(self.current.points) < 3:
            return
        # label requested by parent via signal — temporarily use placeholder
        self.pending_shape = self.current
        self.current = None
        self.shapes_changed.emit()  # parent may prompt for label
        # If parent doesn't handle, we keep as-is
        if hasattr(self, "request_label"):
            self.request_label.emit()
        self.update()

    def commit_pending(self, label: str) -> None:
        if not hasattr(self, "pending_shape") or self.pending_shape is None:
            return
        shape = self.pending_shape
        shape.label = label
        self.shapes.append(shape)
        self.pending_shape = None
        self.select_shape(shape)
        self.shapes_changed.emit()
        self.update()

    def cancel_pending(self) -> None:
        self.pending_shape = None
        self.update()

    def delete_selected(self) -> None:
        if self.selected is None:
            return
        if self.on_before_edit:
            self.on_before_edit()
        self.shapes = [s for s in self.shapes if s is not self.selected]
        self.selected = None
        self.shapes_changed.emit()
        self.update()

    @staticmethod
    def _dist(a: Point, b: Point) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
