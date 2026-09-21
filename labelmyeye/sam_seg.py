"""MobileSAM ONNX inference: click-to-segment optic disc / macula.

Uses the standard SAM ONNX pair (image encoder + mask decoder):
  encoder: input (1,3,1024,1024) float32, normalized
           output image_embeddings (1,256,64,64)
  decoder: inputs image_embeddings, point_coords, point_labels, mask_input,
           has_mask_input, orig_im_size
           outputs masks, iou_predictions, low_res_masks

Usage:
    seg = SamSegmenter(encoder_onnx, decoder_onnx)
    seg.set_image(pil_image)          # runs the encoder (cache per call)
    mask, info = seg.segment([(x, y)], [(1,)])   # positive click in image px
    circle = fit_circle_from_mask(mask)          # -> (cx, cy, r, quality)

The encoder is the expensive part (~1-2 s CPU); call set_image once per
image (e.g. from a worker thread), then segment() is fast (<100 ms).
"""

from __future__ import annotations

import math

import numpy as np

ENCODER_SIZE = 1024
PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def _ort_session(path: str):
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise RuntimeError(
            "需要 onnxruntime 才能运行 AI 分割：pip install onnxruntime") from e
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


class SamSegmenter:
    def __init__(self, encoder_path: str, decoder_path: str):
        self.encoder = _ort_session(encoder_path)
        self.decoder = _ort_session(decoder_path)
        self._img_size = None          # (w, h) of current image
        self._scale = None             # image px -> 1024-space
        self._embedding = None

    # ------------------------------------------------------------- encoder
    def set_image(self, pil_image) -> None:
        """Encode one image (expensive). pil_image: PIL Image (RGB)."""
        w, h = pil_image.size
        self._img_size = (w, h)
        self._scale = ENCODER_SIZE / float(max(w, h))
        tw = max(1, int(round(w * self._scale)))
        th = max(1, int(round(h * self._scale)))
        arr = np.asarray(
            pil_image.convert("RGB").resize((tw, th)), dtype=np.float32)
        arr = (arr - PIXEL_MEAN) / PIXEL_STD
        arr = arr.transpose(2, 0, 1)[None]               # 1x3xthxtw
        pad = np.zeros((1, 3, ENCODER_SIZE, ENCODER_SIZE), dtype=np.float32)
        pad[:, :, :th, :tw] = arr
        outs = self.encoder.run(None, {self.encoder.get_inputs()[0].name: pad})
        self._embedding = outs[0]
        self._thumb_size = (tw, th)

    def clear_image(self) -> None:
        self._img_size = None
        self._scale = None
        self._embedding = None

    @property
    def ready(self) -> bool:
        return self._embedding is not None

    # ------------------------------------------------------------- decoder
    def segment(self, points, labels):
        """Generic prompt: points/labels lists (label: 1 pos, 0 neg), padded to
        the graph's static slot count with -1 (not-a-point). For boxes prefer
        segment_box(). Returns (mask_bool_at_analysis_scale, info dict)."""
        if self._embedding is None:
            raise RuntimeError("set_image() must be called before segment()")
        coords = np.asarray(points, dtype=np.float32).reshape(1, -1, 2)
        lbls = np.asarray(labels, dtype=np.float32).reshape(1, -1)
        n_slots = int(self.decoder.get_inputs()[1].shape[1] or 1)
        if lbls.shape[1] > n_slots:
            raise ValueError("too many prompt points for the exported model")
        while lbls.shape[1] < n_slots:
            coords = np.concatenate([coords, [[[-1.0, -1.0]]]], axis=1)
            lbls = np.concatenate([lbls, [[-1.0]]], axis=1)
        n_real = int(np.asarray(points).reshape(-1, 2).shape[0])
        coords[:, :n_real] *= self._scale
        feed = self._decoder_feed(coords, lbls, 0.0)
        logits, ious = self.decoder.run(None, feed)[:2]
        return self._select_mask(np.asarray(logits), np.asarray(ious))

    def segment_box(self, box):
        """box: (x0, y0, x1, y1) in original image px (any corner order).

        Returns (mask_bool_at_analysis_scale, info dict).
        """
        if self._embedding is None:
            raise RuntimeError("set_image() must be called before segment()")
        x0, y0, x1, y1 = (float(v) * self._scale for v in box)
        coords = np.array([[[x0, y0], [x1, y1]]], dtype=np.float32)
        lbls = np.array([[2.0, 3.0]], dtype=np.float32)
        n_slots = int(self.decoder.get_inputs()[1].shape[1] or 1)
        while lbls.shape[1] < n_slots:
            coords = np.concatenate([coords, [[[-1.0, -1.0]]]], axis=1)
            lbls = np.concatenate([lbls, [[-1.0]]], axis=1)
        feed = self._decoder_feed(coords, lbls, 0.0)
        logits, ious = self.decoder.run(None, feed)[:2]
        return self._select_mask(np.asarray(logits), np.asarray(ious))

    def _decoder_feed(self, coords, lbls, has_mask: float):
        tw, th = self._thumb_size
        candidates = {
            "image_embeddings": self._embedding,
            "point_coords": coords.astype(np.float32),
            "point_labels": lbls.astype(np.float32),
            "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
            "has_mask_input": np.array([has_mask], dtype=np.float32),
            "orig_im_size": np.array([th, tw], dtype=np.float32),
        }
        names = {i.name for i in self.decoder.get_inputs()}
        return {k: v for k, v in candidates.items() if k in names}

    def _select_mask(self, logits, ious):
        """logits: (1, T, 256, 256); token 0 = none-token, 1..3 multimask.

        Picks the highest-IoU non-none candidate whose area is not degenerate
        (a stray huge 'whole image' mask), upsamples bilinearly.
        """
        ious = ious.reshape(-1)
        logits = logits[0]                                # T x 256 x 256
        n = logits.shape[0]
        order = list(range(1, n)) if n > 1 else [0]
        best_k, best_score = None, -1.0
        for k in order:
            frac = float((logits[k] > 0.0).mean())
            if frac > 0.6:                                # degenerate blob
                continue
            if float(ious[k]) > best_score:
                best_k, best_score = k, float(ious[k])
        if best_k is None:                                # all degenerate
            best_k = order[0]
        low = logits[best_k]
        from PIL import Image
        tw, th = self._thumb_size
        up = Image.fromarray(low.astype(np.float32), mode="F").resize(
            (tw, th), Image.BILINEAR)
        m = np.asarray(up) > 0.0                          # th x tw bool
        return m, {"iou": float(ious[best_k]), "scale": self._scale,
                   "thumb": (tw, th), "token": best_k}


# --------------------------------------------------------------- circle fit


def _largest_component(mask: np.ndarray):
    """Largest 4-connected component of a bool mask.

    Run-based connected component labelling (fast on fundus-scale masks).
    Returns (mask_bool_of_component, touches_border: bool).
    """
    h, w = mask.shape
    parent = list(range(1))          # label 0 unused

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:     # path compression
            parent[a], a = root, parent[a]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    runs = {}                        # label -> [xmin, xmax, ymin, ymax, count]
    run_rows = []                    # per row: list of (label, x0, x1)
    prev = []
    touches = False
    for y in range(h):
        row = mask[y]
        if not row.any():
            prev = []
            continue
        xs = np.flatnonzero(np.diff(np.concatenate(
            [[0], row.view(np.int8), [0]])))
        cur = []
        for i in range(0, len(xs), 2):
            x0, x1 = int(xs[i]), int(xs[i + 1] - 1)
            lbl = len(parent)
            parent.append(lbl)
            for (pl, px0, px1) in prev:
                if px0 <= x1 and px1 >= x0:      # 4/8-connectivity via overlap
                    union(lbl, pl)
            cur.append((lbl, x0, x1))
            if y == 0 or y == h - 1 or x0 == 0 or x1 == w - 1:
                touches = True
        # merge run labels to canonical later; store row runs
        run_rows.append((y, cur))
        prev = cur

    if len(parent) <= 1:
        return None, False
    # largest component by pixel count
    counts = {}
    for y, cur in run_rows:
        for lbl, x0, x1 in cur:
            r = find(lbl)
            counts[r] = counts.get(r, 0) + (x1 - x0 + 1)
    if not counts:
        return None, False
    best = max(counts, key=counts.get)
    out = np.zeros_like(mask)
    for y, cur in run_rows:
        for lbl, x0, x1 in cur:
            if find(lbl) == best:
                out[y, x0:x1 + 1] = True
    return out, touches


def fit_circle_from_mask(mask: np.ndarray):
    """Fit a circle to the largest mask component.

    Returns dict(cx, cy, r, area, iou=None, warnings=[...]) in mask
    coordinates (callers rescale to image px).
    """
    comp, touches = _largest_component(mask)
    if comp is None:
        return None
    ys, xs = np.nonzero(comp)
    area = float(len(xs))
    cx, cy = float(xs.mean()), float(ys.mean())
    r = math.sqrt(area / math.pi)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bbox_fill = area / max((x1 - x0 + 1) * (y1 - y0 + 1), 1)
    r_bbox = math.hypot(x1 - x0, y1 - y0) / 2.0
    warnings = []
    if touches:
        warnings.append("mask touches image border")
    if bbox_fill < 0.55:
        warnings.append("mask shape not disc-like (fill %.2f)" % bbox_fill)
    if r_bbox <= 0 or r / r_bbox < 0.45:
        warnings.append("mask elongated, circle fit unreliable")
    return {"cx": cx, "cy": cy, "r": r, "area": area,
            "bbox": (x0, y0, x1, y1), "warnings": warnings}
