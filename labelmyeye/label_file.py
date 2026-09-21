"""LabelMe JSON load / save."""

from __future__ import annotations

import base64
import io
import json
import os
from typing import List, Optional, Tuple

from PIL import Image

from labelmyeye.shape import Shape

LABELME_VERSION = "5.10.1"


def _pil_to_qimage_bytes(path: str) -> Tuple[bytes, int, int]:
    img = Image.open(path)
    img = img.convert("RGB")
    w, h = img.size
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), w, h


def load_labelme(path: str, layer: str = "after") -> Tuple[List[Shape], dict]:
    """Load LabelMe JSON. Returns (shapes, meta).

    meta keys: imagePath, imageHeight, imageWidth, imageData (optional),
               flags, version, source_path
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    supported = ("polygon", "circle", "ellipse")
    shapes = [
        Shape.from_labelme(s, layer=layer)
        for s in data.get("shapes", [])
        if s.get("shape_type", "polygon") in supported
    ]
    meta = {
        "version": data.get("version", LABELME_VERSION),
        "flags": data.get("flags") or {},
        "imagePath": data.get("imagePath", ""),
        "imageHeight": data.get("imageHeight"),
        "imageWidth": data.get("imageWidth"),
        "imageData": data.get("imageData"),
        "source_path": path,
    }
    return shapes, meta


def resolve_image_path(json_path: str, image_path: Optional[str]) -> Optional[str]:
    if not image_path:
        return None
    if os.path.isabs(image_path) and os.path.isfile(image_path):
        return image_path
    base = os.path.dirname(os.path.abspath(json_path))
    candidate = os.path.join(base, image_path)
    if os.path.isfile(candidate):
        return candidate
    # same stem as json
    stem = os.path.splitext(os.path.basename(json_path))[0]
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        p = os.path.join(base, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def image_from_meta(meta: dict, json_path: str) -> Tuple[Optional[str], Optional[Image.Image]]:
    """Return (filesystem_path, PIL image). Prefers external file over imageData."""
    img_path = resolve_image_path(json_path, meta.get("imagePath"))
    if img_path and os.path.isfile(img_path):
        return img_path, Image.open(img_path).convert("RGB")
    data = meta.get("imageData")
    if data:
        raw = base64.b64decode(data)
        return None, Image.open(io.BytesIO(raw)).convert("RGB")
    return None, None


def save_labelme(
    path: str,
    shapes: List[Shape],
    image_path: str,
    image_height: int,
    image_width: int,
    flags: Optional[dict] = None,
    embed_image: bool = False,
) -> None:
    payload = {
        "version": LABELME_VERSION,
        "flags": flags or {},
        "shapes": [s.to_labelme() for s in shapes],
        "imagePath": os.path.basename(image_path) if image_path else "",
        "imageData": None,
        "imageHeight": int(image_height),
        "imageWidth": int(image_width),
    }
    if embed_image and image_path and os.path.isfile(image_path):
        raw, _, _ = _pil_to_qimage_bytes(image_path)
        payload["imageData"] = base64.b64encode(raw).decode("utf-8")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_project(path: str, project: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(project, f, ensure_ascii=False, indent=2)


def load_project(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
