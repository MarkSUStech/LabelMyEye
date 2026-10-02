"""Export LabelMe JSON annotations to binary 0/1 mask images + overlay views.

For each JSON: writes one 0/255 PNG per label (all shapes of that label
unioned), plus one overlay PNG (colored translucent masks on the original
image). Labels are whatever appears in the JSONs (optic, macular,
posterior_pole_boundary, central_inner, 裂孔, ...).

Usage:
  python scripts/export_masks.py <folder-or-json> [--out DIR]

Output layout: <out>/<json-stem>/<json-stem>__<label>.png + __overlay.png
Never writes next to the source images (safe for read-only network shares).
"""

from __future__ import annotations

import argparse
import colorsys
import glob
import os
import sys
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DR_PALETTE = {
    "optic": (237, 96, 97),
    "macular": (68, 130, 217),
    "macular_area": (60, 167, 103),
    "posterior_pole_boundary": (242, 153, 74),
    "central_inner": (155, 81, 224),
}
LESION_PALETTE = {          # 常见病灶固定配色
    "裂孔": (255, 60, 200),
    "hemorrhage": (200, 30, 30),
    "hard_exudate": (250, 230, 90),
    "soft_exudate": (180, 220, 90),
    "microaneurysm": (150, 60, 200),
    "neovascularization": (60, 220, 220),
    "laser_spot": (255, 255, 255),
}


def label_color(name: str):
    if name in DR_PALETTE:
        return DR_PALETTE[name]
    if name in LESION_PALETTE:
        return LESION_PALETTE[name]
    h = zlib.crc32(name.encode("utf-8")) % 360
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, 0.7, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


def rasterize_shapes(shapes, size):
    """Rasterize each shape (polygon/circle/ellipse) into one boolean mask."""
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    for s in shapes:
        st = s.get("shape_type", "polygon")
        pts = s.get("points", [])
        if st == "circle" and len(pts) >= 2:
            (cx, cy), (ex, ey) = pts[0], pts[1]
            r = ((ex - cx) ** 2 + (ey - cy) ** 2) ** 0.5
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
        elif st == "ellipse" and len(pts) >= 3:
            (cx, cy), (x1, y1), (x2, y2) = pts[0], pts[1], pts[2]
            a = ((x1 - cx) ** 2 + (y1 - cy) ** 2) ** 0.5
            th = math.atan2(y1 - cy, x1 - cx)
            b = abs((x2 - cx) * -math.sin(th) + (y2 - cy) * math.cos(th))
            ts = np.linspace(0, 2 * math.pi, 120)
            poly = [(cx + a * np.cos(t) * math.cos(th) - b * np.sin(t) * math.sin(th),
                     cy + a * np.cos(t) * math.sin(th) + b * np.sin(t) * math.cos(th))
                    for t in ts]
            d.polygon(poly, fill=255)
        elif len(pts) >= 3:
            d.polygon([tuple(p) for p in pts], fill=255)
    return mask


import math  # noqa: E402  (used in ellipse branch)


def export_json(json_path: str, out_dir: str, img_dir: str | None) -> int:
    import json
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    stem = os.path.splitext(os.path.basename(json_path))[0]
    W = int(data.get("imageWidth") or 0)
    H = int(data.get("imageHeight") or 0)

    # 找原图（仅读取）
    img = None
    cand_dirs = [img_dir] if img_dir else []
    cand_dirs += [os.path.dirname(json_path)]
    ip = data.get("imagePath") or ""
    for cdir in cand_dirs:
        if not cdir:
            continue
        for cand in (os.path.join(cdir, ip) if ip else None,
                     os.path.join(cdir, stem + ".png"),
                     os.path.join(cdir, stem + ".jpg")):
            if cand and os.path.isfile(cand):
                img = Image.open(cand).convert("RGB")
                break
        if img:
            break
    if img is not None:
        W, H = img.size
    elif not (W and H):
        print(f"  跳过 {stem}: 找不到图像且 JSON 无尺寸")
        return 0

    jdir = os.path.join(out_dir, stem)
    os.makedirs(jdir, exist_ok=True)
    by_label: dict[str, list] = {}
    for s in data.get("shapes", []):
        by_label.setdefault(s.get("label", "unlabeled"), []).append(s)

    n_masks = 0
    if img is not None:
        overlay = img.copy()
        od = ImageDraw.Draw(overlay, "RGBA")
    for label, shapes in sorted(by_label.items()):
        m = rasterize_shapes(shapes, (W, H))
        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in label)
        m.save(os.path.join(jdir, f"{stem}__{safe}.png"))
        n_masks += 1
        if img is not None:
            col = label_color(label)
            arr = np.asarray(m)
            tint = np.zeros((H, W, 4), np.uint8)
            tint[..., :3] = col
            tint[..., 3] = (arr > 0) * 90
            ov = Image.fromarray(tint, "RGBA")
            overlay = Image.alpha_composite(overlay.convert("RGBA"), ov).convert("RGB")
            od = ImageDraw.Draw(overlay, "RGBA")
            pts = shapes[0].get("points", [])
            if pts:
                od.text((pts[0][0] + 4, pts[0][1] - 16), label, fill=col,
                        font=ImageFont.load_default())
    if img is not None:
        overlay.save(os.path.join(jdir, f"{stem}__overlay.png"))
    return n_masks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="JSON 文件或包含 JSON 的文件夹")
    ap.add_argument("--out", default=None, help="输出目录（默认 output/masks/<源目录名>）")
    ap.add_argument("--img-dir", default=None, help="图片所在目录（默认与 JSON 同目录）")
    args = ap.parse_args()

    src = args.source
    if os.path.isdir(src):
        jsons = sorted(glob.glob(os.path.join(src, "*.json")))
        out_dir = args.out or os.path.join("output", "masks", os.path.basename(src.rstrip("/\\")))
    else:
        jsons = [src]
        out_dir = args.out or os.path.join("output", "masks", "single")
    os.makedirs(out_dir, exist_ok=True)

    total = 0
    for j in jsons:
        n = export_json(j, out_dir, args.img_dir)
        total += n
        print(f"{os.path.basename(j)}: {n} masks")
    print(f"\n完成: {len(jsons)} 个 JSON, {total} 张 mask → {out_dir}")


if __name__ == "__main__":
    main()
