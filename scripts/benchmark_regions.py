"""Region-level benchmark: posterior_pole_boundary & central_inner.

与 benchmark_models.py 的区别: 这两个结构是**区域**(不规则多边形)而非圆,
GT = 手绘多边形栅格化的 mask; 评测 = 预测 mask vs GT mask 的栅格 IoU / Dice,
不再做圆拟合。

模型:
  mobilesam      框提示 = GT 多边形 bbox 外扩 1.3×(本机 CPU 可跑)
  medsam3 / medicalsam3 / medclipsamv2
                 读取服务器预导出的预测 mask (--pred-dir/<模型名>/<stem>__<label>.png)

Usage:
  python scripts/benchmark_regions.py --data scripts/calibration_data \
      --models mobilesam [--labels posterior_pole_boundary central_inner]
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

REGIONS = ("posterior_pole_boundary", "central_inner")


def gt_mask_from_shapes(shapes, size):
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    for s in shapes:
        pts = s.get("points", [])
        if len(pts) >= 3:
            d.polygon([tuple(p) for p in pts], fill=255)
    return np.asarray(m) > 127


def mask_iou_dice(pred, gt):
    inter = float((pred & gt).sum())
    union = float((pred | gt).sum())
    iou = inter / union if union else 0.0
    dice = 2 * inter / (float(pred.sum()) + float(gt.sum())) \
        if (pred.sum() + gt.sum()) else 0.0
    return iou, dice


class MobileSamRegion:
    name = "mobilesam"
    prompt = "GT 多边形 bbox 外扩 1.3×"

    def __init__(self):
        from labelmyeye import auto_annotate as aa
        self._aa = aa
        self._seg = None
        self._key = None

    def predict(self, image, image_path, label, gt_mask):
        ys, xs = np.nonzero(gt_mask)
        cx, cy = xs.mean(), ys.mean()
        r = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2
        box = (cx - r * 1.3, cy - r * 1.3, cx + r * 1.3, cy + r * 1.3)
        if self._seg is None:
            self._seg = self._aa.get_segmenter()
        key = f"{image_path}|{os.path.getmtime(image_path)}"
        if key != self._key:
            self._seg.set_image(image)
            self._key = key
        # 大区域任务: 放宽"大 mask=退化"过滤(真区域可占半图), 取最高置信 token
        mask, info = self._seg.segment_box(box, max_frac=0.95)
        from labelmyeye.sam_seg import _largest_component
        comp, _touches = _largest_component(mask)   # 去除 256 上采样的散点噪声
        if comp is None:
            return None
        m = Image.fromarray((comp * 255).astype(np.uint8)).resize(
            (image.width, image.height), Image.NEAREST)
        return np.asarray(m) > 127


class PredMaskRegion:
    def __init__(self, name, pred_dir):
        self.name = name
        self.pred_dir = pred_dir

    def predict(self, image, image_path, label, gt_mask):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        for p in (os.path.join(self.pred_dir, stem, f"{stem}__{label}.png"),
                  os.path.join(self.pred_dir, f"{stem}__{label}.png")):
            if os.path.isfile(p):
                m = np.asarray(Image.open(p).convert("L")) > 127
                if m.shape != (image.height, image.width):
                    m = np.asarray(Image.fromarray(
                        (m * 255).astype(np.uint8)).resize(
                        (image.width, image.height), Image.NEAREST)) > 127
                return m
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("scripts", "calibration_data"))
    ap.add_argument("--models", default="mobilesam")
    ap.add_argument("--labels", nargs="+", default=list(REGIONS))
    ap.add_argument("--out", default=os.path.join("output", "benchmark_regions"))
    args = ap.parse_args()

    models = []
    for name in args.models.split(","):
        name = name.strip()
        if name == "mobilesam":
            models.append(MobileSamRegion())
        else:
            models.append(PredMaskRegion(
                name, os.path.join(args.out, name)))

    jsons = sorted(glob.glob(os.path.join(args.data, "*.json")))
    rows = []
    for j in jsons:
        stem = os.path.splitext(os.path.basename(j))[0]
        img_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            p = os.path.join(args.data, stem + ext)
            if os.path.isfile(p):
                img_path = p
                break
        if img_path is None:
            continue
        with open(j, encoding="utf-8") as f:
            data = json.load(f)
        image = Image.open(img_path).convert("RGB")
        for label in args.labels:
            shapes = [s for s in data.get("shapes", []) if s["label"] == label]
            if not shapes:
                continue
            gt = gt_mask_from_shapes(shapes, image.size)
            for model in models:
                try:
                    pred = model.predict(image, img_path, label, gt)
                except Exception as e:  # noqa: BLE001
                    print(f"  {model.name}/{stem}/{label}: ERROR {e}")
                    pred = None
                if pred is None:
                    rows.append({"model": model.name, "image": stem,
                                 "label": label, "iou": None, "dice": None})
                    continue
                iou, dice = mask_iou_dice(pred, gt)
                rows.append({"model": model.name, "image": stem, "label": label,
                             "iou": round(iou, 4), "dice": round(dice, 4)})
                print(f"  {model.name}/{stem}/{label}: IoU={iou:.3f} Dice={dice:.3f}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)

    print(f"\n{'='*62}")
    for model in models:
        for label in args.labels:
            vs = sorted(r["iou"] for r in rows if r["model"] == model.name
                        and r["label"] == label and r["iou"] is not None)
            if not vs:
                continue
            n = len(vs)
            print(f"{model.name:14s} {label:24s} n={n}  IoU 中位={vs[n//2]:.3f} "
                  f"(min {vs[0]:.3f} / max {vs[-1]:.3f})")


if __name__ == "__main__":
    main()
