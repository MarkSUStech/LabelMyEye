"""Benchmark disk/macula prediction models against hand annotations.

Ground truth: LabelMe JSONs with hand-drawn optic/macular (polygon or native
circle). Metric: raster IoU between predicted disk and hand disk + center
distance (in disc radii).

Models (--models, comma separated):
  mobilesam     本仓库自带 (box prompt = 手绘圆外扩 1.3x)      ← 可直接运行
  medsam3       text-guided, 需 GPU + SAM3 base + LoRA 权重     ← 见 --help 输出
  medclipsamv2  text-guided, 需 MedCLIP-SAMv2 仓库 + checkpoints

Usage:
  python scripts/benchmark_models.py --data scripts/calibration_data \
      --models mobilesam [--out output/benchmark]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from labelmyeye.auto_annotate import circle_from_polygon  # noqa: E402
from calibrate_from_jsons import fit_circle_rms  # noqa: E402

SETUP_NOTES = {
    "medsam3": (
        "MedSAM3 (text-guided, SAM3+LoRA):\n"
        "  1. 在有 GPU 的机器: git clone https://github.com/Joey-S-Liu/MedSAM3\n"
        "  2. 安装 SAM3: https://github.com/facebookresearch/sam3 (需下 Meta 底座权重)\n"
        "  3. 下载 LoRA: https://huggingface.co/lal-Joey/MedSAM3_v1\n"
        "  4. 用仓库的 inference_lora.py 以文本概念(如 'optic disc'/'macula lutea')推理,\n"
        "     把每张图的预测 mask PNG 存到 output/benchmark/<model>/<stem>__<label>.png,\n"
        "     然后重跑本脚本 --models medsam3 --pred-dir output/benchmark/medsam3 即可自动计分"),
    "medclipsamv2": (
        "MedCLIP-SAMv2 (text-guided, CLIP+SAM, MedIA 2025):\n"
        "  1. git clone https://github.com/HealthX-Lab/MedCLIP-SAMv2\n"
        "  2. 按 README 下载 checkpoints (HF/Drive)\n"
        "  3. 以文本 prompt 'optic disc' / 'macula' 推理, 预测 mask 存到\n"
        "     output/benchmark/medclipsamv2/<stem>__<label>.png,\n"
        "     然后重跑本脚本 --models medclipsamv2 --pred-dir ... 自动计分"),
}


def disk_mask(shape, c, r):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    return (xx - c[0]) ** 2 + (yy - c[1]) ** 2 <= r * r


def iou_disk(c1, r1, c2, r2):
    """Raster IoU of two disks (robust for any containment configuration)."""
    x0 = min(c1[0] - r1, c2[0] - r2) - 2
    y0 = min(c1[1] - r1, c2[1] - r2) - 2
    w = int(max(c1[0] + r1, c2[0] + r2) - x0) + 4
    h = int(max(c1[1] + r1, c2[1] + r2) - y0) + 4
    if w * h > 40_000_000:            # 防御性上限（超大图分块太复杂，退化为解析式上限场景）
        return 0.0
    yy, xx = np.mgrid[0:h, 0:w]
    X = xx + x0
    Y = yy + y0
    m1 = (X - c1[0]) ** 2 + (Y - c1[1]) ** 2 <= r1 * r1
    m2 = (X - c2[0]) ** 2 + (Y - c2[1]) ** 2 <= r2 * r2
    union = float((m1 | m2).sum())
    return float((m1 & m2).sum()) / union if union else 0.0


def load_gt(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    S = {s["label"]: s for s in data.get("shapes", [])}
    out = {}
    for label in ("optic", "macular"):
        if label in S:
            fit = circle_from_polygon(S[label]["points"]) or \
                (fit_circle_rms(np.array(S[label]["points"]))[0])
            out[label] = fit          # (cx, cy, r)
    return out


class MobileSamModel:
    name = "mobilesam"
    prompt = "box (手绘圆外扩 1.3x)"

    def __init__(self):
        from labelmyeye import auto_annotate as aa
        self._aa = aa
        self._seg = None
        self._key = None

    def predict(self, image, image_path, label, gt):
        if self._seg is None:
            self._seg = self._aa.get_segmenter()
        key = f"{image_path}|{os.path.getmtime(image_path)}"
        if key != self._key:
            self._seg.set_image(image)
            self._key = key
        cx, cy, r = gt
        box = (cx - r * 1.3, cy - r * 1.3, cx + r * 1.3, cy + r * 1.3)
        mask, info = self._seg.segment_box(box)
        from labelmyeye.sam_seg import fit_circle_from_mask
        fit = fit_circle_from_mask(mask)
        if fit is None:
            return None
        s = info["scale"]
        return (fit["cx"] / s, fit["cy"] / s, fit["r"] / s)


class PredMaskModel:
    """从 output/benchmark/<model>/ 读取已导出的预测 mask PNG 计分。

    文件名约定: <stem>__<label>.png (255=前景)。由各模型的官方推理脚本生成
    (setup notes 里有步骤)。
    """

    def __init__(self, name, pred_dir):
        self.name = name
        self.pred_dir = pred_dir

    def predict(self, image, image_path, label, gt):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        p = os.path.join(self.pred_dir, stem, f"{stem}__{label}.png")
        if not os.path.isfile(p):
            p2 = os.path.join(self.pred_dir, f"{stem}__{label}.png")
            p = p2 if os.path.isfile(p2) else None
        if p is None:
            return None
        m = np.asarray(Image.open(p).convert("L")) > 127
        ys, xs = np.nonzero(m)
        if len(xs) < 10:
            return None
        c = (float(xs.mean()), float(ys.mean()))
        r = math.sqrt(float(m.sum()) / math.pi)
        return (c[0], c[1], r)


def main() -> None:
    ap = argparse.ArgumentParser(
        epilog="\n\n".join(SETUP_NOTES.values()), formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join("scripts", "calibration_data"),
                    help="含手绘 JSON+图片的文件夹")
    ap.add_argument("--models", default="mobilesam")
    ap.add_argument("--pred-dir", default=None,
                    help="预导出 mask 根目录 (medsam3/medclipsamv2 计分用)")
    ap.add_argument("--out", default=os.path.join("output", "benchmark"))
    args = ap.parse_args()

    models = []
    for name in args.models.split(","):
        name = name.strip()
        if name == "mobilesam":
            models.append(MobileSamModel())
        elif name in ("medsam3", "medclipsamv2"):
            pd = args.pred_dir or os.path.join(args.out, name)
            models.append(PredMaskModel(name, pd))
        else:
            print(f"未知模型 {name}, 跳过")
    if not models:
        print("没有可运行的模型")
        return

    jsons = sorted(glob.glob(os.path.join(args.data, "*.json")))
    rows = []
    for j in jsons:
        stem = os.path.splitext(os.path.basename(j))[0]
        gt = load_gt(j)
        if "optic" not in gt:
            continue
        img = None
        for ext in (".png", ".jpg", ".jpeg"):
            p = os.path.join(args.data, stem + ext)
            if os.path.isfile(p):
                img = Image.open(p).convert("RGB")
                break
        if img is None:
            continue
        for model in models:
            for label, gt_c in gt.items():
                try:
                    pred = model.predict(img, os.path.abspath(
                        os.path.join(args.data, stem + ".png")) if
                        os.path.isfile(os.path.join(args.data, stem + ".png"))
                        else os.path.abspath(os.path.join(
                            args.data, stem + ".jpg")), label, gt_c)
                except Exception as e:  # noqa: BLE001
                    pred = None
                    print(f"  {model.name}/{stem}/{label}: ERROR {e}")
                if pred is None:
                    rows.append({"model": model.name, "image": stem,
                                 "label": label, "iou": None, "center_Rd": None})
                    continue
                v = iou_disk((pred[0], pred[1]), pred[2],
                             (gt_c[0], gt_c[1]), gt_c[2])
                Rd = gt["optic"][2]
                dc = math.hypot(pred[0] - gt_c[0], pred[1] - gt_c[1]) / Rd
                rows.append({"model": model.name, "image": stem, "label": label,
                             "iou": round(v, 4), "center_Rd": round(dc, 3)})

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)

    print(f"\n{'='*62}\n逐图明细已写入 {args.out}/results.json\n{'='*62}")
    for model in models:
        for label in ("optic", "macular"):
            vs = [r["iou"] for r in rows
                  if r["model"] == model.name and r["label"] == label
                  and r["iou"] is not None]
            ds = [r["center_Rd"] for r in rows
                  if r["model"] == model.name and r["label"] == label
                  and r["center_Rd"] is not None]
            if not vs:
                continue
            vs_s, ds_s = sorted(vs), sorted(ds)
            n = len(vs_s)
            print(f"{model.name:14s} {label:8s} n={n}  "
                  f"IoU 中位={vs_s[n//2]:.3f} (min {vs_s[0]:.3f} / max {vs_s[-1]:.3f})  "
                  f"中心距中位={ds_s[n//2]:.2f} Rd")


if __name__ == "__main__":
    main()
