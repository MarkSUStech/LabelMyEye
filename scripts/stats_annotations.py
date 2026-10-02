"""统计标注进度：图片数量、JSON 数量、每个标签的形状数。

Usage:
  python scripts/stats_annotations.py <folder> [folder2 ...] [--csv output/stats.csv]

只读取源文件夹，CSV 写到本地 output/，网络盘安全。
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def stats_folder(folder: str) -> dict:
    names = sorted(os.listdir(folder))
    images = [n for n in names if n.lower().endswith(IMG_EXTS)]
    jsons = [n for n in names if n.lower().endswith(".json")
             and not n.startswith("._")]   # 过滤 macOS AppleDouble 垃圾文件
    label_counter: Counter = Counter()
    per_image_counts = []
    empty_jsons = 0
    for j in jsons:
        try:
            with open(os.path.join(folder, j), encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠ 无法读取 {j}: {e}")
            continue
        shapes = data.get("shapes", [])
        labels = [s.get("label", "?") for s in shapes]
        label_counter.update(labels)
        if not labels:
            empty_jsons += 1
        per_image_counts.append((os.path.splitext(j)[0], len(labels)))
    annotated = {os.path.splitext(j)[0] for j in jsons}
    unannotated = [n for n in images if os.path.splitext(n)[0] not in annotated]
    return {
        "folder": folder,
        "images": len(images),
        "jsons": len(jsons),
        "empty_jsons": empty_jsons,
        "unannotated": len(unannotated),
        "labels": label_counter,
        "total_shapes": sum(label_counter.values()),
        "per_image": per_image_counts,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--csv", default=None, help="把逐图统计写到该 CSV（本地路径）")
    args = ap.parse_args()

    all_rows = []
    for folder in args.folders:
        st = stats_folder(folder)
        print(f"\n===== {folder} =====")
        print(f"  图片总数      : {st['images']}")
        print(f"  已标注(JSON)  : {st['jsons']}  (空标注 {st['empty_jsons']}, "
              f"未标注图片 {st['unannotated']})")
        print(f"  标注形状总数  : {st['total_shapes']}")
        if st["labels"]:
            width = max(len(k) for k in st["labels"])
            for label, cnt in st["labels"].most_common():
                print(f"    {label:<{width}} : {cnt}")
        for img, cnt in st["per_image"]:
            all_rows.append({"folder": folder, "image": img, "shapes": cnt})

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["folder", "image", "shapes"])
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n逐图统计已写入 {args.csv}")


if __name__ == "__main__":
    main()
