"""E005 runner (Medical-SAM3): region text-prompted inference.

输出: out_regions/medicalsam3/<stem>__<label>.png (255=前景), 最大连通域清理。
"""

import glob
import os
import sys

import cv2
import numpy as np
from PIL import Image

WS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(WS, "Medical-SAM3")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "inference"))

CHECKPOINT = os.path.join(WS, "weights", "checkpoint_2D.pt")
OUT = os.path.join(WS, "out_regions", "medicalsam3")

PROMPTS = {
    "posterior_pole_boundary": ["posterior pole", "posterior pole of retina",
                                "posterior pole of the eye"],
    "central_inner": ["macular region", "posterior pole region",
                      "retinal center"],
}


def largest_component(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == best


def main() -> None:
    from sam3_inference import SAM3Model, resize_mask  # noqa: E402

    os.makedirs(OUT, exist_ok=True)
    sam3 = SAM3Model(confidence_threshold=0.1, device="cuda",
                     checkpoint_path=CHECKPOINT)
    sam3.load_model()

    images = sorted(glob.glob(os.path.join(WS, "images", "*.png")))
    print(f"{len(images)} images")
    for i, img_path in enumerate(images):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        image = np.asarray(Image.open(img_path).convert("RGB"))
        H, W = image.shape[:2]
        state = sam3.encode_image(image)
        for label, prompts in PROMPTS.items():
            out_path = os.path.join(OUT, f"{stem}__{label}.png")
            if os.path.isfile(out_path):
                continue
            best = None
            for p in prompts:
                try:
                    m = sam3.predict_text(state, p)
                except Exception as e:  # noqa: BLE001
                    print(f"  {stem}/{label} prompt={p!r} ERROR {e}")
                    continue
                if m is None or not m.any():
                    continue
                conf = sam3.get_confidence(state)
                if best is None or conf > best[1]:
                    best = (m, conf, p)
            if best is None:
                Image.fromarray(np.zeros((H, W), np.uint8)).save(out_path)
                print(f"[{i+1}/{len(images)}] {stem}/{label}: 无预测")
                continue
            m, conf, p = best
            mask = resize_mask(m, (H, W)) > 0
            comp = largest_component(mask)
            if comp is None:
                Image.fromarray(np.zeros((H, W), np.uint8)).save(out_path)
                print(f"[{i+1}/{len(images)}] {stem}/{label}: 清理后为空")
                continue
            Image.fromarray((comp.astype(np.uint8)) * 255).save(out_path)
            print(f"[{i+1}/{len(images)}] {stem}/{label}: prompt={p!r} "
                  f"conf={conf:.2f} 占比={float(comp.mean()):.3f}")


if __name__ == "__main__":
    main()
