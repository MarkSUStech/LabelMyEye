"""Server-side runner: Medical-SAM3 text-prompted inference on fundus images.

Reads  images/*.png, writes  out/<stem>__<label>.png (255=foreground).
Label → prompt candidates; the highest-confidence non-empty mask wins.
Run on the Titan server inside ~/labelmyeye_bench with the venv activated:

    CUDA_VISIBLE_DEVICES=0 python run_bench.py
"""

import glob
import os
import sys

import numpy as np
from PIL import Image

WS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(WS, "Medical-SAM3")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "inference"))

from sam3_inference import SAM3Model, resize_mask  # noqa: E402

CHECKPOINT = os.path.join(WS, "weights", "checkpoint_2D.pt")
OUT = os.path.join(WS, "out")
PROMPTS = {
    "optic": ["optic disc", "optic nerve head", "optic nerve"],
    "macular": ["macula lutea", "macula of retina", "fovea"],
}


def main() -> None:
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
            best_mask, best_conf, best_prompt = None, -1.0, ""
            for p in prompts:
                try:
                    m = sam3.predict_text(state, p)
                except Exception as e:  # noqa: BLE001
                    print(f"  {stem}/{label} prompt={p!r} ERROR {e}")
                    continue
                if m is None or not m.any():
                    continue
                conf = sam3.get_confidence(state)
                if conf > best_conf:
                    best_mask, best_conf, best_prompt = m, conf, p
            if best_mask is None:
                blank = Image.fromarray(np.zeros((H, W), np.uint8))
                blank.save(out_path)
                print(f"[{i+1}/{len(images)}] {stem}/{label}: 无预测")
                continue
            mask = resize_mask(best_mask, (H, W))
            Image.fromarray((mask * 255).astype(np.uint8)).save(out_path)
            frac = float(mask.mean())
            print(f"[{i+1}/{len(images)}] {stem}/{label}: prompt={best_prompt!r} "
                  f"conf={best_conf:.2f} 占比={frac:.3f}")


if __name__ == "__main__":
    main()
