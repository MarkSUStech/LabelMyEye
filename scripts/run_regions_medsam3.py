"""E005 runner (MedSAM3): region text-prompted inference.

Labels: posterior_pole_boundary / central_inner.
输出: out_regions/medsam3/<stem>__<label>.png (255=前景), 已做最大连通域清理。
"""

import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

WS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(WS, "MedSAM3")
os.chdir(REPO)
sys.path.insert(0, REPO)

BASE = os.path.join(WS, "weights", "sam3_base.pt")
LORA = os.path.join(WS, "weights", "medsam3_lora.pt")
CONFIG = os.path.join(REPO, "configs", "full_lora_config.yaml")
OUT = os.path.join(WS, "out_regions", "medsam3")

PROMPTS = {
    "posterior_pole_boundary": ["posterior pole", "posterior pole of retina",
                                "posterior pole of the eye"],
    "central_inner": ["macular region", "posterior pole region",
                      "retinal center"],
}


def largest_component(mask):
    import cv2
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == best


def load_base_state(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    else:
        sd = ckpt
    sample = next(iter(sd), "")
    if "detector." in sample:
        sd = {k.replace("detector.", ""): v for k, v in sd.items() if "detector" in k}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"底座加载: missing={len(missing)} unexpected={len(unexpected)}")


def main() -> None:
    import infer_sam  # noqa: E402

    real_build = infer_sam.build_sam3_image_model

    def patched_build(**kwargs):
        kwargs["load_from_HF"] = False
        kwargs["checkpoint_path"] = None
        model = real_build(**kwargs)
        load_base_state(model, BASE)
        return model

    infer_sam.build_sam3_image_model = patched_build

    os.makedirs(OUT, exist_ok=True)
    infer = infer_sam.SAM3LoRAInference(
        config_path=CONFIG, weights_path=LORA,
        detection_threshold=0.3, nms_iou_threshold=0.5)  # 大区域: 放低阈值

    images = sorted(glob.glob(os.path.join(WS, "images", "*.png")))
    print(f"{len(images)} images")
    for i, img_path in enumerate(images):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        pil = Image.open(img_path).convert("RGB")
        W, H = pil.size
        for label, prompts in PROMPTS.items():
            out_path = os.path.join(OUT, f"{stem}__{label}.png")
            if os.path.isfile(out_path):
                continue
            try:
                results = infer.predict(img_path, prompts)
            except Exception as e:  # noqa: BLE001
                print(f"[{i+1}/{len(images)}] {stem}/{label}: ERROR {e}")
                continue
            best = None
            for idx, r in results.items():
                if str(idx).startswith("_"):
                    continue
                masks = r.get("masks")
                scores = r.get("scores")
                masks = [] if masks is None else list(masks)
                scores = [] if scores is None else list(scores)
                for mk, sc in zip(masks, scores):
                    if best is None or float(sc) > best[1]:
                        best = (mk, float(sc))
            if best is None:
                Image.fromarray(np.zeros((H, W), np.uint8)).save(out_path)
                print(f"[{i+1}/{len(images)}] {stem}/{label}: 无预测")
                continue
            mask, conf = best
            mask = np.squeeze(np.asarray(mask))
            if mask.shape != (H, W):
                m_img = Image.fromarray((mask > 0).astype(np.uint8) * 255)
                mask = np.asarray(m_img.resize((W, H), Image.NEAREST)) > 127
            comp = largest_component(mask.astype(np.uint8))
            if comp is None:
                Image.fromarray(np.zeros((H, W), np.uint8)).save(out_path)
                print(f"[{i+1}/{len(images)}] {stem}/{label}: 清理后为空")
                continue
            Image.fromarray((comp.astype(np.uint8)) * 255).save(out_path)
            print(f"[{i+1}/{len(images)}] {stem}/{label}: conf={conf:.2f} "
                  f"占比={float(comp.mean()):.3f}")


if __name__ == "__main__":
    main()
