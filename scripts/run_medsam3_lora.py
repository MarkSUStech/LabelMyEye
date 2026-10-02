"""E003 runner: MedSAM3 (Joey-S-Liu, SAM3+LoRA) text-prompted inference.

底座不用 gated 的 facebook/sam3——用 1038lab/sam3 转存的 sam3_base.pt
（monkeypatch build_sam3_image_model 强制 load_from_HF=False 并加载本地底座），
再叠加 MedSAM3 的 best_lora_weights.pt。

用法（在 ~/labelmyeye_bench, 已激活 venv）:
    CUDA_VISIBLE_DEVICES=0 python run_medsam3_lora.py
输出: out/<stem>__<label>.png (255=前景)，与 benchmark_models.py 对接。
"""

import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

WS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(WS, "MedSAM3")
os.chdir(REPO)  # bpe_path 等在仓库内是相对路径
sys.path.insert(0, REPO)
sys.path.insert(0, REPO)  # sam3 包与 lora_layers 都在仓库根

BASE = os.path.join(WS, "weights", "sam3_base.pt")
LORA = os.path.join(WS, "weights", "medsam3_lora.pt")
CONFIG = os.path.join(REPO, "configs", "full_lora_config.yaml")
OUT = os.path.join(WS, "out")

PROMPTS = {
    "optic": ["optic disc", "optic nerve head", "optic nerve"],
    "macular": ["macula lutea", "macula of retina", "fovea"],
}


def load_base_state(model, path):
    """复用 Medical-SAM3 的格式兼容逻辑加载 SAM3 底座。"""
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
    import infer_sam  # noqa: E402  (import 时即完成其模块级初始化)

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
        config_path=CONFIG,
        weights_path=LORA,
        detection_threshold=0.5,
        nms_iou_threshold=0.5,
    )

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
                    continue          # results['_image'] 等可视化辅助键
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
            Image.fromarray((mask.astype(np.uint8)) * 255).save(out_path)
            print(f"[{i+1}/{len(images)}] {stem}/{label}: conf={conf:.2f} "
                  f"占比={float(mask.mean()):.3f}")


if __name__ == "__main__":
    main()
