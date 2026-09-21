"""One-click SAM model setup: download (or locally export) the ONNX models.

Download sources are tried in order; edit MODEL_BASE_URLS if you host the
models elsewhere (e.g. a GitHub Release of your fork). If all downloads fail
but this machine has PyTorch + mobile_sam installed, `local_export_cmd`
returns the command to export the models locally instead.
"""

from __future__ import annotations

import os
import sys
import urllib.request

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "assets", "models")
FILES = ("mobile_sam_encoder.onnx", "mobile_sam_decoder.onnx")

# 下载源按顺序尝试（%s 会被替换为文件名）。推送到 GitHub 后，建议把两个 onnx
# 上传到仓库的 Release（tag 名 models），并把第一条改成你的仓库地址。
MODEL_BASE_URLS = [
    "https://github.com/LabelMyEye/LabelMyEye/releases/download/models-v1",
]


def models_present() -> bool:
    return all(os.path.isfile(os.path.join(MODEL_DIR, f)) for f in FILES)


def missing_files() -> tuple[str, ...]:
    return tuple(f for f in FILES if not os.path.isfile(os.path.join(MODEL_DIR, f)))


def local_export_cmd() -> list[str] | None:
    """若本机装有 PyTorch + mobile_sam, 返回本地导出命令; 否则 None。"""
    try:
        import torch  # noqa: F401
        script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "scripts", "prepare_sam_onnx.py")
        return [sys.executable, script] if os.path.isfile(script) else None
    except Exception:  # noqa: BLE001
        return None


def download_models(progress=None) -> None:
    """下载缺失的 onnx 到 MODEL_DIR。progress: (done_files, total, file_pct or None)。

    progress 为 None 时静默下载。任一文件所有源都失败则抛出 RuntimeError。
    """
    missing = missing_files()
    if not missing:
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    total = len(missing)
    for i, fname in enumerate(missing):
        dest = os.path.join(MODEL_DIR, fname)
        last_err: Exception | None = None
        for base in MODEL_BASE_URLS:
            url = f"{base}/{fname}"
            try:
                if progress is not None:
                    progress(i, total, None)

                def hook(blocks: int, bs: int, total_size: int) -> None:
                    if progress is not None and total_size > 0:
                        frac = (blocks * bs) / total_size
                        progress(i, total, min(1.0, frac))

                urllib.request.urlretrieve(url, dest + ".part", reporthook=hook)
                os.replace(dest + ".part", dest)
                last_err = None
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                if os.path.exists(dest + ".part"):
                    try:
                        os.remove(dest + ".part")
                    except OSError:
                        pass
        if last_err is not None or not os.path.isfile(dest):
            raise RuntimeError(
                f"下载 {fname} 失败：{last_err}\n"
                f"请检查网络，或手动将文件放入 assets/models/ 后重试。")


def models_dir() -> str:
    return MODEL_DIR
