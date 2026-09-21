"""Reconstruct an imaginary "eyeball terrain" from vessel annotations.

Pipeline (Poisson reconstruction / gradient-field integration):

  1. vessel mask (HRF manual1) -> vesselness weight v (thick vessels weigh more)
  2. structure tensor of the mask -> local vessel TANGENT t-hat and
     coherence c (0..1, how confident the orientation is)
  3. guidance gradient field  G = w * c * (-n-hat)          (descend into vessel)
                              + beta * w * t-hat            (flow along vessel)
  4. fast Neumann-Poisson solve  laplace(h) = div(G)  via DCT (O(n log n))
  5. add a parabolic dome centred on the optic-disc proxy (vessel-density peak)
     -> vessels radiate downhill from the disc like river valleys on a dome
  6. render: shaded-relief map + 3D surface

Usage:
  python scripts/reconstruct_vessel_terrain.py 01_dr 01_h
  python scripts/reconstruct_vessel_terrain.py 03_g --grid 512

Outputs to output/vessel_terrain/: <case>_terrain_3d.png, <case>_terrain_map.png,
<case>_height.npy.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.fft import dctn, idctn
from scipy.ndimage import gaussian_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.abspath(os.path.join(
    ROOT, "..", "..", "..", "Dataset", "HRF", "all"))
OUT = os.path.join(ROOT, "output", "vessel_terrain")


# ------------------------------------------------------------------ io utils

def _find(d: str, stem: str) -> str | None:
    for ext in (".tif", ".tiff", ".png", ".gif", ".jpg", ".JPG"):
        p = os.path.join(d, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def load_gray(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0


def block_reduce(a: np.ndarray, f: int) -> np.ndarray:
    H, W = a.shape
    H2, W2 = H - H % f, W - W % f
    return a[:H2, :W2].reshape(H2 // f, f, W2 // f, f).mean(axis=(1, 3))


# ------------------------------------------------------- field construction

def vessel_fields(mask: np.ndarray, sigma_ridge: float = 2.0,
                  sigma_ct: float = 5.0):
    """Vesselness weight, vessel tangent/normal unit fields and coherence."""
    m = gaussian_filter(mask, 1.0)
    m = m / (m.max() + 1e-12)
    gx = gaussian_filter(m, sigma_ridge, order=(0, 1))
    gy = gaussian_filter(m, sigma_ridge, order=(1, 0))
    jxx = gaussian_filter(gx * gx, sigma_ct)
    jyy = gaussian_filter(gy * gy, sigma_ct)
    jxy = gaussian_filter(gx * gy, sigma_ct)
    theta_n = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)   # dominant normal dir
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2) / (jxx + jyy + 1e-9)
    nx, ny = np.cos(theta_n), np.sin(theta_n)          # normal (across vessel)
    tx, ty = -ny, nx                                   # tangent (along vessel)
    # vesselness: blurred mask, thick vessels stronger
    v = gaussian_filter(mask, 2.5)
    v = np.clip(v / (np.percentile(v, 99.5) + 1e-12), 0.0, 1.0)
    return v, nx, ny, tx, ty, np.clip(coh, 0.0, 1.0)


def poisson_neumann(div: np.ndarray) -> np.ndarray:
    """Solve laplace(h) = div with homogeneous Neumann BC via DCT."""
    M, N = div.shape
    k = np.arange(M).reshape(-1, 1)
    l = np.arange(N).reshape(1, -1)
    lam = 4.0 * (np.sin(np.pi * k / (2 * M)) ** 2
                 + np.sin(np.pi * l / (2 * N)) ** 2)
    lam[0, 0] = np.inf                       # drop the zero (mean) mode
    H = dctn(div, norm="ortho") / lam
    H[0, 0] = 0.0
    return idctn(H, norm="ortho")


def disc_proxy(mask: np.ndarray, fov: np.ndarray) -> tuple[int, int]:
    """Optic-disc proxy: vessel-density peak within the central FOV region."""
    dens = gaussian_filter(mask, 15.0)
    ys, xs = np.nonzero(fov > 0.5)
    cy, cx = int(ys.mean()), int(xs.mean())
    r_fov = np.sqrt(fov.sum() / np.pi)
    keep = np.zeros_like(dens, dtype=bool)
    keep[int(cy - 0.55 * r_fov):int(cy + 0.55 * r_fov),
         int(cx - 0.55 * r_fov):int(cx + 0.55 * r_fov)] = True
    keep &= fov > 0.5
    dens = np.where(keep, dens, 0.0)
    y, x = np.unravel_index(int(np.argmax(dens)), dens.shape)
    return int(x), int(y)


# ----------------------------------------------------------------- pipeline

def build_terrain(mask: np.ndarray, fov: np.ndarray, grid: int = 448,
                  flow: float = 0.25, ravine: float = 0.55,
                  dome: float = 1.0):
    """mask/fov in [0,1] at full resolution -> height field on ~grid."""
    f = max(1, round(max(mask.shape) / grid))
    m = block_reduce(mask, f)
    fov_s = block_reduce(fov, f) > 0.5
    m = m * fov_s

    v, nx, ny, tx, ty, coh = vessel_fields(m)
    w = v ** 1.5
    gx = -w * coh * nx + flow * w * tx     # descend into vessel + flow along
    gy = -w * coh * ny + flow * w * ty
    gx *= fov_s
    gy *= fov_s
    div = np.gradient(gx, axis=1) + np.gradient(gy, axis=0)
    h0 = poisson_neumann(div)
    h0 = gaussian_filter(h0, 2.0)
    lo, hi = np.percentile(h0[fov_s], [2, 98])
    h0 = np.clip((h0 - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    fov_core = ndimage.binary_erosion(fov_s, iterations=6)  # FOV 边界离群值
    rim_w = gaussian_filter(fov_core.astype(float), 3.0)    # 羽化, 避免硬边
    h0 *= np.clip(rim_w, 0.0, 1.0)
    h0 = gaussian_filter(h0, 1.0)
    h0 *= fov_s

    cx, cy = disc_proxy(m, fov_s)
    yy, xx = np.mgrid[0:m.shape[0], 0:m.shape[1]]
    r = np.hypot(xx - cx, yy - cy)
    r_max = float(r[fov_s].max())          # FOV 内离视盘最远点
    # 圆形穹顶延伸到 1.35*r_max(边界处仍约 0.4 高), 再由宽羽化窗在约 90px
    # 内缓慢降到 0 —— 等高线圆润, 且不会在边界形成陡壁
    dome_h = dome * np.clip(1.0 - (r / (1.35 * r_max)) ** 2.2, 0.0, 1.0)
    # 乘法雕刻: 谷深按当地穹顶高度的比例下切, 边界处穹顶→0, 不会挖出护城河
    h = dome_h * (1.0 - ravine * h0)
    rim = ndimage.binary_erosion(fov_s, iterations=40).astype(float)
    rim = np.clip(gaussian_filter(rim, 16.0), 0.0, 1.0)
    h *= rim
    h[~fov_s] = np.nan
    h -= np.nanmin(h)
    h /= np.nanmax(h)
    meta = {"disc": (cx, cy), "factor": f}
    return h, (m, v, nx, ny, tx, ty, coh, fov_s), meta


# ---------------------------------------------------------------- rendering

def render(case: str, img_full, h, aux, meta, out_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    m, v, nx, ny, tx, ty, coh, fov_s = aux
    ls = LightSource(azdeg=315, altdeg=45)
    hh = np.nan_to_num(h, nan=np.nanmin(h))

    # ---- 2D shaded-relief map with vessels + flow directions ----
    fig, ax = plt.subplots(figsize=(9, 9 * h.shape[0] / h.shape[1]))
    rgb = ls.shade(hh, cmap=plt.cm.terrain, vert_exag=2.5, blend_mode="soft")
    rgb[~fov_s, :3] = 0.05
    rgb[~fov_s, 3] = 1.0
    ax.imshow(rgb)
    ys, xs = np.nonzero((m > 0.25) & fov_s)
    ax.scatter(xs, ys, s=0.2, c="deepskyblue", alpha=0.45, linewidths=0)
    step = 14
    yy, xx = np.mgrid[step // 2:h.shape[0]:step, step // 2:h.shape[1]:step]
    sel = fov_s[yy, xx] & (v[yy, xx] > 0.12)
    ax.quiver(xx[sel], yy[sel], tx[yy, xx][sel], -ty[yy, xx][sel],
              color="k", alpha=0.5, scale=60, width=0.0022)
    cx, cy = meta["disc"]
    ax.plot(cx, cy, "r*", ms=16, mec="w", mew=0.8)
    ax.text(cx + 12, cy + 6, "disc", color="w", fontsize=10, weight="bold")
    ax.set_title(f"{case} - vessel-guided terrain (shaded relief)\n"
                 "arrows: vessel tangent field, blue: vessels, *: disc proxy",
                 fontsize=10)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{case}_terrain_map.png"), dpi=170)
    plt.close(fig)

    # ---- 3D surface ----
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")
    st = 2
    X, Y = np.meshgrid(np.arange(0, h.shape[1]), np.arange(0, h.shape[0]))
    Z = np.where(fov_s, hh, np.nan)
    rgba = ls.shade(hh, cmap=plt.cm.terrain, vert_exag=2.5, blend_mode="soft")
    rgba = rgba.copy()
    rgba[~fov_s, 3] = 0.0
    ax.plot_surface(X[::st, ::st], Y[::st, ::st], Z[::st, ::st],
                    facecolors=rgba[::st, ::st], rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
    ax.set_zlim(np.nanmin(Z) - 0.3, np.nanmax(Z) + 0.1)
    ax.view_init(elev=42, azim=-63)
    ax.set_axis_off()
    ax.set_box_aspect((h.shape[1], h.shape[0], 0.55 * h.shape[1]))
    ax.set_title(f"{case} - imaginary eyeball terrain reconstructed from "
                 "vessel orientation (Poisson integration)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{case}_terrain_3d.png"), dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------- main

def process(case: str, data_root: str, out_dir: str, grid: int):
    img_p = _find(os.path.join(data_root, "images"), case)
    mask_p = _find(os.path.join(data_root, "manual1"), case)
    fov_p = _find(os.path.join(data_root, "mask"), case)
    if mask_p is None:
        raise SystemExit(f"vessel mask not found for {case} in {data_root}")
    img = load_gray(img_p) if img_p else None
    mask = (load_gray(mask_p) > 0.5).astype(np.float64)
    fov = load_gray(fov_p) if fov_p else np.ones_like(mask)
    h, aux, meta = build_terrain(mask, fov, grid=grid)
    np.save(os.path.join(out_dir, f"{case}_height.npy"), h)
    render(case, img, h, aux, meta, out_dir)
    print(f"{case}: grid {h.shape[1]}x{h.shape[0]}, disc~{meta['disc']}, "
          f"saved to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="+", help="e.g. 01_dr 01_h 03_g")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--grid", type=int, default=448)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for case in args.cases:
        process(case, args.data, args.out, args.grid)


if __name__ == "__main__":
    sys.exit(main())
